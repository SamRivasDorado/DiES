import time
import sys
import os
import gc
import numpy as np
from numpy import ma
import scipy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)

#########################################################

# Obtain the classes from FracnSlip
from DiES_core import crack
from DiES_core import model
from DiES_core import require_bool
from DiES_core import InputError

# Results containers and figures (Phase 1, Nov 2026). The figures module is
# imported as dies_plotting because dynamic_model has an argument called plotting
from results import DiESResults, DynamicResults, FAILED_MODES
import plotting as dies_plotting

# Obtain the function to evaluate the stress state (evaluate_stress_states: the
# version for arrays, used by compute_dynamic_model since Phase 2)
from Failure_Determination import evaluate_stress_state
from Failure_Determination import evaluate_stress_states
from Failure_Determination import failure_contact
from Failure_Determination import failure_plane_orientation
from Failure_Determination import FAILURE_TOLERANCE

# Obtaining helper functions
from Helper_functions import create_figure
from Helper_functions import points_distance
from Helper_functions import boundaries_contours
from Helper_functions import true_min_max
from Helper_functions import downsample_2darray
from Helper_functions import G
from Helper_functions import normalize_lengths_graben
from Helper_functions import normalize_elevations
from Helper_functions import compare_outputs_and_faults

#########################################################

########################### Defining standard parameters ########################### 

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', title_fontsize=11)    # legend title fontsize
plt.rc('legend', fontsize=11)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams.update({'figure.max_open_warning': 0})


def layered_inputs(model_inputs, layer_tops=None):
    """
    Layers of single_model, dynamic_model and compute_dynamic_model (multilayer since
    Nov 2026, Phase 2). Converts the user's description, from the top down, into the
    inputs of model(), which takes the layer boundaries in ascending order and the
    values from the deepest layer up.
    
    Parameters
    ----------
    model_inputs : 5 values, (E, v, density, friction, T). Each one is a number (the same
                   for every layer) or, with layer_tops, a list with one value per
                   layer, from the top down.
    layer_tops : None for a single-layer model, or the y of the top of each layer, in m,
                 from the top down (strictly decreasing, e.g. [0, -3000]): each layer
                 goes from its top down to the top of the next one, and the last one
                 down to the base of the model and below. The uppermost layer is taken
                 to reach the surface (y = 0) for the lithostatic load.
    
    Returns
    -------
    y_position_layers, E, v, density, friction, T : the inputs of model() (the boundaries and
    the values of each layer, from the deepest up). For a single layer, the values are
    numbers and y_position_layers is 0 (any single value means one layer).
    """
    if len(model_inputs)!=5:
        raise InputError('model_inputs must have 5 values (E, v, density, friction, T), got '
                         +str(len(model_inputs))+'. The cohesion is no longer an input: it is implicit as C = 2|T|')
    
    # Single layer: every value must be a number, as until Nov 2026
    if layer_tops is None:
        if any(np.ndim(value)>0 for value in model_inputs):
            raise InputError('model_inputs has lists of values (one per layer), but layer_tops is not given: '
                             'pass the top of each layer, from the top down')
        return (0,)+tuple(model_inputs)
    
    tops=np.atleast_1d(np.asarray(layer_tops,dtype=float))
    n=tops.size
    if n>1 and np.any(np.diff(tops)>=0):
        raise InputError('layer_tops must be strictly decreasing (the tops of the layers from the top down), got '
                         +str(list(tops)))
    
    # Each value: one number for every layer, or one value per layer (from the top down)
    names=('E','v','density','friction','T')
    values=[]
    for name,value in zip(names,model_inputs):
        value=np.atleast_1d(np.asarray(value,dtype=float))
        if value.size==1:
            value=np.repeat(value,n)
        elif value.size!=n:
            raise InputError(name+' has '+str(value.size)+' values, but layer_tops defines '+str(n)+' layer(s)')
        # model() wants the deepest layer first
        values.append(value[::-1])
    
    # A single top: one layer, with plain numbers
    if n==1:
        return (0,)+tuple(float(v[0]) for v in values)
    
    # Ascending boundaries for model(): layer i spans [bounds[i], bounds[i+1]). The first
    # one only has to be below the top of the deepest layer (model() assigns everything
    # below it to the deepest layer); the others are the tops, from the deepest layer up
    bounds=np.concatenate(([tops[-1]-1.0],tops[::-1]))
    return (bounds,)+tuple(values)


def crack_layer_properties(current_model, upper_tip, lower_tip):
    """
    E and v that relate the aperture (or the slip) of a crack and its driving stress:
    those of the layer that contains the centre of the crack (Sam's choice, Nov 2026).
    If the tips are in different layers, a warning says that the relation is then
    approximate (the analytical solution is that of a homogeneous medium).
    """
    import warnings
    centre_y=(upper_tip[1]+lower_tip[1])/2
    E=np.atleast_1d(np.asarray(current_model.layers_E,dtype=float))
    v=np.atleast_1d(np.asarray(current_model.layers_v,dtype=float))
    index=int(current_model.layer_index(centre_y))
    if current_model.layer_index(upper_tip[1])!=current_model.layer_index(lower_tip[1]):
        warnings.warn('The crack crosses a layer boundary: its aperture (or slip) and its driving stress are '
                      'related with the E and v of the layer at its centre (y = '+str(centre_y)+' m)')
    return float(E[index]), float(v[index])


def single_model(crack_inputs, model_inputs, dx, dy, dz, rx,ry,g,litho,angle,xlims,ylims,resolution,parameter,
                 layer_tops=None):
    """

    Parameters
    ----------
    crack_inputs :  list-like, contains the crack parameters,
                    crack total aperture, as a float/int in m, measured
                    at the center of the crack between its two walls.
                    If the model is of a fault, then aperture should be 0,
                    crack upper tip x coordinate,
                    crack upper tip y coodintate,
                    crack lower tip x coordinate,
                    crack lower tip y coodintate,
                    where the coordinates should be consistent with
                    the x and y values used in xlims and ylims.
                    The total height of the crack, b, is the distance
                    between its two tips and is calculated from them.
    model_inputs :  list-like, contains the model parameters (each one a number,
                    for a single layer or the same in every layer, or, with
                    layer_tops, a list with one value per layer from the top
                    down; multilayer since Nov 2026):
                    Young's modulus as a float/int in Pa,
                    Poissons ratio as an float/int between 0-0.5,
                    density as an float/int in kgm^3,
                    friction, the coefficient of internal friction, as an float/int,
                    tension, tensile strength as an float/int in Pa, negative
                    (tension is negative on the Mohr diagram), e.g. -1.5e6.
                    The cohesion is not an input: it is implicit in the
                    modified Griffith criterion as C = 2|T|.
    
    dx : driving stress orthogonal to the crack wall as an float/int, in Pa.
    Only used if the crack aperture is 0 (a fault), because otherwise it is
    calculated from the aperture. >0 values model a dilatant fault,
    and <0 values model a compaction band
    dy : driving stress parallel to the crack wall as an float/int, in Pa.
    >0 values make the hanging wall slide down-dip (normal sense) and <0
    values up-dip (reverse sense). For a vertical crack, >0 makes the block
    at x > 0 move down. See docs/crack_coordinates_and_rotation.md
    dz : out-of-plane driving stress as an float/int, in Pa, usually 0
    rx : constant remote stress along the X axis as an float/int, in Pa
    ry : constant remote stress along the Y axis as an float/int, in Pa
    g : gravitational acceleration as a float/int in ms^-2
    litho : a bool, True to take into account the lithostatic
    stresses in the calculations, False, to not do it. This has no effect
    on the displacements. It must be a real boolean, not a str.
    angle : angle at which the resolve the principal stresses, if plotting 'sn'
    or 'ss', as an int/float between 0-90
    
    xlims : list-like, X-boundaries of the model, can be any range
    ylims : list-like, Y-boundaries of the model, needs to be in the form (-y,0)
    resolution : list-like, two values indicating cell-size along the X and Y axes  
    downscaling: int, a value to scale up or down the frequency with which
    the discontinuities are plotted. The smaller the cellsize, the larger the 
    downscaling factor should be.

    layer_tops : None (a single layer, as until Nov 2026), or the y of the top
    of each layer in m, from the top down, e.g. [0, -3000] for two layers
    (see layered_inputs). With several layers, the aperture of a dike is
    converted into its driving stress with the E and v of the layer at the
    centre of the crack (a warning says so if the crack crosses a boundary).
    parameter : a str, key for the parameter that one desires to plot. It can be;
                'ux': displacements along the x axis
                'uy': displacements along the y axis
                'uxy': combined xy displacement
                
                The crack stresses are shown in tension-positive convention
                'sx': stress along the x axis
                'sy':stress along the y axis
                'sxy': xy stress
                'sav': average stress
                't': shear stress
                
                The principal stresses are shown in compression-positive convention
                's1': maximum principal stress, sigma1
                's3': minimum principal stress, sigma3
                'st': s1 and s3 stress trajectories
                
                'sn': normal stresses when resolving s1-s3 onto planes of 
                the specified angle
                'ss':shear stresses when resolving s1-s3 onto planes of 
                the specified angle

    Returns
    -------
    A figure object
    """
    
    
    # Defining variables from crack_inputs
    a = crack_inputs[0] ; 
    upper_crack_tip_x = crack_inputs[1] ;  upper_crack_tip_y = crack_inputs[2]
    lower_crack_tip_x = crack_inputs[3] ;  lower_crack_tip_y = crack_inputs[4]
    
    # b is the total height of the crack: the distance between its two tips
    b = points_distance((upper_crack_tip_x,upper_crack_tip_y), (lower_crack_tip_x,lower_crack_tip_y))
    
    # Layers (one or several) and their properties, as model() takes them
    layers, young, poisson, density, friction, tension = layered_inputs(model_inputs, layer_tops)
    
    # litho must be a real boolean, not a str
    litho=require_bool(litho,'litho')
    
    # Preparing the model: a coarser grid for the vectors and the trajectories
    if parameter=='ux' or parameter=='uy' or parameter=='uxy' or parameter=='st':
        current_model=model(xlims,ylims,resolution[0]*2.5,resolution[1]*3,layers,young,poisson,density,friction,tension,g)
    else:
        current_model=model(xlims,ylims,resolution[0],resolution[1],layers,young,poisson,density,friction,tension,g)
    
    # E and v that relate the aperture and the driving stress: those of the layer at
    # the centre of the crack (the only layer, in a single-layer model).
    # shear is the shear modulus, G. It must not be confused with the
    # coefficient of internal friction (friction; named cfi until Nov 2026)
    crack_E, poisson = crack_layer_properties(current_model, (upper_crack_tip_x,upper_crack_tip_y),
                                              (lower_crack_tip_x,lower_crack_tip_y))
    shear = G(crack_E, poisson)
    
    if a==0:
        # A fault: the driving stress orthogonal to the crack is an input
        px=dx
        disc=crack((upper_crack_tip_x,upper_crack_tip_y),(lower_crack_tip_x,lower_crack_tip_y),dx,dy,dz)
    else:
        # An opening fracture: the driving stress orthogonal to the crack
        # follows from its total aperture, a, and its total height, b.
        # It reproduces the opening profile of crack.mode_I_crack_displacements
        # (Eq. 7.21 in Pollard and Martel, 2020)
        px = a*shear/(b*(1-poisson))
        # Creating the crack object
        disc=crack((upper_crack_tip_x,upper_crack_tip_y),(lower_crack_tip_x,lower_crack_tip_y),px,dy,dz)
       
    
    grid=current_model.grid()

    # The contour steps are chosen automatically from the values (step None;
    # Nov 2026). Until then each parameter had a fixed step (0.25-5 MPa or m),
    # too dense for the stresses of deep models with Earth gravity.
    if parameter=='ux':
        plot=current_model.plot_displacement(disc, 'ux', 'vectors', None)
    elif parameter=='uy':
        plot=current_model.plot_displacement(disc, 'uy', 'vectors', None)    
    elif parameter=='uxy':
        plot=current_model.plot_displacement(disc, 'uxy', 'vectors', None)
    
    elif parameter=='sx':
        plot=current_model.plot_stresses(disc,'sxa',rx, ry,litho, None)
    elif parameter=='sy':
        plot=current_model.plot_stresses(disc,'sya',rx, ry,litho, None)
    elif parameter=='sxy':
        plot=current_model.plot_stresses(disc,'sxya',rx, ry,litho, None)
    elif parameter=='sav': 
        plot=current_model.plot_stresses(disc,'sav',rx, ry,litho, None)
    elif parameter=='t':
        plot=current_model.plot_stresses(disc,'ta',rx, ry,litho, None)
    
    elif parameter=='s1':
        plot=current_model.plot_principal_stresses(disc, 'principal',rx, ry, litho, None)
    elif parameter=='s3':
        plot=current_model.plot_principal_stresses(disc, 'secondary',rx, ry, litho, None)
    elif parameter=='st':
        plot=current_model.plot_stress_trajectories(disc,False, rx, ry,litho)
    
    elif parameter=='sn':
        plot=current_model.plot_plane_stresses(disc,'normal',rx,ry,angle,litho, None)
    elif parameter=='ss':
        plot=current_model.plot_plane_stresses(disc,'shear',rx,ry,angle,litho, None)
   
    return plot
    
################ A function to run a dynamic model of the opening or slip of a crack ################

def compute_dynamic_model(crack_inputs,model_inputs,dx,dy,dz,rx,ry,g,litho,investigation_range,xlims,ylims,resolution,
                          layer_tops=None):
    """
    Computes the dynamic model step by step, without drawing (Phase 1, Nov 2026).
    
    Same inputs as dynamic_model, without downscaling, plotting and folder.
    model_inputs and layer_tops as in single_model: one layer, or several
    (since Nov 2026).
    The driving stresses grow with investigation_range; at each step the stress
    state of every node is evaluated with evaluate_stress_state, and the first
    failure found at a node is kept in all the following steps. Since Nov 2026
    (Phase 2, point 3A) each step evaluates the whole grid in one call
    (evaluate_stress_states, failure_plane_orientation with arrays) instead of
    a double loop over rows and nodes; the results are identical.
    Since point 3B the criterion is analytical (see evaluate_stress_state), and
    a node that jumps from 'Below' to 'Above' between two steps keeps the mode,
    stresses and plane of the point where its Mohr circle touched the envelope;
    those nodes are marked with failure_jumped = 1.
    
    Returns
    -------
    A DynamicResults object (results.py) with one DiESResults per step, holding
    failure_sn and failure_ss (MPa), failure_theta (deg), failure_mode,
    above_failure and the orientations of the failure planes (plane_*, radians
    counterclockwise from +x), plus the metadata of the step. Draw a step with
    plotting.plot_dynamic_step.
    """    

    # Defining variables from crack_inputs
    a = crack_inputs[0] ; 
    upper_crack_tip_x = crack_inputs[1] ;  upper_crack_tip_y = crack_inputs[2]
    lower_crack_tip_x = crack_inputs[3] ;  lower_crack_tip_y = crack_inputs[4]
    
    # b is the total height of the crack: the distance between its two tips
    b = points_distance((upper_crack_tip_x,upper_crack_tip_y), (lower_crack_tip_x,lower_crack_tip_y))
    
    # Layers (one or several) and their properties, as model() takes them
    layers, young, poisson, density, friction, tension = layered_inputs(model_inputs, layer_tops)
    
    # litho must be a real boolean, not a str
    litho=require_bool(litho,'litho')
    
    # Creating the model
    current_model=model(xlims,ylims,resolution[0],resolution[1],layers,young,poisson,density,friction,tension,g)
    
    # E and v that relate the aperture (or slip) and the driving stress: those of the
    # layer at the centre of the crack (the only layer, in a single-layer model).
    # shear is the shear modulus, G. It must not be confused with the
    # coefficient of internal friction (friction; named cfi until Nov 2026)
    crack_E, poisson = crack_layer_properties(current_model, (upper_crack_tip_x,upper_crack_tip_y),
                                              (lower_crack_tip_x,lower_crack_tip_y))
    shear = G(crack_E, poisson)
    
    # Final driving stress orthogonal to the crack, and final displacement
    # of the crack walls, i.e., the values reached at 100% of the model
    if a==0:
        # A fault, which may be dilatant (dx>0), a compaction band (dx<0),
        # or purely shearing (dx=0)
        dx_final=dx
        
        # EQUATION TO BE CHECKED
        # Note (Nov 2026): this has the antiplane (mode III) form. For a mode II
        # crack in plane strain a factor (1-v) is expected, i.e.
        # dy*b*(1-v)/shear. Left unchanged for now; it only affects the
        # final slip shown in the legend, not the stresses.
        length_final = dy*b/shear
        length_label = 'u'

    else:
        # An opening fracture: the driving stress orthogonal to the crack
        # follows from its total aperture, a, and its total height, b.
        # It reproduces the opening profile of crack.mode_I_crack_displacements
        # (Eq. 7.21 in Pollard and Martel, 2020). The input dx is not used.
        dx_final = a*shear/(b*(1-poisson))
        length_final = a
        length_label = 'a'
        
    # All the driving stresses grow together with each step of the model,
    # each one by the percentage of its step
    dxs_to_use=[] ; dys_to_use=[] ; dzs_to_use=[] ; lengths_to_use=[]

    for percentage in investigation_range:
        dxs_to_use.append(dx_final*percentage)
        dys_to_use.append(dy*percentage)
        dzs_to_use.append(dz*percentage)
        lengths_to_use.append(length_final*percentage)
     
    grid=current_model.grid()
    x, y, = grid[0], grid[1]
        
    # The failure criterion takes the friction and T of each node (they change with the layer)
    node_friction, node_T = grid[6], grid[8]
    
    # Results of the previous step, to keep the first failure of each node
    # (None before the first step)
    previous=None
    
    # Failure modes before the crack is loaded (remote and lithostatic stresses
    # only): a node that is already 'Above' then did not fail because of the
    # crack, and is not taken as having jumped over the failure band at step 1
    unloaded=crack((upper_crack_tip_x,upper_crack_tip_y),(lower_crack_tip_x,lower_crack_tip_y),0,0,0)
    s1_0, s3_0, _, _ = current_model.principal_stresses(unloaded, rx, ry, litho)
    previous_mode=evaluate_stress_states(s1_0, s3_0, node_friction, node_T)[3]
    del unloaded
    
    steps=[]
    step=0
    
    for step_dx, step_dy, step_dz, length, percentage in zip(dxs_to_use, dys_to_use, dzs_to_use, lengths_to_use, investigation_range):
        step=step+1
        
        # Create the crack with the driving stresses of this step
        disc=crack((upper_crack_tip_x,upper_crack_tip_y),(lower_crack_tip_x,lower_crack_tip_y),step_dx,step_dy,step_dz)
        
        upper_tip=disc.up_tip() ; crack_center=disc.c() ; lower_tip=disc.low_tip()
        
        # Calculating the principal stresses and the orientation of s1, counterclockwise from the x axis
        s1, s3, s1a, s3a = current_model.principal_stresses(disc, rx, ry, litho)
        
        # The stress state of every node of the grid is evaluated in one call
        # (Phase 2, point 3A, Nov 2026). Until then this was a double loop over
        # rows and nodes, with one call to evaluate_stress_state per node.
        sn, ss, two_theta, mode = evaluate_stress_states(s1, s3, node_friction, node_T)
        
        # Nodes that jumped over the failure band since the previous step (from
        # 'Below' to 'Above': the driving stress grew too much in one step).
        # They get the failure mode and the stresses of the point where their
        # Mohr circle touched the envelope (the envelope point: the best estimate
        # of the stresses on the plane at the moment of failure) (Phase 2, point 3B, Nov 2026; until
        # then they stayed 'Above', without mode or plane)
        jumped=(mode=='Above') & (previous_mode=='Below')
        if np.any(jumped):
            _, contact_sn, contact_ss, contact_two_theta, contact_mode = failure_contact(s1, s3, node_friction, node_T)
            sn=np.where(jumped, contact_sn, sn) ; ss=np.where(jumped, contact_ss, ss)
            two_theta=np.where(jumped, contact_two_theta, two_theta)
            mode=np.where(jumped, contact_mode, mode)
        
        # Obtaining theta from 2theta
        angle=two_theta/2
        
        # 0 where the circle is above failure, NaN elsewhere (used to draw those nodes)
        above=np.where(mode=='Above', 0.0, np.nan)
        
        # Failure planes. Left of the crack (x <= x of its center) side = -1,
        # right of it side = +1: the conjugate plane that dips towards the crack
        # is selected (failure_plane_orientation)
        side=np.where(x<=crack_center[0], -1, 1)
        plane=failure_plane_orientation(s1a, angle, mode, side)
        left=side<0 ; right=~left
        
        # Each plane goes to the array of its mode and side, NaN elsewhere. The
        # Mode I plane (it contains sigma1) does not depend on the side.
        planes={'plane_mixed_left': (mode=='Mixed-mode I-II') & left,
                'plane_tc_left': (mode=='Mode II - tension-compression') & left,
                'plane_comp_left': (mode=='Mode II - compression') & left,
                'plane_mixed_right': (mode=='Mixed-mode I-II') & right,
                'plane_tc_right': (mode=='Mode II - tension-compression') & right,
                'plane_comp_right': (mode=='Mode II - compression') & right,
                'plane_mode_I': mode=='Mode I'}
        planes={name: np.where(selected, plane, np.nan) for name, selected in planes.items()}
        
        current={'failure_sn':sn, 'failure_ss':ss, 'failure_theta':angle, 'failure_mode':mode,
                 'above_failure':above, 'failure_jumped':np.where(jumped, 1.0, 0.0)}
        current.update(planes)
        
        # This ensures that the first tensile or shear failure values (or 'Above')
        # are preserved at each node for all the following timesteps
        if previous is not None:
            keep=np.isin(previous['failure_mode'], FAILED_MODES)
            current={name: np.where(keep, previous[name], values) for name, values in current.items()}
        
        # Storing the current results as the previous results for the next run
        previous=current
        previous_mode=current['failure_mode']
        
        # Storing the results of this step (Phase 1, Nov 2026: the figures are no
        # longer drawn inside this loop; see plotting.plot_dynamic_step)
        step_metadata=current_model.metadata(disc, rx, ry, litho)
        step_metadata.update({'step':step, 'n_steps':len(investigation_range), 'percentage':float(percentage),
                              'dx':float(step_dx), 'dy':float(step_dy), 'dz':float(step_dz),
                              'length':float(length), 'length_label':length_label})
        steps.append(DiESResults(x, y, current, step_metadata))
        
        del disc
        gc.collect()
    
    # Run metadata: model_inputs keeps a number per property (one layer) or a list per
    # layer (from the top down), and layer_tops is None for a single layer
    return DynamicResults(steps, {'model_inputs':[np.asarray(v,dtype=float).tolist() for v in model_inputs],
                                  'layer_tops':None if layer_tops is None else np.asarray(layer_tops,dtype=float).tolist(),
                                  'crack_inputs':[float(v) for v in crack_inputs],
                                  'investigation_range':[float(v) for v in investigation_range],
                                  'rx':float(np.mean(rx)), 'ry':float(np.mean(ry)), 'litho':litho, 'g':float(g),
                                  'failure_criterion':'analytical', 'failure_tolerance':FAILURE_TOLERANCE})


def dynamic_model(crack_inputs,model_inputs,dx,dy,dz,rx,ry,g,litho,investigation_range,xlims,ylims,resolution,downscaling,plotting,folder,
                  layer_tops=None):
    """

    Parameters
    ----------
    crack_inputs :  list-like, contains the crack parameters,
                    crack total aperture, as a float/int in m, measured
                    at the center of the crack between its two walls.
                    If the model is of a fault, then aperture should be 0,
                    crack upper tip x coordinate,
                    crack upper tip y coodintate,
                    crack lower tip x coordinate,
                    crack lower tip y coodintate,
                    where the coordinates should be consistent with
                    the x and y values used in xlims and ylims.
                    Values should be negative so the crack is placed in the subsurface.
                    The total height of the crack, b, is the distance
                    between its two tips and is calculated from them.
    model_inputs :  list-like, contains the model parameters (each one a number,
                    for a single layer or the same in every layer, or, with
                    layer_tops, a list with one value per layer from the top
                    down; multilayer since Nov 2026):
                    Young's modulus as a float/int in Pa,
                    Poissons ratio as an float/int between 0-0.5,
                    density as an float/int in kgm^3,
                    friction, the coefficient of internal friction, as an float/int,
                    tension, tensile strength as an float/int in Pa, negative
                    (tension is negative on the Mohr diagram), e.g. -1.5e6.
                    The cohesion is not an input: it is implicit in the
                    modified Griffith criterion as C = 2|T|.
    
    dx : final driving stress orthogonal to the crack wall as an float/int, in Pa.
    Only used if the crack aperture is 0 (a fault), because otherwise it is
    calculated from the aperture. It can be >0 (a dilatant fault), <0
    (a compaction band), or 0 (pure shearing).
    dy : final driving stress parallel to the crack wall as an float/int, in Pa.
    >0 values make the hanging wall slide down-dip (normal sense) and <0
    values up-dip (reverse sense). For a vertical crack, >0 makes the block
    at x > 0 move down. See docs/crack_coordinates_and_rotation.md
    dz : final out-of-plane driving stress as an float/int, in Pa, usually 0
    All three driving stresses are scaled together with each step of the model,
    so the values given here are those reached at 100% of the investigation range.
    rx : constant remote stress along the X axis as an float/int, in Pa
    ry : constant remote stress along the Y axis as an float/int, in Pa
    g : gravitational acceleration as a float/int in ms^-2
    litho: a bool, True to take into account the lithostatic
    stresses in the calculations, False, to not do it. It must be a real
    boolean, not a str.
    investigation_range :   list-like, set of X number of values between 0-1,
                            between which the evolution of the crack opening
                            or slip is to be modeled. The number of values in this
                            parameter will determine the steps of the model
    
    xlims : list-like, X-boundaries of the model
    ylims : list-like, Y-boundaries of the model
    resolution : list-like, two values indicating cell-size along the X and Y axes  
    plotting : a bool, True to create and save plots of the figures in each step,
    False, to not do it. It must be a real boolean, not a str.
    downscaling: int, a value to scale up or down the frequency with which
    the discontinuities are plotted. The smaller the cellsize, the larger the 
    downscaling factor should be. Typical values should be between 1 and 5.

    folder : path in which to save the figures, as a str
    layer_tops : None (a single layer, as until Nov 2026), or the y of the top
    of each layer in m, from the top down (see single_model and layered_inputs).

    Returns
    -------
    current_mode: a 2D array with the failure mode found at each cell of the
    grid at the last step of the model.
    If plotting is True, it also saves the figure of each of the modeled
    steps in the specified folder. Each figure shows a legend indicating
    the aperture (a) or slip (u), and the driving stresses of each step.
    By default, the figures plot the discontinuities and map the color of the
    normal stresses found for each discontinuity.

    """
    
    # plotting must be a real boolean, not a str (checked before computing)
    plotting=require_bool(plotting,'plotting')
    
    # Since Nov 2026 (Phase 1) the computation and the figures are separate:
    # compute_dynamic_model returns the results of every step, and
    # plotting.plot_dynamic_step draws them. The figures and the returned
    # array are the same as before.
    results=compute_dynamic_model(crack_inputs,model_inputs,dx,dy,dz,rx,ry,g,litho,
                                  investigation_range,xlims,ylims,resolution,layer_tops)
    
    if plotting:
        # E, G and T in the file names: those of the layer at the centre of the crack
        # (the only layer, in a single-layer model)
        centre_layer=model(xlims,ylims,resolution[0],resolution[1],*layered_inputs(model_inputs,layer_tops),g)
        index=int(centre_layer.layer_index((crack_inputs[2]+crack_inputs[4])/2))
        young=float(np.atleast_1d(centre_layer.layers_E)[index]) ; tension=float(np.atleast_1d(centre_layer.layers_T)[index])
        shear=G(young,float(np.atleast_1d(centre_layer.layers_v)[index]))
        color_range=dies_plotting.dynamic_color_range(results)
        for step_results in results:
            meta=step_results.metadata
            # The same colour range for every step: that of the final step (Nov 2026)
            figure=dies_plotting.plot_dynamic_step(step_results, downscaling, color_range)
            figure.savefig(os.path.join(folder,'sn'+'_'+str(round(shear/1e9,2))+'_'+str(young/1e9)+'_'+str(tension/1e6)
                                        +'_'+str(meta['step'])+'_'+str(round(meta['dx']/1e6,2))+'_MPa.png'),dpi=300)
            plt.close(figure)
            del figure
    
    return results.final_mode()
