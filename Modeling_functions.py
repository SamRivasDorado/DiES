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

# Obtain the function to evaluate the stress state
from Failure_Determination import evaluate_stress_state
from Failure_Determination import failure_plane_orientation

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


def single_model(crack_inputs, model_inputs, dx, dy, dz, rx,ry,g,litho,angle,xlims,ylims,resolution,parameter):
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
    model_inputs :  list-like, contains the model parameters for a
                    single-model layer.
                    Young's modulus as a float/int in Pa,
                    Poissons ratio as an float/int between 0-0.5,
                    density as an float/int in kgm^3,
                    mu, the coefficient of internal friction, as an float/int,
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
    
    # Defining variables from model_inputs
    if len(model_inputs)!=5:
        raise InputError('model_inputs must have 5 values (E, v, density, cfi, T), got '
                         +str(len(model_inputs))+'. The cohesion is no longer an input: it is implicit as C = 2|T|')
    young=model_inputs[0] ; poisson=model_inputs[1] ; density=model_inputs[2]
    cfi=model_inputs[3] ; tension=model_inputs[4]
    
    # litho must be a real boolean, not a str
    litho=require_bool(litho,'litho')

    # shear is the shear modulus, G. It must not be confused with the
    # coefficient of internal friction, cfi
    shear = G(model_inputs[0],model_inputs[1])
        
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
       
     # Preparing the model
    if parameter=='ux' or parameter=='uy' or parameter=='uxy' or parameter=='st':
         current_model=model(xlims,ylims,resolution[0]*2.5,resolution[1]*3,(200),(young),(poisson),(density),(cfi),(tension),g)
    
    else:
        current_model=model(xlims,ylims,resolution[0],resolution[1],(500),(young),(poisson),(density),(cfi),(tension),g)
    
    grid=current_model.grid()

    if parameter=='ux':
        plot=current_model.plot_displacement(disc, 'ux', 'vectors', 0.25)
    elif parameter=='uy':
        plot=current_model.plot_displacement(disc, 'uy', 'vectors', 0.50)    
    elif parameter=='uxy':
        plot=current_model.plot_displacement(disc, 'uxy', 'vectors', 0.25)
    
    elif parameter=='sx':
        plot=current_model.plot_stresses(disc,'sxa',rx, ry,litho,2.5)
    elif parameter=='sy':
        plot=current_model.plot_stresses(disc,'sya',rx, ry,litho,5)
    elif parameter=='sxy':
        plot=current_model.plot_stresses(disc,'sxya',rx, ry,litho,1)
    elif parameter=='sav': 
        plot=current_model.plot_stresses(disc,'sav',rx, ry,litho,5)
    elif parameter=='t':
        plot=current_model.plot_stresses(disc,'ta',rx, ry,litho,5)
    
    elif parameter=='s1':
        plot=current_model.plot_principal_stresses(disc, 'principal',rx, ry, litho, 5)
    elif parameter=='s3':
        plot=current_model.plot_principal_stresses(disc, 'secondary',rx, ry, litho, 2.5)
    elif parameter=='st':
        plot=current_model.plot_stress_trajectories(disc,False, rx, ry,litho)
    
    elif parameter=='sn':
        plot=current_model.plot_plane_stresses(disc,'normal',rx,ry,angle,litho,0.5)
    elif parameter=='ss':
        plot=current_model.plot_plane_stresses(disc,'shear',rx,ry,angle,litho,0.5)
   
    return plot
    
################ A function to run a dynamic model of the opening or slip of a crack ################

def dynamic_model(crack_inputs,model_inputs,dx,dy,dz,rx,ry,g,litho,investigation_range,xlims,ylims,resolution,downscaling,plotting,folder):
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
    model_inputs :  list-like, contains the model parameters for a
                    single-model layer.
                    Young's modulus as a float/int in Pa,
                    Poissons ratio as an float/int between 0-0.5,
                    density as an float/int in kgm^3,
                    mu, the coefficient of internal friction, as an float/int,
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

    # Defining variables from crack_inputs
    a = crack_inputs[0] ; 
    upper_crack_tip_x = crack_inputs[1] ;  upper_crack_tip_y = crack_inputs[2]
    lower_crack_tip_x = crack_inputs[3] ;  lower_crack_tip_y = crack_inputs[4]
    
    # b is the total height of the crack: the distance between its two tips
    b = points_distance((upper_crack_tip_x,upper_crack_tip_y), (lower_crack_tip_x,lower_crack_tip_y))
    
    # Defining variables from model_inputs
    if len(model_inputs)!=5:
        raise InputError('model_inputs must have 5 values (E, v, density, cfi, T), got '
                         +str(len(model_inputs))+'. The cohesion is no longer an input: it is implicit as C = 2|T|')
    young=model_inputs[0] ; poisson=model_inputs[1] ; density=model_inputs[2]
    cfi=model_inputs[3] ; tension=model_inputs[4]
    
    # litho and plotting must be real booleans, not str
    litho=require_bool(litho,'litho')
    plotting=require_bool(plotting,'plotting')

    # shear is the shear modulus, G. It must not be confused with the
    # coefficient of internal friction, cfi
    shear = G(model_inputs[0],model_inputs[1])

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
     
    # Creating the model
    current_model=model(xlims,ylims,resolution[0],resolution[1],(500),(young),(poisson),(density),(cfi),(tension),g)
    grid=current_model.grid()
    x, y, = grid[0], grid[1]
        
    # Initial arrays with 0 vaules for all the main parameters
    prev_sn=np.zeros(np.shape(grid[0])); prev_ss=np.zeros(np.shape(grid[0])) ; prev_above=np.zeros(np.shape(grid[0]))
    prev_angle=np.zeros(np.shape(grid[0])) ; prev_mode=np.zeros(np.shape(grid[0]))
    
    # Initial arrays for the planes in the left hand side of the crack
    prev_blt=np.zeros(np.shape(grid[0])) ; prev_blu=np.zeros(np.shape(grid[0])) ; prev_blc=np.zeros(np.shape(grid[0])) 
    
    # Initial arrays for the planes in the right hand side of the crack
    prev_brt=np.zeros(np.shape(grid[0])) ; prev_bru=np.zeros(np.shape(grid[0])) ; prev_brc=np.zeros(np.shape(grid[0]))
    
    prev_betat=np.zeros(np.shape(grid[0]))
    
    step=0
    for step_dx, step_dy, step_dz, length, percentage in zip(dxs_to_use, dys_to_use, dzs_to_use, lengths_to_use, investigation_range):
        step=step+1
        
        # Create the crack with the driving stresses of this step
        disc=crack((upper_crack_tip_x,upper_crack_tip_y),(lower_crack_tip_x,lower_crack_tip_y),step_dx,step_dy,step_dz)
        upper_tip=disc.up_tip() ; crack_center=disc.c() ; lower_tip=disc.low_tip()
        
        # Calculating the principal stresses and the orientation of s1, counterclockwise from the x axis
        s1, s3, s1a, s3a = current_model.principal_stresses(disc, rx, ry, litho)
        
        # Now the stress state needs to be evaluated at each point of the grid and the result stored
        # In the first loop, the first results are stored for later use.
        current_sn=[] ; current_ss=[] ; current_angle=[] ; current_mode=[] ; current_above=[]
        
        # The angles of the failure planes relative to x
        current_blt=[] ; current_blu=[] ; current_blc=[]
        current_brt=[] ; current_bru=[] ; current_brc=[]
        
        current_betat=[]
        
        # Running the calculations for each model between the specified investigation ranges
        
        for rs1,rs3,rs1a,rs3a,prsn,prss,pra,prm,prabove,prblt,prblu,prblc,prbrt,prbru,prbrc,prbt,rS,rm,rT,x_row in zip (s1,s3,s1a,s3a,prev_sn,prev_ss,prev_angle,prev_mode,prev_above,prev_blt,prev_blu,prev_blc,prev_brt,prev_bru,prev_brc,prev_betat,grid[5],grid[6],grid[8],x):
    
            mu=rm[0] ; T=rT[0]
    
            row_sn=[] ; row_ss=[] ; row_angle=[] ; row_mode=[] ; row_above=[]
            
            row_blt=[] ; row_blu=[] ; row_blc=[] #Lists for the beta values in the rows for the left hand side of the crack 
                                                 #for tension-compression (t), uniaxial compression (u), and biaxial compression (c)
            row_brt=[] ; row_bru=[] ; row_brc=[] #The same but for the right hand side of the crack
            row_betat=[]
            
            # Now a single list containing all the lists for the rows of the beta values for all planes
            rows_beta=[row_blt, row_blu, row_blc ,row_brt, row_bru, row_brc, row_betat]
            
            for vs1,vs3,vs1a,vs3a,psn,pss,pa,pm,pabove,pblt,pblu,pblc,pbrt,pbru,pbrc,pbt,cx in zip(rs1,rs3,rs1a,rs3a,prsn,prss,pra,prm,prabove,prblt,prblu,prblc,prbrt,prbru,prbrc,prbt,x_row):
                
                sn, ss, angle, mode = evaluate_stress_state(vs1,vs3,mu,T,'values')
                
                # Obtaining theta from 2theta
                angle=angle/2
                
                # This ensures that the first tensile or shear failure values 
                # are preserved at each node for all the following timesteps
                if pm=='Mode I' or pm=='Mixed-mode I-II'or pm=='Mode II - tension-compression' or pm=='Mode II - compression'  or pm=='Above':
                    row_sn.append(psn)
                    row_ss.append(pss)
                    row_angle.append(pa)
                    row_mode.append(pm)
                    row_above.append(pabove)
                    
                    for row_beta,value in zip(rows_beta,(pblt,pblu,pblc,pbrt,pbru,pbrc,pbt)):
                        row_beta.append(value)
                            
                else:
                    row_sn.append(sn) ; row_ss.append(ss)
                    row_angle.append(angle) ; row_mode.append(mode)
                    
                    # This is done to generate to sets of vectors, one for the
                    # shear failure planes, another for the tension failure planes
                    if mode=='Above':
                        row_above.append(0)
                    else:
                        row_above.append(float('NaN'))
                    
                    if mode=='Mode I':
                        i=1
                        for row_beta in rows_beta:
                            if i==len(rows_beta):
                                # Mode I plane: contains sigma1
                                row_beta.append(failure_plane_orientation(vs1a,angle,mode,0))
                            else:
                                row_beta.append(float('NaN'))
                            i=i+1    
                                                                      
                    # The normal-fault-favorable failure planes 
                    # must be selected from the conjugate planes
                    else:
                        # Identifying where the point is located relative to the
                        # crack and the sign of the principal stress, which determines
                        # the formula to calculate the angle that should be used
                        # The conjugate plane that dips towards the crack is selected
                        # (failure_plane_orientation). j offsets the left/right arrays.
                        if cx<=crack_center[0]:
                            j=0 ; side=-1
                        else:
                            j=3 ; side=1
                        
                        angle_value=failure_plane_orientation(vs1a,angle,mode,side)
                        
                        # Identifying the failure mode
                        if mode=='Mixed-mode I-II':
                            i=1+j

                        elif mode=='Mode II - tension-compression':
                            i=2+j

                        elif mode=='Mode II - compression':
                            i=3+j

                        else:
                            i=0

                        j=1
                        for row_beta in rows_beta:
                            if i==j:
                                row_beta.append(angle_value)
                            else:
                                row_beta.append(float('NaN')) 
                            j=j+1
            
            current_sn.append(row_sn) ; current_ss.append(row_ss)
            current_angle.append(row_angle) ; current_mode.append(row_mode)
            current_above.append(row_above)
            
            current_blt.append(row_blt) 
            current_blu.append(row_blu) 
            current_blc.append(row_blc) 
            current_brt.append(row_brt) 
            current_bru.append(row_bru) 
            current_brc.append(row_brc) 
            current_betat.append(row_betat)
        
        current_sn=np.asarray(current_sn) ; current_ss=np.asarray(current_ss)        
        current_angle=np.asarray(current_angle) ; current_mode=np.asarray(current_mode)
        current_above=np.asarray(current_above)
        
        current_blt=np.asarray(current_blt)
        current_blu=np.asarray(current_blu)
        current_blc=np.asarray(current_blc)
        current_brt=np.asarray(current_brt)
        current_bru=np.asarray(current_bru)
        current_brc=np.asarray(current_brc)
        current_betat=np.asarray(current_betat)
                
        # Storing the current results as the previous results for the next run
        prev_sn=current_sn ; prev_ss=current_ss
        prev_angle=current_angle ; prev_mode=current_mode
        prev_above=current_above
        
        prev_blt=current_blt
        prev_blu=current_blu
        prev_blc=current_blc
        prev_brt=current_brt
        prev_bru=current_bru
        prev_brc=current_brc
        prev_betat=current_betat
                        
        ############# Plotting #############
        if plotting:
            figure=create_figure(xlims,ylims)
                    
            ax=figure.axes[0]
            ax.plot((upper_tip[0],lower_tip[0]),(upper_tip[1],lower_tip[1]),c='red',lw=3, label='Crack') 
            
            # Plotting stresses
            plot=ax.pcolormesh(x,y,current_sn,cmap='viridis', vmin=true_min_max(current_sn)[0], 
                                vmax=true_min_max(current_sn)[1], shading='nearest')  
            
            cbar=figure.colorbar(plot,extend='both',format='%.2f')
            cbar.set_label(r'$\sigma_n$ (MPa)', rotation=270, labelpad=20, fontsize=15)
            cbar.ax.tick_params(labelsize=12)
            
            ax.contour(x,y,current_sn, levels=[0], colors='black',linewidths=0.75,linestyles='--')
            
            # Plotting the gridcells which are found above failure in each timestep
            ax.pcolormesh(x,y,current_above,cmap='Reds_r',shading='nearest')  
            
            # Plotting downsampled versions of the failure plane orientations
            ds_brt=downsample_2darray(current_brt,downscaling)
            ds_bru=downsample_2darray(current_bru,downscaling)
            ds_brc=downsample_2darray(current_brc,downscaling)
            ds_blt=downsample_2darray(current_blt,downscaling)
            ds_blu=downsample_2darray(current_blu,downscaling)
            ds_blc=downsample_2darray(current_blc,downscaling)
            ds_betat=downsample_2darray(current_betat,downscaling)
    
            # Plane traces at orientation b (counterclockwise from +x) are drawn as
            # (cos b, sin b). Until Nov 2026 they were drawn mirrored, as (cos b, -sin b).
            # Plotting mode-II discs under compression
            ax.quiver(x,y,np.cos(ds_brc),np.sin(ds_brc),
                            headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='red')
            ax.quiver(x,y,np.cos(ds_blc),np.sin(ds_blc),
                            headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='red')          
            
            # Plotting mode-II discs under tension-compression
            ax.quiver(x,y,np.cos(ds_bru),np.sin(ds_bru),
                            headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='purple')
            ax.quiver(x,y,np.cos(ds_blu),np.sin(ds_blu),
                            headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='purple')  
            
            # Plotting mixed-mode discs
            ax.quiver(x,y,np.cos(ds_brt),np.sin(ds_brt),
                            headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='dodgerblue')
            ax.quiver(x,y,np.cos(ds_blt),np.sin(ds_blt),
                            headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='dodgerblue')
                    
            # Plotting mode-I discs
            ax.quiver(x,y,np.cos(ds_betat),np.sin(ds_betat),
                            headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='blue')
            
            # plt.legend(loc=3)
            
            ax.text(0.97,0.14, length_label+'·'+str(round(percentage,4))+' = '+str(round(length,2))+' m\n'+
                    r'$\sigma_I$·'+str(round(percentage,4))+' = '+str(round(step_dx/1e6,2))+' MPa\n'+
                    r'$\sigma_{II}$·'+str(round(percentage,4))+' = '+str(round(step_dy/1e6,2))+' MPa',
                    bbox=dict(fc='white', ec='black'),
                    horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes,fontsize=11, zorder=5)
            
            # sys.exit()
            
            figure.savefig(os.path.join(folder,'sn'+'_'+str(round(shear/1e9,2))+'_'+str(young/1e9)+'_'+str(tension/1e6)+'_'+str(step)+'_'+str(round(step_dx/1e6,2))+'_MPa.png'),dpi=300)
            
            #Closing and deleting the figure related info
            plt.close()
            del figure
            del ds_brt, ds_bru, ds_brc, ds_blt, ds_blu, ds_blc, ds_betat
        
        
        # Deleting all previous model calculations to clean memory
        del disc
        del current_sn, current_ss, current_angle, current_above
        del current_blt, current_blu, current_blc, current_brt, current_bru, current_brc, current_betat 
        
        
        gc.collect()
        
    return current_mode
     
        ############# End of plotting #############