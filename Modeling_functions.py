import time
import sys
import os
import cv2
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

# Obtain the function to evaluate the stress state
from Failure_Determination import evaluate_stress_state

# Obtaining helper functions
from Helper_functions import create_figure
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
    crack_inputs :   list-like, contains the dike parameters,
                    crack aperture as a float/int in m, if the model is of a
                    fault, then aperture should be 0
                    crack height as a float/int in m,
                    crack upper tip x coordinate,
                    crack upper tip y coodintate,
                    crack lower tip x coordinate,
                    crack lower tip y coodintate,
                    where these two should be consistent with 
                    the x and y values used in xlims and ylims
    model_inputs :  list-like, contains the model parameters for a
                    single-model layer.
                    Young's modulus as a float/int in Pa,
                    Poissons ratio as an float/int between 0-0.5,
                    density as an float/int in kgm^3,
                    cohesion as an float/int in Pa,
                    mu, the coefficient of internal friction, as an float/int,
                    tension, tensile strength as an float/int in Pa.
    
    px : driving stress orthogonal to the crack wall as an float/int, in Pa, 
    only needs to be non-zero if the model is of a fault. >0 values model
    a dilatant fault, and <0 values model a compaction band
    py : driving stress parallel to the crack wall as an float/int, in Pa, where
    <0 values are right-lateral motion, and >0 are a left-lateral motion
    pz : out-of-plane driving stress as an float/int, in Pa, usually 0
    rx : constant remote stress along the X axis as an float/int, in Pa
    ry : constant remote stress along the Y axis as an float/int, in Pa
    g : gravitational acceleration as a float/int in ms^-2
    litho : a boolean as a str, 'True' to take into account the lithostatic
    stresses in the calculations, 'False', to not do it. This has no effect
    on the displacements.
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
    a = crack_inputs[0] ; b = crack_inputs[1] ; 
    upper_crack_tip_x = crack_inputs[2] ;  upper_crack_tip_y = crack_inputs[3]
    lower_crack_tip_x = crack_inputs[4] ;  lower_crack_tip_y = crack_inputs[5]
    
    # Defining variables from model_inputs
    young=model_inputs[0] ; poisson=model_inputs[1] ; density=model_inputs[2]
    cohesion=model_inputs[3] ; cfi=model_inputs[4] ; tension=model_inputs[5]
    
    # Calculating final crack driving stress. This is the value from which the 
    # pressures to model are calculated using the input pressure range
    mu = G(model_inputs[0],model_inputs[1])
        
    if a==0:
        px=dx
        disc=crack((upper_crack_tip_x,upper_crack_tip_y),(lower_crack_tip_x,lower_crack_tip_y),dx,dy,dz)
    else:
        # From Pollard et al., 1986
        px = a*mu/(2*b*(1-model_inputs[1]))
        # Creating the crack object
        disc=crack((upper_crack_tip_x,upper_crack_tip_y),(upper_crack_tip_x,upper_crack_tip_x-b),px,dy,dz)
       
     # Preparing the model
    if parameter=='ux' or parameter=='uy' or parameter=='uxy' or parameter=='st':
         current_model=model(xlims,ylims,resolution[0]*2.5,resolution[1]*3,(200),(young),(poisson),(density),(cohesion),(cfi),(tension),g)
    
    else:
        current_model=model(xlims,ylims,resolution[0],resolution[1],(500),(young),(poisson),(density),(cohesion),(cfi),(tension),g)
    
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
        plot=current_model.plot_stress_trajectories(disc,'False', rx, ry,litho)
    
    elif parameter=='sn':
        plot=current_model.plot_plane_stresses(disc,'normal',rx,ry,angle,litho,0.5)
    elif parameter=='ss':
        plot=current_model.plot_plane_stresses(disc,'shear',rx,ry,angle,litho,0.5)
   
    return plot
    
################ A function to run a dynamic model of dike opening ################

def dynamic_model(dike_inputs,model_inputs,dy,dz,rx,ry,g,litho,investigation_range,xlims,ylims,resolution,downscaling,folder):
    """

    Parameters
    ----------
    dike_inputs :   list-like, contains the dike parameters,
                    dike aperture as a float/int in m,
                    dike height as a float/int in m,
                    dike upper tip x coordinate,
                    dike upper tip y coodintate,
                    where these two should be consistent with 
                    the x and y values used in xlims and ylims
    model_inputs :  list-like, contains the model parameters for a
                    single-model layer.
                    Young's modulus as a float/int in Pa,
                    Poissons ratio as an float/int between 0-0.5,
                    density as an float/int in kgm^3,
                    cohesion as an float/int in Pa,
                    mu, the coefficient of internal friction, as an float/int,
                    tension, tensile strength as an float/int in Pa.
    
    py : driving stress parallel to the dike wall as an float/int, in Pa
    pz : out-of-plane driving stress as an float/int, in Pa, usually 0
    rx : constant remote stress along the X axis as an float/int, in Pa
    ry : constant remote stress along the Y axis as an float/int, in Pa
    g : gravitational acceleration as a float/int in ms^-2
    litho: a boolean as a str, 'True' to take into account the lithostatic
    stresses in the calculations, 'False', to not do it.
    investigation_range :   list-like, two values between 0-100 between which
                            the evolution of the dike opening is to be modeled
    
    xlims : list-like, X-boundaries of the model
    ylims : list-like, Y-boundaries of the model
    resolution : list-like, two values indicating cell-size along the X and Y axes  
    downscaling: int, a value to scale up or down the frequency with which
    the discontinuities are plotted. The smaller the cellsize, the larger the 
    downscaling factor should be.

    folder : path in which to save the figures, as a str

    Returns
    -------
    None. Saves the figure of each of the modeled steps in the specified
    folder. Each figure shows a legend indicating dike aperture and
    driving stress in each step of the model. By default, the figures
    plot the discontinuities and map the color of the normal stresses
    found for each discontinuity.

    """    

    # Defining variables from dike_inputs
    a = dike_inputs[0] ; b = dike_inputs[1] ; 
    upper_dike_tip_x = dike_inputs[2] ;  upper_dike_tip_y = dike_inputs[3]
    
    # Defining variables from model_inputs
    young=model_inputs[0] ; poisson=model_inputs[1] ; density=model_inputs[2]
    cohesion=model_inputs[3] ; mu=model_inputs[4] ; tension=model_inputs[5]
    
    # Calculating final dike driving stress. This is the value from which the 
    # pressures to model are calculated using the input pressure range
    mu = G(model_inputs[0],model_inputs[1])
        
    # From Pollard et al., 1986
    dx = a*mu/(2*b*(1-model_inputs[1]))
    
    pxs_to_use=[] ; apertures_to_use=[]
    for percentage in investigation_range:
        pxs_to_use.append(dx*percentage)
        apertures_to_use.append(a*percentage)
            
    # Creating the model
    current_model=model(xlims,ylims,resolution[0],resolution[1],(500),(young),(poisson),(density),(cohesion),(mu),(tension),g)
    grid=current_model.grid()
    x, y, = grid[0], grid[1]
    
    # Initial arrays with 0 vaules for all the main parameters
    prev_sn=np.zeros(np.shape(grid[0])); prev_ss=np.zeros(np.shape(grid[0])) ; prev_above=np.zeros(np.shape(grid[0]))
    prev_angle=np.zeros(np.shape(grid[0])) ; prev_mode=np.zeros(np.shape(grid[0]))
    
    # Initial arrays for the planes in the left hand side of the dike
    prev_blt=np.zeros(np.shape(grid[0])) ; prev_blu=np.zeros(np.shape(grid[0])) ; prev_blc=np.zeros(np.shape(grid[0])) 
    
    # Initial arrays for the planes in the right hand side of the dike
    prev_brt=np.zeros(np.shape(grid[0])) ; prev_bru=np.zeros(np.shape(grid[0])) ; prev_brc=np.zeros(np.shape(grid[0]))
    
    prev_betat=np.zeros(np.shape(grid[0]))
    
    a=0
    for px, aperture, percentage in zip(pxs_to_use, apertures_to_use, investigation_range):
        a=a+1
        
        # Create the dike at a given pressure
        dike=crack((upper_dike_tip_x,upper_dike_tip_y),(upper_dike_tip_x,upper_dike_tip_y-b),dx,dy,dz)
        upper_tip=dike.up_tip() ; dike_center=dike.c() ; lower_tip=dike.low_tip()
        
        # Calculating the principal stresses and the orientation of s1 clockwise relative to the x axis
        s1, s3, s1a, s3a = current_model.principal_stresses(dike, rx, ry, litho)
        
        # Now the stress state needs to be evaluated at each point of the grid and the result stored
        # In the first loop, the first results are stored for later use.
        current_sn=[] ; current_ss=[] ; current_angle=[] ; current_mode=[] ; current_above=[]
        
        # The angles of the failure planes relative to x
        current_blt=[] ; current_blu=[] ; current_blc=[]
        current_brt=[] ; current_bru=[] ; current_brc=[]
        
        current_betat=[]
        
        for rs1,rs3,rs1a,rs3a,prsn,prss,pra,prm,prabove,prblt,prblu,prblc,prbrt,prbru,prbrc,prbt,rS,rm,rT,rx in zip (s1,s3,s1a,s3a,prev_sn,prev_ss,prev_angle,prev_mode,prev_above,prev_blt,prev_blu,prev_blc,prev_brt,prev_bru,prev_brc,prev_betat,grid[5],grid[6],grid[8],x):
    
            mu=rm[0] ; T=rT[0]
    
            row_sn=[] ; row_ss=[] ; row_angle=[] ; row_mode=[] ; row_above=[]
            
            row_blt=[] ; row_blu=[] ; row_blc=[] #Lists for the beta values in the rows for the left hand side of the dike 
                                                 #for tension-compression, uniaxial compression, and biaxial compression
            row_brt=[] ; row_bru=[] ; row_brc=[] #The same but for the right hand side of the dike
            row_betat=[]
            
            # Now a single list containing all the lists for the rows of the beta values for all planes
            rows_beta=[row_blt,row_blu,row_blc ,row_brt,row_bru,row_brc, row_betat]
            
            for vs1,vs3,vs1a,vs3a,psn,pss,pa,pm,pabove,pblt,pblu,pblc,pbrt,pbru,pbrc,pbt,cx in zip(rs1,rs3,rs1a,rs3a,prsn,prss,pra,prm,prabove,prblt,prblu,prblc,prbrt,prbru,prbrc,prbt,rx):
                
                sn, ss, angle, mode = evaluate_stress_state(vs1,vs3,mu,T,'values')
                
                # Obtaining theta from 2theta
                angle=angle/2
                
                # This ensures that the first tensile or shear failure values 
                # are preserved at each node for all the following timesteps
                if pm=='mode I' or pm=='mixed-mode'or pm=='mode II - tension-compression' or pm=='mode II - compression'  or pm=='above':
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
                    if mode=='above':
                        row_above.append(0)
                    else:
                        row_above.append(float('NaN'))
                    
                    if mode=='mode I':
                        i=1
                        for row_beta in rows_beta:
                            if i==len(rows_beta):
                                row_beta.append(np.deg2rad(angle+vs1a-90))
                            else:
                                row_beta.append(float('NaN'))
                            i=i+1    
                                                                      
                    # The normal-fault-favorable failure planes 
                    # must be selected from the conjugate planes
                    else:
                        # Identifying where the point is located relative to the
                        # dike and the sign of the principal stress, which determines
                        # the formula to calculate the angle that should be used
                        if cx<=dike_center[0]:
                            j=0
                            if vs1a>=0:
                                angle_value=np.deg2rad(vs1a+angle-90)
                                                              
                            elif vs1a<0:
                                # angle_value=-np.deg2rad(90+abs(vs1a)-angle)
                                angle_value=-np.deg2rad(abs(vs1a)-angle+90) #Why? This does not match with my drawings
        
                        elif cx>dike_center[0]:
                            j=3
                            if vs1a>=0:
                                # angle_value=np.deg2rad(90+vs1a-angle)
                                angle_value=np.deg2rad(vs1a-angle+90) #Why? As above
                                
                            elif vs1a<0:
                                angle_value=-np.deg2rad(abs(vs1a)+angle-90)
                        
                        # Identifying the failure mode
                        if mode=='mixed-mode':
                            i=1+j

                        elif mode=='mode II - tension-compression':
                            i=2+j

                        elif mode=='mode II - compression':
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
        figure=create_figure(xlims,ylims)
                
        ax=figure.axes[0]
        ax.plot((upper_tip[0],lower_tip[0]),(upper_tip[1],lower_tip[1]),c='red',lw=3, label='Dike') 
        
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

        # Plotting mode-II discs under compression
        ax.quiver(x,y,np.cos(ds_brc),np.sin(-ds_brc),
                        headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='red')
        ax.quiver(x,y,np.cos(ds_blc),np.sin(-ds_blc),
                        headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='red')          
        
        # Plotting mode-II discs under tension-compression
        ax.quiver(x,y,np.cos(ds_bru),np.sin(-ds_bru),
                        headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='purple')
        ax.quiver(x,y,np.cos(ds_blu),np.sin(-ds_blu),
                        headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='purple')  
        
        # Plotting mixed-mode discs
        ax.quiver(x,y,np.cos(ds_brt),np.sin(-ds_brt),
                        headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='dodgerblue')
        ax.quiver(x,y,np.cos(ds_blt),np.sin(-ds_blt),
                        headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='dodgerblue')
                
        # Plotting mode-I discs
        ax.quiver(x,y,np.cos(ds_betat),np.sin(-ds_betat),
                        headlength=0, pivot='middle', scale=0.0040, units='xy', width=20, headwidth=1, color='blue')
        
        # plt.legend(loc=3)
        
        ax.text(0.97,0.14, 'a·'+str(round(percentage,4))+' = '+str(round(aperture,2))+' m\n'+
                r'$\sigma_I$·'+str(round(percentage,4))+' = '+str(round(px/1e6,2))+' MPa',
                bbox=dict(fc='white', ec='black'),
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes,fontsize=11, zorder=5)
                
        figure.savefig(os.path.join(folder,'sn'+'_'+'_'+str(round(mu,2))+'_'+str(young/1e9)+'_'+str(tension/1e6)+'_'+str(a)+'_'+str(round(px/1e6,2))+'_MPa.png'),dpi=300)
            
        #Deleting useless stuff and and cleaning memory
        plt.close() 
        del dike, figure
        del current_sn, current_ss, current_angle, current_mode, current_above
        del current_blt, current_blu, current_blc, current_brt, current_bru, current_brc, current_betat 
        del ds_brt, ds_bru, ds_brc, ds_blt, ds_blu, ds_blc, ds_betat
        
        gc.collect()
     
        ############# End of plotting #############