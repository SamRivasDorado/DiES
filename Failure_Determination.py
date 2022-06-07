import sys
import numpy as np
import scipy as sp
import math
import os
import matplotlib.pyplot as plt
import numpy.ma as ma

# Obtaining helper functions
from Helper_functions import linear
from Helper_functions import circle_eq_y
from Helper_functions import points_distance
from Helper_functions import angle_between_lines

def evaluate_stress_state(p,s,mu,T,output):
    """
    
    Parameters
    ----------
    s1 : int/float, the maximum compressive stress in Pa
    s2 : int/float, the minimum compressive stress in Pa
    S: positive int/float, cohesion in Pa
    T : negative int/float, tensile strength in Pa.
    mu : Coefficient of internal friction
    output: a str, 'plots' or 'values'

    Returns
    -------
    If output = 'plots', returns a Mohr circle plot of the given stress state
    and the Griffith and Coulomb criteria.
    If output = 'values', returns a tuple with the normal, shear stress, 
    failure angle, and a str: 'below' if the circle is below failure, 
                              'above' if the circle is above failure,
                              'tension' if the circle is at tensional failure
                              'shear' if the circle is at shear failure

    """
    
    if p<s:
        print('p is set as the principal stress but is smaller than s.')
        print('Please review the values and set p>s.')
        sys.exit()
    
    elif p==s:
        print ('p = s and thus the state of stress is isotropic.')
        print ('A Mohr circle cannot be drawn in this situation.')
        sys.exit()
        
    # elif p<0 and s<0:
    #     print ('This is a biaxial tension situation.')
    #     print ('This algorithm is not suited to evaluate this scenario.')
    #     print ('Please revise your values and try again.')
    
    # Redefining the inputs tp MPa, and T as a tensional value
    p=p/1e6 ; s=s/1e6 ; T=(-T)/1e6
    
    # Mohr circle radius # Mohr circle x coordinate
    r = abs(p - s)/2 ; x_center = p-(p-s)/2
                                    
    # The points around the Mohr circle
    circle_points = np.linspace(s,p,int(r)*50)
    
    # Calculating the slope and intercept of the Mohr-Coulomb failure line
    slope_l = mu ; intercept_l = -2*T
    
    temp_sn=[] ; temp_ss=[] ; temp_angle=[]
        
    circle_tangent_params=[] ; parabole_tangent_params=[] ; line_tangent_params=[]
    
    i=0
    
    for x in circle_points:
            
        # Getting the full coordinates of the Mohr Circle point
        y_circle = circle_eq_y(x_center, r, x)[0]
        
        if y_circle==0:
            y_circle = 0.0001
        
        # Calculating the tangent line to the Mohr circle
        slope_c = (x_center-x)/y_circle
        intercept_c = y_circle - slope_c*x
        
        # 1 ) We avoid avoiding doing extra operations if x<T, 
        # when the circle is surely above the failure line.
        
        if x<T:
            i=2
            sn=float('NaN') ; ss=float('NaN') ; angle=float('NaN') ; mode='above'
            # print ('The circle is above failure')
            # print (sn, ss, angle)
            break
        
        # 2 ) If this is not the case, then a number of things may happen.
                                
        # 2.1 ) If the parabole slope (T/(x-T))**0.5 were zero, it means that 
        # the tangent to the parabole at this point would be a vertical line.
            
        # This can only occur when x==T, and therefore when sigma3 is equal 
        # to T, when the left edge of the circle is exactly touching T. 
        
        # In this case, the tangent of the parabole and the Mohr circle are
        # the same, and the circle may be at tensional failure. However, this 
        # needs to be checked because the circle may be already past failure.

        elif 0.95<x/T<1.05:
                 
            # Measuring the vertical difference between the y points of the
            # parabole and the Mohr circle to define if it is above failure
            y_differences=[]
            
            for x in circle_points:
                y_circle = circle_eq_y(x_center, r, x)[0]            
                y_parabole=(abs(4*T*(x-T)))**0.5
                
                if y_parabole==0:
                    y_parabole=0.001
    
                y_difference=y_parabole-y_circle ; y_differences.append(y_difference)
                                
            j=0 ; k=0
            for y in y_differences:
                # To quantify how many points of the circle are below
                # the failure criterion
                if y > 0:
                    j=j+1
                # To quantify how many points of the circle are above 
                # the failure criterion
                elif y < 0:
                    k=k+1
             
            # 2.1.2 ) If all points of the circle are below 
            #  failure, then it is considered to be below failure
            # if len(circle_points)==j:
            #     i=2
            #     sn=float('NaN') ; ss=float('NaN') ; angle=float('NaN'); mode='Below'
            
            # 2.1.2 ) If several points of the circle are found to be, 
            #  above failure, then it is considered to be past failure
            if k>=10:
                i=2
                sn=float('NaN') ; ss=float('NaN') ; angle=float('NaN'); mode='above'
                break
            
            # 2.1.3 ) If all points except the first are below failure, 
            # the circle is under tensional failure.
            elif 0.95<len(circle_points)/j<1.05:
                i=1
            
                slope_p = slope_c
                intercept_p = T - slope_p*T 
                
                circle_tangent_params.append((slope_c, intercept_c))
                parabole_tangent_params.append((slope_p, intercept_p))
                
                sn=T ; ss=0 ; angle=180 ; mode='mode I'
                
                break
            
        # 2.2) If there is no possibility of tensional failure, then the 
        # point may be tangent to the Griffith or Coulomb criterion
        
        else:
            # 2.2.1 ) When x<0 the Griffith criterion should be used
            if x<0:
                # Calculating the corresponding y point of the parabole at this x
                y_p = (abs(4*T*(x-T)))**0.5
                
                # Calculating the tangent line to the parabola at the x position
                slope_p = abs(T/(x-T))**0.5
                intercept_p = y_p - slope_p*x 
                
                # Check if a point is tangent to the parabole
                if 0.95<(intercept_c/intercept_p)<1.05 and 0.95<(slope_c/slope_p)<1.05:
                    i=1
                    
                    circle_tangent_params.append((slope_c, intercept_c))
                    parabole_tangent_params.append((slope_p, intercept_p))
                    
                    temp_sn.append(x) ; temp_ss.append(y_circle)
                    
                    angle=angle_between_lines((x_center,0), (p,0), (x_center,0), (x,y_circle))
                    temp_angle.append(angle)
            
            # 2.2.2 ) When x<0 the Coulomb criterion should be used
            elif x>=0:
                # Check if a point is tangent to the line
                if 0.95<(intercept_c/intercept_l)<1.05 and 0.95<(slope_c/slope_l)<1.05:
                    i=1
                    
                    circle_tangent_params.append((slope_c, intercept_c))
                    line_tangent_params.append((slope_l, intercept_l))
                    
                    temp_sn.append(x) ; temp_ss.append(y_circle)
                    
                    angle=angle_between_lines((x_center,0), (p,0), (x_center,0), (x,y_circle))
                    temp_angle.append(angle)
    
            # Because these conditions may fulfill failure at multiple points,
            # we average the sn, ss, and angle corresponding in these points
            # to obtain a final result
            
            if i==1:
                if len(temp_ss)==1:
                    sn=temp_sn[0] ; ss=temp_ss[0] ; angle=temp_angle[0]
                    
                elif len(temp_ss)>1:
                    sn=np.mean(temp_sn) ; ss=np.mean(temp_ss); angle=np.mean(temp_angle)
            
                if s<0 and p>0:
                    if sn>0:
                        mode='mode II - tension-compression'
                    elif sn<0:
                        mode='mixed-mode'
                elif s>0 and p>0:
                    mode='mode II - compression'
            
    # 3) If no solution has been found so far, then the circle is either
    # above or below the failure lines
    if i==0:
        y_differences=[]
        
        for x in circle_points:
            
            # 3.1 ) When x<0  we check if the point is below the Griffith
            # failure line
            if x<0:
                y_circle = circle_eq_y(x_center, r, x)[0]            
                y_parabole=(abs(4*T*(x-T)))**0.5
                
                if y_parabole==0:
                    y_parabole=0.001
    
                y_difference=y_parabole-y_circle ; y_differences.append(y_difference)
            
            # 3.2 ) When x>0  we check if the point is below the Coulomb
            # failure line
            elif x>0:
                y_circle = circle_eq_y(x_center, r, x)[0]            
                y_line=-2*T+mu*x
                
                if y_line==0:
                    y_line=0.001
    
                y_difference=y_line-y_circle ; y_differences.append(y_difference)
            
            # This methodology works because it allows to apply the failure
            # criterion that is needed as a function of the value of x
            
        j=0 ; k=0
        for y in y_differences:
            # To quantify how many points of the circle are below
            # the failure criterion
            if y > 0:
                j=j+1
            # To quantify how many points of the circle are above 
            # the failure criterion
            elif y < 0:
                k=k+1
                    
        if len(circle_points)==j:
            i=2
            # print ('The circle is below failure')
            sn=float('NaN') ; ss=float('NaN') ; angle=float('NaN'); mode='below'
            
   
        elif k>=1:
            i=2
            # print ('The circle is above failure')
            sn=float('NaN'); ss=float('NaN') ; angle=float('NaN'); mode='above'
                    
    if output=='plot':
        # Create some variables for plotting
        x_boundaries=np.asarray((x_center-r*2,x_center+r*2))
        y_boundaries=np.asarray((r*1,-r*2))
        x_values_G = np.linspace(T,0,200)
        x_values_MC = np.linspace(0,x_boundaries[1],200)
        
        # Plot some axis elements
        figure=plt.figure(figsize=(5,4))
        ax=figure.add_subplot(111,xlim=x_boundaries,ylim=-y_boundaries)
        ax.set_ylabel(r'$\sigma_s$ (MPa)')
        ax.set_xlabel(r'$\sigma_n$ (MPa)') 
        ax.hlines(0,x_boundaries[0],x_boundaries[1], color='black',lw=0.5)
        ax.vlines(0,y_boundaries[0]*2,y_boundaries[1], color='black',lw=0.5)
        
        # Plotting the Mohr Circle and its defining parameters
        if mode=='below' or mode=='above':
            color='black'
        
        elif mode=='mode II - compression':
            color='red'
        
        elif mode=='mode II - tension-compression':
            color='purple'
        
        elif mode=='mixed-mode':
            color='dodgerblue'
            
        elif mode=='mode I':
            color='blue'
        
        ax.plot(circle_points,circle_eq_y(x_center,r,circle_points)[0],lw=3,c=color, zorder=3)
        ax.scatter(x_center,0,c='black', s=30)
        ax.scatter(p,0, lw=0.5, edgecolor='black', c=color, s=70, zorder=3)
        ax.scatter(s,0, lw=0.5, edgecolor='black', c=color, s=70, zorder=3)
                    
        # Plotting the modified Griffith criterionn
        y_values = (abs(4*T*(x_values_G-T)))**0.5 
        ax.plot(x_values_G,y_values,ls='--',lw=3,c='darkred')
        
        y_values = -(abs(4*T*(x_values_G-T)))**0.5 
        ax.plot(x_values_G,y_values,ls='--',lw=3,c='darkred')
        
        y_values = -2*T + mu*x_values_MC
        ax.plot(x_values_MC,y_values,ls='--',lw=3,c='tomato')
        
        y_values = -2*T + mu*x_values_MC
        ax.plot(x_values_MC,-y_values,ls='--',lw=3,c='tomato')
        
        # Cutoff line for the Mohr-Coulomb criterion under uniaxial tension
        # ax.vlines(T,0,-2*T + mu*T,ls=':',color='red')
        
         # Plotting the contact point only if there is any kind of failure
        if i==1:
            ax.scatter(sn,ss,marker='*',edgecolor='black', c='orange', s=200, zorder=3)   
            ax.plot((x_center,sn),(0,ss), ls='--', lw=1.5, c='black')
            ax.text(sn+r,ss+r*1,r'2$\theta$ = '+str(round(angle,2)),bbox=dict(fc='white', ec='black'), fontsize=12)
            # +' °\n'
                    # +r'$\theta$ = '+str(round(angle/2,2))+' °')
            
            # Plotting the tangent lines
            # x_values=np.linspace(-r*4,r*4,100)
            # ax.plot(x_values,circle_tangent_params[0][0]*x_values+circle_tangent_params[0][-1],c='grey',lw=1)
            
            # if s<0:
            #     ax.plot(x_values,parabole_tangent_params[0][0]*x_values+parabole_tangent_params[0][-1],c='indianred',lw=1)
            # elif s>=0:
            #     ax.plot(x_values,line_tangent_params[0][0]*x_values+line_tangent_params[0][-1],c='indianred',lw=1)
                
        ax.text(p-r/6,0-r/1,r'$\sigma_1$ = '+str(round(p,2))+' MPa',bbox=dict(fc='white', ec='black'), fontsize=12)
        ax.text(s-r/0.5,0-r/1,r'$\sigma_3$ = '+str(round(s,2))+' MPa',bbox=dict(fc='white', ec='black'), fontsize=12)
        
        return figure
        
    elif output=='values':
        return sn, ss, angle, mode
