import sys
import time
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import (MultipleLocator)

# Obtaining helper functions
from Helper_functions import G
from Helper_functions import points_distance
from Helper_functions import angle_between_lines
from Helper_functions import linear
from Helper_functions import boundaries_contours
from Helper_functions import point_side_of_line

########################### Defining standard parameters ########################### 

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', title_fontsize=11)    # legend title fontsize
plt.rc('legend', fontsize=11)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

########################### Exceptions ###########################

class Error(Exception):
    """
    Base class for exceptions in this module.
    """
    
    pass
    
class InputError(Error):
    def __init__(self,message):
        self.message = message

def require_bool(value,name):
    """
    Makes sure that an on/off input is a real boolean (True or False).
    Strings such as 'True' or 'False' are not accepted, because they would
    silently match no condition in the checks that follow.

    Parameters
    ---------------
    value: the input to check,
    name: a str, the name of the input, used in the error message.

    Returns
    ---------------
    The value itself as a bool if it is valid. Otherwise, raises an InputError.
    """

    if not isinstance(value,(bool,np.bool_)):
        raise InputError("'"+name+"' must be a boolean, True or False, not "
                         +type(value).__name__+": "+repr(value)+".")

    return bool(value)

########################### Functions for the two spatial parameters ###########################

def Rs(upper_tip,crack_center,lower_tip,point):
    """
    Takes the coordinates of the crack's upper and lower tips,
    and the coordinates of the point of interest,
    both as tuples/lists/arrays of two values.
    At least one of the x coordinates of the crack position
    needs to be a non-zero number, but should be a very close
    value to 0 to model a vertical crack.
    
    Returns the r, r1,r2, and R parameters.
    """
    
    r=points_distance(crack_center,point)
    r1=points_distance(lower_tip, point)
    r2=points_distance(upper_tip,point)
    
    return r,r1,r2,(r1*r2)**0.5

def THETAS(upper_tip,crack_center,lower_tip,point):
    """
    Takes the coordinates of the crack's upper and lower tips,
    the coordinates of the crack center,
    and the coordinates of the point of interest,
    both as tuples/lists/arrays of two values.
    
    'Returns the theta, theta1, theta2, and THETA parameters, in degrees. 
    """
    
    theta=angle_between_lines(upper_tip,lower_tip,crack_center,point)
    theta1=angle_between_lines(upper_tip,lower_tip,lower_tip,point)
    theta2=angle_between_lines(upper_tip,lower_tip,upper_tip,point)
    THETA=(theta1+theta2)/2
    
    return theta, theta1, theta2, THETA

########################### Displacement field equations ###########################

def displacements(dpx,dpy,dpz,nu,E,upper_tip,crack_center,lower_tip,point):
    """
    Parameters
    ---------------
    dpx = Driving stress acting along the x axis in Pa,
    dpy = Driving stress acting along the y axis in Pa,
    v = Poissons ratio (unitless),
    E = Youngs Modulus in Pa,
    upper_tip: the coordinates of the crack's upper tip,
    crack_center: the coordinates of the crack center,
    lower_tip: the coordinates of the crack's lower tip,
    point: the coordinates of the point of interest
    All coordinates are 1D array-like objects of two values. 
     
    
    Returns 
    ---------------
    The in-plane displacements at the point of interest, as 1D array-like
    objects, u_x, u_y, u_z
    """
    
    # Modifying the investigation point when the investigated x value
    # is at the x coordinates of the tips, to avoid zero-errors
    # if point[0]==crack_center[0]:
    #     point=(point[0]+0.1,point[-1])
    
    r=Rs(upper_tip,crack_center,lower_tip,point)
    t=THETAS(upper_tip,crack_center,lower_tip,point)
    
    t_rad=[]
    for t in t:
        t_rad.append(np.deg2rad(t))
    
    """
    Calculating u_x 
    """
    
    """
    First, we fullfill the condition of positive traction directed 
    toward one side of the fracture, and negative traction to the other 
    """
    
    x=(upper_tip[0],crack_center[0],lower_tip[0])
    y=(upper_tip[1],crack_center[1],lower_tip[1])
    
    if upper_tip[0]==lower_tip[0]:
        if point[0]<crack_center[0]:
            dpx=-dpx
    
    else:
        params=np.polyfit(x,y,1)
        function=np.poly1d(params)
        y_check=function(point[0])
            
        if point[1]<y_check and params[0]<0:
            dpx=-dpx
        
        elif point[1]>y_check and params[0]>0:
            dpx=-dpx
             
    first_term=2*(1-nu)
    second_term=r[-1]*np.sin(t_rad[-1])-r[0]*np.sin(t_rad[0])
    third_term=r[0]*np.sin(t_rad[0])*((r[0]/r[-1])*np.cos(t_rad[0]-t_rad[-1])-1)
    
    fourth_term=(1-2*nu)
    fifth_term=r[-1]*np.cos(t_rad[-1])-r[0]*np.cos(t_rad[0])
    sixth_term=r[0]*np.sin(t_rad[0])*((r[0]/r[-1])*np.sin(t_rad[0]-t_rad[-1]))
    
    upper_term=dpx*(first_term*second_term-third_term)-dpy*(fourth_term*fifth_term+sixth_term)
    ux=upper_term/(2*G(E,nu))
    
    """
    Calculating u_y 
    """
    
    first_term=2*(1-nu)
    second_term=r[-1]*np.sin(t_rad[-1])-r[0]*np.sin(t_rad[0])
    third_term=r[0]*np.sin(t_rad[0])*((r[0]/r[-1])*np.cos(t_rad[0]-t_rad[-1])-1)
    
    fourth_term=(1-2*nu)
    fifth_term=r[-1]*np.cos(t_rad[-1])-r[0]*np.cos(t_rad[0])
    sixth_term=r[0]*np.sin(t_rad[0])*((r[0]/r[-1])*np.sin(t_rad[0]-t_rad[-1]))
    
    upper_term=dpy*(first_term*second_term+third_term)+dpx*(fourth_term*fifth_term-sixth_term)
    uy=upper_term/(2*G(E,nu))
    
    """
    Correcting the direction of the displacement vector uy,
    so that it points downward above the crack center, 
    and upward below the crack center.
    """
    
    if upper_tip[0]==lower_tip[0]:
        if point[0]>crack_center[0]:
            uy=-uy
            
    else:
        if point[1]<y_check and params[0]<0:
            uy=-uy
        
        elif point[1]>y_check and params[0]>0:
            uy=-uy
            
    """
    Calculating u_z 
    Pollard and Segall, 1986
    """          
    uz=(2*dpz*(r[-1]*np.sin(t_rad[-1])-r[0]*np.sin(t_rad[0])))/(2*G(E,nu))
        
    return ux,uy,uz

########################### Traction boundary value problem, stress field equations ###########################

def stresses(dpx,dpy,dpz,rxx,ryy,rxy,rxz,ryz,nu,E,upper_tip,crack_center,lower_tip,a,point):
    """
    Parameters
    ---------------
    dpx: Driving stress normal to the X plane,
    dpy: Driving stress normal to the Y plane,
    rxx: Remote stress in Pa, normal to the X plane,
    ryy: Maximum remote stress in Pa, normal to the Y plane,
    rxy: Shear stress on plane X in the direction of the Y axis,
    rxz: Shear stress on plane X in the direction of the Z axis,
    ryz: Shear stress on plane Y in the direction of the Z axis,
    all in Pa.
    
    v: Poissons ratio (unitless),
    E: Youngs Modulus in Pa,
    the coordinates of the crack's upper and lower tips, and crack center,
    a: the fractures half length in m,
    point: A 1D array-like object of two values, 
    the coordinates of the point of interest 
    
    Return
    ---------------
    The in-plane stresses at the point of interest,
    sxx, syy, sxy (=syx), sxz, syz, szz and t (the maximum in-plane shear stress),
    in this order, all in Pa.
    """

    """
    Modifying the investigation point when the investigated x value
    is at the x coordinates of the tips, to avoid zero-errors
    """
    
    if point[0]==crack_center[0]:
        point=(crack_center[0]+0.1,point[-1])
        
    r=Rs(upper_tip,crack_center,lower_tip,point)
    t=THETAS(upper_tip,crack_center,lower_tip,point)
    
    t_rad=[]
    for t in t:
        t_rad.append(np.deg2rad(t))
   
    """
    Correcting the sign of the fracture parallel stresses 
    depending on the side of the fracture to accurately model shearing: 
    left, positive, right, negative.
    """
    
    side = point_side_of_line(upper_tip,lower_tip,point)
    if side==1:
        dpy=-dpy 
    
    # Calculating sigma_xx
    first_term=rxx
    second_term=((r[0]/r[-1])*np.cos(t_rad[0]-t_rad[-1]))-1+((a**2)*(r[0]/(r[-1]**3)))*np.sin(t_rad[0])*np.sin(3*t_rad[-1])
    third_term=((a**2)*(r[0]/(r[-1]**3)))*np.sin(t_rad[0])*np.cos(3*t_rad[-1])
    sxx=first_term+dpx*second_term+dpy*third_term
    
    # Calculating sigma_yy
    first_term=ryy
    second_term=((r[0]/r[-1])*np.cos(t_rad[0]-t_rad[-1]))-1-((a**2)*(r[0]/(r[-1]**3)))*np.sin(t_rad[0])*np.sin(3*t_rad[-1])
    third_term=(2*r[0]/(r[-1]))*np.sin(t_rad[0]-t_rad[-1])
    fourth_term=((a**2)*(r[0]/(r[-1]**3)))*np.sin(t_rad[0])*np.cos(3*t_rad[-1])
    syy=first_term+dpx*second_term+dpy*(third_term-fourth_term)
    
    # Calculating sigma_xy
    first_term=rxy
    
    # When calculating the shear stresses, driving stress should be negative 
    # when applied to the left side of the fracture
    
    # if first_term==rxy:
    #     # if point[0]<crack_center[0]:
    #     if point[0]<crack_center[0] and point[1]<crack_center[1]:
    #         dpx=-dpx
    #         dpy=-dpy
    #     elif point[0]<crack_center[0] and point[1]>crack_center[1]:
    #         dpx=-dpx   
    #         dpy=-dpy
            
    second_term=((r[0]/r[-1])*np.cos(t_rad[0]-t_rad[-1]))-1-(((a**2)*(r[0]/(r[-1]**3)))*np.sin(t_rad[0])*np.sin(3*t_rad[-1]))
    third_term=(((a**2)*(r[0]/(r[-1]**3)))*np.sin(t_rad[0])*np.cos(3*t_rad[-1]))
    sxy=first_term+dpy*second_term+dpx*third_term
    
    # if point[0]<crack_center[0] and point[1]<crack_center[1]:
    #     sxy=-sxy
    
    # elif point[0]<crack_center[0] and point[1]>crack_center[1]:
    #     sxy=-sxy
    
    # else:
    #     sxy=sxy

    # Calculating sigma_xz
    sxz=rxz+dpz*((r[0]/r[-1])*np.cos(t_rad[0]-t_rad[-1])-1)
    
    # Calculating sigma_yz
    syz=ryz+dpz*((r[0]/r[-1])*np.sin(t_rad[0]-t_rad[-1])-1)
    
    # Calculating sigma_zz
    szz=nu*(sxx+syy)
    
    # Calculating t, the maximum shear stress
    t=0.5*(((syy-sxx)**2)+4*(sxy**2))**0.5
        
    return sxx,syy,sxy,sxz,syz,szz,t


########################### Model class ###########################

class crack(object):
    def __init__(self, upper_tip, lower_tip, dpx, dpy, dpz):
        """
        Takes the upper tip, the center, and the
        lower tip of the crack, both as tuples of ints.,
        which should represent meters.
        """
        
        self.center=(upper_tip[0]+lower_tip[0])/2,(upper_tip[1]+lower_tip[1])/2 ;
        self.upper_tip=upper_tip ; 
        self.lower_tip=lower_tip ;
        self.dpx=dpx ; self.dpy=dpy ; self.dpz=dpz
        
    def extent(self):
        """
        Returns the extent (could be vertical or lateral, depending on
        ones desires, of the crack in m.
        """
        h=points_distance(self.upper_tip,self.lower_tip)
        
        return h
    
    def up_tip(self):
        """
        Return the coordinates of the upper tip
        """
        return self.upper_tip
    
    def c(self):
        """
        Return the coordinates of the center.
        """
        return self.center
    
    def low_tip(self):
        """
        Return the coordinates of the lower tip
        """
        return self.lower_tip
    
    def x(self):
        """
        Return the driving stress along x
        """
        return self.dpx
    
    def y(self):
        """
        Return the driving stress along y
        """
        return self.dpy
    
    def z(self):
        """
        Return the driving stress along z
        """
        return self.dpz
    
    def mode_I_crack_displacements(self,v,E):
        """
        Uses the dpx defined with the instance of the class,
        which is the force that acts along the x axis
        and is directed outwards, trying to open the fracture.
        It has a positive sign. Then:
        
        v = Poissons ratio (unitless),
        E = Youngs Modulus in Pa
        
        Returns the magnitudes of the displacement discontinuity on the crack walls,
        ux_pos (m), orthogonal to the crack, in one x direction,
        ux_neg (m), orthogonal to the crack, in the opposite x direction,
        uy_pos (m), parallel to the crack, in one y direciton,
        uy_pos (m), parallel to the crack, in the opposite y direciton.
        
        Follows Eqs. 7.21 in Pollard and Martel, 2020,
        noting that the axes are switched.
        """
        
        dp=self.x()
        h=self.extent()
        a=h/2
        
        y=np.linspace(-h/2,h/2,50)
    
        ux_pos=dp*(((1-v)*(a**2-(y)**2)**0.5))/G(E,v)
        ux_neg=-dp*(((1-v)*(a**2-(y)**2)**0.5))/G(E,v)
        uy_pos=-dp*(((1-2*v)*y)/(2*G(E,v)))
        uy_neg=-dp*(((1-2*v)*y)/(2*G(E,v)))
        
        return ux_pos, ux_neg, uy_pos, uy_neg
    
    def plot_mode_I_crack_displacements(self,v,E):
        """
        Calculates and plots the displacements at the crack's 
        discontinuity given the host rock mechanical properties.
        
        Returns the figure plotting said displacements.
        """
        
        disps=self.mode_I_crack_displacements(v,E)
        h=self.extent()
        length=np.linspace(-h/2,h/2,50)
        
        figure=plt.figure(figsize=(7,4))
        ax=figure.add_subplot(111, xlabel='y distance from crack center (m)', ylabel=r'u$_i$ (m)')
        
        ax.plot(length,disps[0],c='purple',ls='--',lw=2,label=r'u$_x$, x>0')
        ax.plot(length,disps[1],c='purple',ls=':',lw=2,label=r'u$_x$, x<0')
        ax.scatter(length,disps[2],c='peru',facecolor='white',alpha=0.5,edgecolor='black',marker='o',s=40,label=r'u$_y$, x>0')
        ax.scatter(length,disps[3],c='brown',marker='o',s=5,label=r'u$_y$, x<0')
        
        ax.legend(title=r'$\sigma_e$ = '+str(self.dpx/1E6)+' MPa\nv ='+str(v)+'\nE = '+str(E/1E9)
                  +' GPa\na = '+str(round(h/2/1E3,2))+' km',bbox_to_anchor=(1,1))

        return figure

class model(object):
    def __init__(self, x_limits,y_limits,x_cellsize,y_cellsize,y_position_layers,layers_E,layers_v,layers_d,layers_mu,layers_T, g):
        """
        Parameters
        --------------
        x_limits: a tuple of two ints defining the x boundaries of the model
        y_limits: a tuple of two ints defining the y boundaries of the model
        both in meters.
        
        x_cellsize: a float/int larger than 0 defining the size of the cells along x
        y_cellsize: a float/int larger than 0 defining the size of the cells along x
        where the cell size is in meters.
        
        y_position_layers: a tuple of float/ints defining the depth at which 
        each of the layers begins, from bottom to top
        
        Tips:
            For a 1-layer model, keep y_position_layers as a single-value
            array-like structure equal to the top of the model, 
            and layers_E, layers_v, etc. as single-value arrays-like objects as well.
            
            For a multi-layer model, set the base of the deepest layer in
            y_position_layers below the base of the model for a better
            visualization of contours when plotting stresses.
        
        layers_E: an array-like object with floats/ints defining the Youngs Modulus 
        of each of the layers,in Pa
        
        layers_v: an array-like object with floats/ints defining the Poissons Ratio 
        of each of the layers, unitless
        
        layers_d: an array-like object with floats/ints defining the density
        of each of the layers,in kg m-3
        
        layers_mu: an array-like object with floats/ints defining the coefficient of 
        internal friction of each of the layers, unitless
        
        layers_T: an array-like object with floats/ints defining the max
        tensile stress for each of the layers, in Pa
        
        g: gravitational acceleration in ms-2 for this model.
        
        The cohesion of each layer is not an input: it is implicit in the
        modified Griffith criterion, C = 2|T| (Rivas-Dorado et al., 2023,
        eq. 12b), and is stored in layers_c for the grid and the labels.
        """
        
        self.x_limits=x_limits ; self.y_limits=y_limits
        self.x_cellsize=x_cellsize ; self.y_cellsize=y_cellsize
        self.y_position_layers=y_position_layers
        self.layers_E=layers_E ; self.layers_v=layers_v ; self.layers_d=layers_d
        self.layers_mu=layers_mu; self.layers_T=layers_T
        
        # Cohesion implicit in the modified Griffith criterion: C = 2|T|
        self.layers_c=2*np.abs(np.asarray(self.layers_T,dtype=float))
        
        # The coulomb angle is defined implicitly after mu for each layer
        self.layers_coula=0.5*np.arctan(1/np.asarray(self.layers_mu,dtype=float))
    
        # self.layers_m=layers_m
        self.g=g
        
    def n_layers(self):
        """
        Returns the number of layers of the model, as an int.
        A single value in y_position_layers defines a 1-layer model;
        n+1 values (bases from bottom to top, plus the top) define n layers.
        """
        bounds=np.atleast_1d(np.asarray(self.y_position_layers,dtype=float))
        
        if bounds.size==1:
            return 1
        
        return bounds.size-1
    
    def layer_index(self,y):
        """
        Parameters
        ----------
        y: a float/int or an array-like object with y coordinates, in m.
        
        Returns
        ----------
        An int array with the shape of y containing, for each y, the index
        of the layer it belongs to (0 = deepest layer). A layer contains its
        base and extends up to, but not including, the base of the next layer.
        Points below the base of the deepest layer are assigned to it, and
        points above the top of the model to the uppermost layer.
        """
        y=np.asarray(y,dtype=float)
        
        if self.n_layers()==1:
            return np.zeros(np.shape(y),dtype=int)
        
        bounds=np.asarray(self.y_position_layers,dtype=float)
        if np.any(np.diff(bounds)<=0):
            raise InputError('y_position_layers must be strictly increasing, from bottom to top')
        
        index=np.searchsorted(bounds,y,side='right')-1
        
        return np.clip(index,0,self.n_layers()-1)
    
    def layer_values(self,values,name):
        """
        Returns the per-layer values of a property as a 1D float array,
        after checking that there is exactly one value per layer.
        """
        values=np.atleast_1d(np.asarray(values,dtype=float))
        
        if values.size!=self.n_layers():
            raise InputError(name+' has '+str(values.size)+' value(s), but the model has '
                             +str(self.n_layers())+' layer(s)')
        
        return values
        
    def grid(self): 
        """
        Returns a 3D array where the first two arrays are the 
        X and Y coordinates of the grid, and the rest are the 
        E, v, density, cohesion (2|T|), internal friction coeff., Coulomb angle
        and tensile strength values at each node of the grid.
        """
        
        x_cells=np.arange(self.x_limits[0],self.x_limits[1],self.x_cellsize)
        y_cells=np.arange(self.y_limits[0],self.y_limits[1],self.y_cellsize)
        x, y = np.meshgrid(x_cells+self.x_cellsize/2,y_cells)
        
        # Each node takes the properties of the layer that contains it
        index=self.layer_index(y)
        
        properties=[(self.layers_E,'layers_E'),(self.layers_v,'layers_v'),
                    (self.layers_d,'layers_d'),(self.layers_c,'layers_c'),
                    (self.layers_mu,'layers_mu'),(self.layers_coula,'layers_coula'),
                    (self.layers_T,'layers_T')]
        
        values=[self.layer_values(prop,name)[index] for prop,name in properties]
                
        grid=np.array([x, y]+values)
        
        return grid
    
    def resolution(self):
        """
        Returns the resolution of the grid.
        """
        
        grid=self.grid()
        x_cells=len(grid[0][0])
        y_cells=len(grid[1].transpose()[0])
        
        return (str(x_cells)+' x '+str(y_cells)+' = '+str(x_cells*y_cells)+' cells'), x_cells*y_cells
    
    def show_properties(self):
        """
        Returns a plot that shows and summarises the properties of the model.
        """
        
        grid=self.grid()
        
        figure=plt.figure(figsize=(6,6),tight_layout=True)
        ax=figure.add_subplot(111, xlim=self.x_limits,ylim=self.y_limits)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        if self.n_layers()==1:
            ax.text(0.93,0.93,
                    'E = '+str(self.layer_values(self.layers_E,'layers_E')[0]/1E9)+' GPa\n'+'v = '+str(self.layer_values(self.layers_v,'layers_v')[0])
                    +'\n'+r'$\rho$ = '+str(self.layer_values(self.layers_d,'layers_d')[0])+r' kg/m$^3$'
                    +'\n'+r'C$_o$ = '+str(self.layer_values(self.layers_c,'layers_c')[0]/1e6)+' MPa'
                    +'\n'+r'$\mu$ = '+str(self.layer_values(self.layers_mu,'layers_mu')[0]),
                    bbox=dict(fc='white', ec='black'),
                    horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes,fontsize=14, zorder=5)
        
        else:
            i=0
            for y,E,v,d,c,m in zip(self.y_position_layers[1:],self.layers_E, self.layers_v, self.layers_d, self.layers_c, self.layers_mu):
                x=self.x_limits[-1]-0.05*points_distance((self.x_limits[0],0),(self.x_limits[-1],0))
                y=y-0.05*points_distance((self.x_limits[-1],self.y_position_layers[i]),(self.x_limits[-1],y))
                i=i+1
                ax.text(x,y,
                        'E = '+str(E/1E9)+' GPa\n'+'v = '+str(v)+'\n'
                        +r'$\rho$ = '+str(d)+r' kg/m$^3$'
                        +'\n'+r'C$_o$ = '+str(c/1e6)+' MPa'
                        +'\n'+r'$\mu$ = '+str(m),
                        bbox=dict(fc='white', ec='black'), 
                        horizontalalignment='right', verticalalignment='top', fontsize=11)
            
        ax.pcolormesh(grid[0],grid[1],grid[2], shading='auto')
        
        return figure
        
    def stresses_at_depth(self):
        """
        Calculates the lithostatic stresses at depth in Pa,
        sigma1=sigma3 (s1=s3, the horizontal stresses), 
        and sigma2 (s2, the vertical stress), 
        for every point of the grid. 
        
        Assumes that the y coordinates are negative depth, with the
        surface at y=0, and uniaxial lithostatic conditions. The vertical
        stress is the weight of the overlying column, integrated layer by
        layer: the uppermost layer is assumed to extend up to the surface
        and the deepest one indefinitely downwards.
        
        Needs g in m/s2. 
        
        Returns two 2D arrays; the first contains syy, the vertical stress,
        the second, sxx=szz, the horizontal stresses, both in Pa. 
        Because the lithostatic stress is always compressive 
        and the code uses the tension-positive convention,
        the returned stresses are negative.
        """
        
        g=self.g
        
        grid=self.grid()
        y=grid[1] ; v=grid[3]
        
        densities=self.layer_values(self.layers_d,'layers_d')
        n=self.n_layers()
        
        # Lower and upper limits of each layer
        if n==1:
            lower=np.array([-np.inf]) ; upper=np.array([0.0])
        else:
            bounds=np.asarray(self.y_position_layers,dtype=float)
            lower=bounds[:-1].copy() ; upper=bounds[1:].copy()
            lower[0]=-np.inf ; upper[-1]=0.0
        
        # Weight of the column above each point: thickness of every layer
        # between the point and the surface times its density and g
        syy=np.zeros(np.shape(y))
        for d,low,up in zip(densities,lower,upper):
            thickness=np.clip(up-np.maximum(low,y),0,None)
            syy=syy+d*g*thickness
        
        sxx=syy*(v/(1-v))
        
        return -syy*1,-sxx*1
    
    def model_displacements(self,crack):
        """
        Parameters
        ----------
        crack: a object,
        
        Returns
        ----------
        Three 2D arrays, all values in m:
        dx: displacement along x 
        dy: displacement along y
        dxy: displacement in xy
        """
        
        grid=self.grid()
        x, y, Es, vs = grid[0], grid[1], grid[2], grid[3] 
                
        upper_tip=crack.up_tip() ; crack_center=crack.c() ; lower_tip=crack.low_tip()
        dpx=crack.x() ; dpy=crack.y() ; dpz=crack.z()
        
        dx=[] ; dy=[] ; dz=[]
    
        for xrow,yrow,Erow,vrow in zip(x,y, Es,vs):
            dx_temp=[] ; dy_temp=[] ; dz_temp=[]
            for i,j,E,v in zip(xrow,yrow,Erow,vrow):
                point=(i,j)
                d=displacements(dpx,dpy,dpz,v,E,upper_tip,crack_center,lower_tip,point)
                dx_temp.append(d[0])
                dy_temp.append(d[1])
                dz_temp.append(d[2])
            
            dx.append(np.asarray(dx_temp))
            dy.append(np.asarray(dy_temp))
            dz.append(np.asarray(dz_temp))
        
        dx=np.asarray(dx)
        dy=np.asarray(dy)
        dz=np.asarray(dz)
                
        return dx,dy,dz
    
    def plot_displacement(self, crack, displacement, option, step):
        """
        Parameters
        ----------
        crack: a object,
        displaement: a str, the desired displacement to plot, 
            ux: displacement along x 
            uy: displacement along y
            uxy: displacement in xy
        option: a str, contours or vectors, to plot the displacement as
        step: if selecting contours, a non-0 int indicating the contour step
        
        Returns
        ----------
        A figure object plotting the desired displacement
        """
        
        grid=self.grid()
        
        upper_tip , lower_tip = crack.up_tip(), crack.low_tip()
        
        displacements=self.model_displacements(crack)
        dx, dy= displacements[0] , displacements[1]
        x, y = grid[0], grid[1]
              
        figure=plt.figure(figsize=(6,6),tight_layout=True)
        ax=figure.add_subplot(111, xlim=self.x_limits,ylim=self.y_limits)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.tick_params(labelsize=12, axis='both')
        
        ax.plot((upper_tip[0],lower_tip[0]),(upper_tip[1],lower_tip[1]),c='red',lw=3, label='Crack')
        
        if displacement=='ux':
            dp=dx ; ds=0 ; dt=dx ; plot_params=boundaries_contours(dp, step) ; label=r'u$_x$ (m)'
            
        elif displacement=='uy':
            dp=0 ; ds=dy ; dt=dy ; plot_params=boundaries_contours(ds, step) ; label=r'u$_y$ (m)'
            
        elif displacement=='uxy':
            dp=dx ; ds=dy ; dt=(dx+dy)/2 ; label=r'u$_{xy}$ (m)'
            
        if option=='vectors':
            plot=ax.quiver(x,y,dp,ds,dt, width=0.0050, scale=np.amax(dt)/0.05, cmap='viridis')
            cbar=figure.colorbar(plot,extend='both')
            cbar.set_label(label, rotation=270, labelpad=20, fontsize=15)
            cbar.ax.tick_params(labelsize=12)

            
        elif option=='contours':
            
            plot=ax.pcolormesh(x,y,dt,cmap='viridis',vmin=plot_params[0], vmax=plot_params[1], shading='nearest')
            cbar=figure.colorbar(plot,extend='both')
            cbar.set_label(label, rotation=270, labelpad=20, fontsize=15)
            cbar.ax.tick_params(labelsize=12)
            
            contours=ax.contour(x,y,dt, levels=plot_params[-1], colors='black',linewidths=0.75,linestyles='--')
        
            if step<=0.001:
                ax.clabel(contours,fmt='%1.4f')
            elif step<0.01:
                ax.clabel(contours,fmt='%1.3f')
            elif 0.01<=step<0.1:
                ax.clabel(contours,fmt='%1.2f')
            elif step==0.1:
                ax.clabel(contours,fmt='%1.1f')
            elif step>=0.1:
                ax.clabel(contours,fmt='%1.1f')
        else:
            raise InputError('Invalid displacement. Input "ux", "uy", or "uxy" instead.')
                
        ax.text(0.97,0.10,r'$\sigma_I$ = '+str(round(crack.x()/1E6,2))
                +' MPa\n'+r'$\sigma_{II}$ = '+str(round(crack.y()/1E6,2))+' MPa',
                bbox=dict(fc='white', ec='black'),
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes,fontsize=11, zorder=5)
        
        return figure
        
        
    def model_stresses(self, crack, rxx, ryy, boolean):
        """
        Parameters
        ----------
        crack: a crack object,        
        rxx: remote stress normal to plane x
        ryy: remote stress normal to plane y. Both in MPa. Both may be 0.
        boolean: a bool. If True, includes in the calculation the stresses
        associated with the lithostatic load at each point of the grid.
        If False, it does not do this. It must be a real boolean, not a str.
        
        Returns
        ----------
        Seven 2D arrays, all values in Pa:
        sxa: resulting normal stress to the x plane
        sya: resulting normal stress to the y plane
        sav: average stress 
        sxya: resulting shear stress in the xy plane 
        sxza: resulting shear stress in the xz plane 
        szza: resulting normal stress to the z plane 
        ta: shear stress
        """        
        grid=self.grid()
        x, y, Es, vs= grid[0], grid[1], grid[2], grid[3]
        
        upper_tip=crack.up_tip() ; crack_center=crack.c() ; lower_tip=crack.low_tip() ; a2=crack.extent()
        dpx=crack.x() ; dpy=crack.y() ; dpz=crack.z()
        
        # Following the remote boundary conditions
        rxy=0 ; rxz=0 ; ryz=0
        
        sxa=[] ; sya=[] ; sxya=[] ; sxza=[] ; syza=[] ; szza=[] ; ta=[]
          
        # A str such as 'True' would match neither branch below and return
        # empty arrays without any warning, so it is rejected here
        boolean=require_bool(boolean,'boolean')

        if boolean:
            model_stress=self.stresses_at_depth()
            ryy, rxx= ryy+model_stress[0], rxx+model_stress[1]
            
            for xrow,yrow,Erow,vrow,rxrow,ryrow in zip(x,y, Es,vs,rxx,ryy):
                sx=[] ; sy=[] ; sxy=[] ; sxz=[] ; syz=[] ; szz=[] ; t=[]
                for i,j,E,v,rx,ry in zip(xrow,yrow,Erow,vrow,rxrow,ryrow):
                    point=(i,j)
                    d=stresses(dpx,dpy,dpz,rx,ry,rxy,rxz,ryz,v,E,upper_tip,crack_center,lower_tip,a2/2,point)
                    sx.append(d[0])
                    sy.append(d[1])
                    sxy.append(d[2])
                    sxz.append(d[3])
                    syz.append(d[4])
                    szz.append(d[5])
                    t.append(d[6])
                
                sxa.append(np.asarray(sx))
                sya.append(np.asarray(sy))
                sxya.append(np.asarray(sxy))
                sxza.append(np.asarray(sxz))
                syza.append(np.asarray(syz))
                szza.append(np.asarray(szz))
                ta.append(np.asarray(t))
                        
        else:
            rxx, ryy, rxy = rxx, ryy, 0
            
            for xrow,yrow,Erow,vrow in zip(x,y,Es,vs):
                sx=[] ; sy=[] ; sxy=[] ; sxz=[] ; syz=[] ; szz=[] ; t=[]
                for i,j,E,v in zip(xrow,yrow,Erow,vrow):
                    point=(i,j)
                    d=stresses(dpx,dpy,dpz,rxx,ryy,rxy,rxz,ryz,v,E,upper_tip,crack_center,lower_tip,a2/2,point)
                    sx.append(d[0])
                    sy.append(d[1])
                    sxy.append(d[2])
                    sxz.append(d[3])
                    syz.append(d[4])
                    szz.append(d[5])
                    t.append(d[6])
                
                sxa.append(np.asarray(sx))
                sya.append(np.asarray(sy))
                sxya.append(np.asarray(sxy))
                sxza.append(np.asarray(sxz))
                syza.append(np.asarray(syz))
                szza.append(np.asarray(szz))
                ta.append(np.asarray(t))
                
        sxa=(np.asarray(sxa)) ; sya=(np.asarray(sya)) ; sav=(sxa+sya)/2
        sxya=(np.asarray(sxya)) ; sxza=(np.asarray(sxza)) ; syza=(np.asarray(syza))
        szza=(np.asarray(szza)) ; ta=(np.asarray(ta))
            
        return sxa, sya, sav, sxya, sxza, szza, ta
    
    def plot_stresses(self,crack,stress,s11,s22,boolean,step):
        """
        Parameters
        ----------
        crack: a crack object,
        stress: a str indicating the stress to be plotted,
            'sxa', the stress along the x axis
            'sya', the stress along the y axis
            'sav', the average stress 
            'sxya', the shear stress on x along y, and on y along x
            'sxza', the shear stress on x along z
            'sxya', the normal stress along z
            'ta', the shear stress on the principal planes
        s11 : array-like, the remote stress acting along the x axis, in Pa,
        s22 : array-like, the remote stress acting along the y axis, in Pa,
        boolean: a bool. True, to indicate that the lithostatic stress must
        be calculated, False, for the opposite. Not a str.
        step: int, the contour step
        
        Returns
        ----------
        A figure object plotting the desired stress in MPa
        """
        
        grid=self.grid()
        x, y, = grid[0], grid[1]
        
        upper_tip=crack.up_tip() ; lower_tip=crack.low_tip()
        dpx=crack.x() ; dpy=crack.y()
        
        # soi=stress of interest
        if stress=='sxa':
            soi=self.model_stresses(crack,s11,s22,boolean)[0]
            label=r'$\sigma_{xx}$ (MPa)'
            
        elif stress=='sya':
            soi=self.model_stresses(crack,s11,s22,boolean)[1]
            label=r'$\sigma_{yy}$ (MPa)'
            
        elif stress=='sav':
            soi=self.model_stresses(crack,s11,s22,boolean)[2]
            label=r'$\sigma_{av}$ (MPa)'
            
        elif stress=='sxya':
            soi=self.model_stresses(crack,s11,s22,boolean)[3]
            label=r'$\sigma_{xy}$ (MPa)'
            
        elif stress=='sxza':
            soi=self.model_stresses(crack,s11,s22,boolean)[4]
            label=r'$\sigma_{xz}$ (MPa)'
            
        elif stress=='sza':
            soi=self.model_stresses(crack,s11,s22,boolean)[5]
            label=r'$\sigma_{zz}$ (MPa)'
            
        elif stress=='ta':
            soi=self.model_stresses(crack,s11,s22,boolean)[6]
            label=r'$\tau$ (MPa)'
              
        figure=plt.figure(figsize=(6,6),tight_layout=True)
        ax=figure.add_subplot(111, xlim=self.x_limits, ylim=self.y_limits)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.tick_params(labelsize=12, axis='both')
        
        ax.plot((upper_tip[0],lower_tip[0]),(upper_tip[1],lower_tip[1]),c='red',lw=3)
        
        soi=soi/1e6
        
        plot_params=boundaries_contours(soi, step)

        plot=ax.pcolormesh(x,y,soi,cmap='viridis',vmin=plot_params[0], vmax=plot_params[1], shading='nearest')
        
        ax.grid(which='both',alpha=0.5)
                    
        cbar=figure.colorbar(plot,extend='both')
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        
        contours=ax.contour(x,y,soi, levels=plot_params[-1], colors='black',linewidths=0.75,linestyles='--')
        
        if step<=0.001:
            ax.clabel(contours,fmt='%1.4f')
        elif step<0.01:
            ax.clabel(contours,fmt='%1.3f')
        elif 0.01<=step<=1:
            ax.clabel(contours,fmt='%1.2f')
        elif step>=1:
            ax.clabel(contours,fmt='%1.1f')

        # ax.text(0.97,0.09,r'$\sigma_I$ = '+str(round(dpx/1E6,2))
        #         +' MPa\n'+r'$\sigma_{II}$ = '+str(round(dpy/1E6,2))+' MPa', 
        #         bbox=dict(fc='white', ec='black'),
        #         horizontalalignment='right', verticalalignment='top',
        #         transform=ax.transAxes,fontsize=11, zorder=5)
        
        return figure
    
    def principal_stresses(self, crack, s11, s22, boolean):
        """

        Parameters
        ----------
        crack: a crack object,
        s11 : array-like, the remote stress acting along the x axis, in Pa,
        s12 : array-like, the remote stress acting along the y axis, in Pa,
        boolean: a bool. True, to indicate that the lithostatic stress must
        be calculated, False, for the opposite. Not a str.
        
        Returns
        -------
        Principal stresses in a compression-positive convention.
        All 2D arrays with values in Pa:
        s1: maximum compressive stress at each point of the grid, in Pa
        s3: minimum compressive stress at each point of the grid, in Pa
        new_s1a: float, angle from the x axis,
        to the s1 principal plane, in degrees
        new_s3a: float, angle from the x axis,
        to the s3 principal plane, in degrees
        """
        
        dpx=crack.x() ; dpy=crack.y()
        
        s=self.model_stresses(crack, s11, s22, boolean)
        
        sxx=-s[0] ; syy=-s[1] ; sxy=-s[3]
                                
        s1=(0.5*(sxx+syy)+(((sxx-syy)*0.5)**2+sxy**2)**0.5)
        s3=(0.5*(sxx+syy)-(((sxx-syy)*0.5)**2+sxy**2)**0.5)
            
        # Jaeger, 2007
        # To make sure that when the crack and remote stresses are zero the 
        # stress field is dominated by the vertical lithostatic stress
        if dpx==0 and s11==0:
            s1a=np.full(np.shape(s1),90)
            s3a=np.full(np.shape(s3),0)
        
        else:
            # This is relative to the y axis
            # s1a=np.degrees(np.arctan(2*sxy/(sxx-syy)))/2
            # s3a=s1a+90
            
            # This is relative to the x axis
            s1a_lower_term=(sxx-syy)+(4*(sxy**2)+(sxx-syy)**2)**0.5
            s3a_lower_term=(sxx-syy)-(4*(sxy**2)+(sxx-syy)**2)**0.5

            # Making sure there are no-zero division errors
            s1a_lower_term=np.where(s1a_lower_term==0, 0.001, s1a_lower_term)
            s3a_lower_term=np.where(s3a_lower_term==0, 0.001, s3a_lower_term)
            
            s1a=np.degrees(np.arctan(2*sxy/s1a_lower_term))
            s3a=np.degrees(np.arctan(2*sxy/s3a_lower_term))
                
        return s1, s3, s1a, s3a
    
    def plot_principal_stresses(self, crack, stress, s11, s22, boolean, step):
        """

        Parameters
        ----------
        crack: a crack object,
        stress: str, specifies the stress to be plotted, 
            'principal' for the maximum compressive stress, 
            'secondary', for the minimum compressive stress,
            'angle', for the angle between the vertical and the principal stress,
        s11 : array-like, the remote stress acting along the x axis, in Pa,
        s11 : array-like, the remote stress acting along the y axis, in Pa,
        boolean: a bool. True, to indicate that the lithostatic stress must
        be calculated, False, for the opposite. Not a str.
        step: int, the desired contour interval in MPa

        Returns
        -------
        A figure object plotting the desired principal stress in MPa, or the
        2theta angle in degrees

        """
        grid=self.grid()
        x, y = grid[0], grid[1]
        
        upper_tip=crack.up_tip() ; lower_tip=crack.low_tip()
        dpx=crack.x() ; dpy=crack.y()
        
        ps=self.principal_stresses(crack,s11,s22,boolean)    
        p=ps[0] ; s=ps[1] ; angles=ps[-1]
        
        figure=plt.figure(figsize=(6,6),tight_layout=True)
        ax=figure.add_subplot(111, xlim=self.x_limits, ylim=self.y_limits)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.tick_params(labelsize=12, axis='both')
        ax.plot((upper_tip[0],lower_tip[0]),(upper_tip[1],lower_tip[1]),c='red',lw=3)        
        
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        
        if stress=='principal':
            soi=self.principal_stresses(crack,s11,s22,boolean)[0]
            soi=soi/1e6 ; plot_params=boundaries_contours(soi, step)
            label=r'$\sigma_1$ (MPa)'
            
        elif stress=='secondary':
            soi=self.principal_stresses(crack,s11,s22,boolean)[1]
            soi=soi/1e6 ; plot_params=boundaries_contours(soi, step)
            label=r'$\sigma_3$ (MPa)'
            
        elif stress=='angle':
            soi=self.principal_stresses(crack,s11,s22,boolean)[2]
            soi=soi ; plot_params=boundaries_contours(soi, step)
            label=r'$\theta$ (°)'
        
        plot=ax.pcolormesh(x,y,soi,cmap='viridis', vmin=plot_params[0], vmax=plot_params[1], shading='nearest')
            
        cbar=figure.colorbar(plot,extend='both')
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        
        contours=ax.contour(x,y,soi, levels=plot_params[-1], colors='black',linewidths=0.75,linestyles='--')
        
        if step<=0.001:
            ax.clabel(contours,fmt='%1.4f')
        elif step<0.01:
            ax.clabel(contours,fmt='%1.3f')
        elif 0.01<=step<=1:
            ax.clabel(contours,fmt='%1.2f')
        elif step>=1:
            ax.clabel(contours,fmt='%1.1f')
        
        # ax.text(0.97,0.09,r'$\sigma_I$ = '+str(round(dpx/1E6,2))
        #         +' MPa\n'+r'$\sigma_{II}$ = '+str(round(dpy/1E6,2))+' MPa', 
        #         bbox=dict(fc='white', ec='black'),
        #         horizontalalignment='right', verticalalignment='top',
        #         transform=ax.transAxes,fontsize=11, zorder=5)
        
        return figure
    
    def plot_stress_trajectories(self, crack, planes,  s11, s22, boolean):
        """

        Parameters
        ----------
        crack: a crack object,
        planes: a bool. If True, plots the optimal shear planes defined by mu.
        If False, it does not do this. It must be a real boolean, not a str.
        s11 : array-like, the remote stress acting along the x axis, in Pa,
        s11 : array-like, the remote stress acting along the y axis, in Pa,
        boolean: a bool. True, to indicate that the lithostatic stress must
        be calculated, False, for the opposite. Not a str.

        Returns
        -------
        A figure plotting the trajectories of the principal stresses and
        optionally the orientation of the optimal shear planes defined by mu.
        
        """

        # Failing early, before the stresses are calculated, if the input is invalid
        planes=require_bool(planes,'planes')
        
        grid=self.grid()
        x, y = grid[0], grid[1]
        s, mu, coula = grid[5], grid[6], grid[7]
        
        upper_tip=crack.up_tip() ; lower_tip=crack.low_tip()
        dpx=crack.x() ; dpy=crack.y()
        
        ps=self.principal_stresses(crack,s11,s22,boolean)    
        p=ps[0]/1e6 ; s=ps[1]/1e6 ; p_angle=ps[2] ; s_angle=ps[-1]
          
        p_angles=np.deg2rad(p_angle)
        s_angles=np.deg2rad(s_angle)
                
        figure=plt.figure(figsize=(7,6),tight_layout=True)
        ax=figure.add_subplot(111, xlim=self.x_limits, ylim=self.y_limits)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.tick_params(labelsize=12, axis='both')
        ax.plot((upper_tip[0],lower_tip[0]),(upper_tip[1],lower_tip[1]),c='red',lw=3)        
        
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
                
        plot1=ax.quiver(x,y,np.cos(p_angles),np.sin(-p_angles),p,headlength=0, pivot='middle', scale=0.0090, units='xy', width=10, headwidth=1, cmap='copper_r')
        plot2=ax.quiver(x,y,np.cos(s_angles),np.sin(-s_angles),s,headlength=0, pivot='middle', scale=0.0090, units='xy', width=10, headwidth=1, cmap='cool')
        
        # plot1=ax.quiver(x,y,sins,cosins, p, headlength=0, pivot='middle', scale=0.0030, units='xy', width=30, headwidth=1, cmap='copper_r')
        # plot2=ax.quiver(x,y,sins_ortho,cosins_ortho,s,headlength=0, pivot='middle', scale=0.0045, units='xy', width=30, headwidth=1, cmap='cool')
        
        cbar=figure.colorbar(plot1,extend='both',shrink=0.60, aspect=30, pad=0.05)
        cbar.set_label(r'$\sigma_1$ (MPa)', rotation=270, labelpad=12, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        cbar=figure.colorbar(plot2,extend='both',shrink=0.60, aspect=30)
        cbar.set_label(r'$\sigma_3$ (MPa)', rotation=270, labelpad=16, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        if planes:
            # beta are the angles between the two shear planes and the vertical
            beta=p_angles-coula
            beta_p=2*p_angles-beta
            
            # '#FF1493', #3FFF00
            ax.quiver(x,y,np.cos(beta),np.sin(-beta),headlength=0, pivot='middle', scale=0.0090, units='xy', width=10, headwidth=1, color='#4DAD4B',alpha=0.8)
            ax.quiver(x,y,np.cos(beta_p),np.sin(-beta_p),headlength=0, pivot='middle', scale=0.0090, units='xy', width=10, headwidth=1, color='#FA422B',alpha=0.8)
            
        # ax.legend(loc=1)
        
        # ax.text(0.96,0.14,r'$\sigma_I$ = '+str(round(dpx/1E6,2))+' MPa; '+r'$\sigma{^r_x}$ = '+str(round(s11/1e6,0))+' MPa\n'
        #         +r'$\alpha$ = '+str(angle)+'°; '+r'$\rho$ = '+str(self.layers_d)+' kg m$^{-3}$\n'
        #         +r'$\mu$ = '+str(self.layers_mu)+'; '+r'$\theta_{opt}$ = '+str(np.round(90-np.degrees(self.layers_coula),2))+'°',
        #         bbox=dict(fc='white', ec='black'),
        #         horizontalalignment='right', verticalalignment='top',
        #         transform=ax.transAxes,fontsize=11, zorder=5)
        
        return figure
        
    def plane_stresses(self, crack, s11, s22, boolean):
        """
        Parameters
        ----------
        crack: a crack object,
        s11 : array-like, the remote stress acting along the x axis, in Pa,
        s11 : array-like, the remote stress acting along the y axis, in Pa,
        boolean: a bool. True, to indicate that the lithostatic stress must
        be calculated, False, for the opposite. Not a str.

        Returns
        -------
        Two 2D arrays, in Pa, of the stresses acting on the optimal planes
        defined by the coefficient of internal friction,
        sn: the normal stress  
        ss: the shear stress    
        """
        grid=self.grid()
        x, y = grid[0], grid[1] 
        s, mu, coula = grid[5], grid[6], grid[7]
        
        
        ########### With the principal stresses ###########
        s=self.principal_stresses(crack, s11, s22, boolean)
        s1=s[0] ; s3=s[1]
        
        # Pollard and Martel, 2020

        sn=0.5*(s1+s3)+0.5*(s1-s3)*np.cos(2*coula)
        ss=0.5*(s1-s3)*np.sin(2*coula)

        return sn, ss
    
    
    def plot_plane_stresses(self, crack, stress, s11, s22, angle, boolean, step):
        """
        Parameters
        ----------
        crack: a crack object,
        stress: a str, indicating the stress to be plotted,
           'sn': for the normal stress  
           'ss': for the shear stress    
        s11 : array-like, the remote stress acting along the x axis, in Pa,
        s11 : array-like, the remote stress acting along the y axis, in Pa,
        boolean: a bool. True, to indicate that the lithostatic stress must
        be calculated, False, for the opposite. Not a str.

        Returns
        -------
        A figure plotting the desired stress in MPa    
        """
        
        grid=self.grid()
        x, y = grid[0], grid[1]
        s, mu, coula = grid[5], grid[6], grid[7]
        
        upper_tip=crack.up_tip() ; lower_tip=crack.low_tip()
        dpx=crack.x() ; dpy=crack.y()
                
        if stress=='normal':
            soi=self.plane_stresses(crack, s11, s22, boolean)[0]
            soi=soi/1e6 ; plot_params=boundaries_contours(soi, step)
            label=r'$\sigma_n$ (MPa)'
            
        elif stress=='shear':
            soi=self.plane_stresses(crack, s11, s22, boolean)[1]
            soi=soi/1e6 ; plot_params=boundaries_contours(soi, step)
            label=r'$\sigma_s$ (MPa)'
                            
        figure=plt.figure(figsize=(6,6),tight_layout=True)
        ax=figure.add_subplot(111, xlim=self.x_limits, ylim=self.y_limits)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.tick_params(labelsize=12, axis='both')
        ax.plot((upper_tip[0],lower_tip[0]),(upper_tip[1],lower_tip[1]),c='red',lw=3)        
        
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        
        plot=ax.pcolormesh(x,y,soi,cmap='viridis', vmin=plot_params[0], vmax=plot_params[1], shading='nearest')
            
        cbar=figure.colorbar(plot,extend='both')
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        
        contours=ax.contour(x,y,soi, levels=plot_params[-1], colors='black',linewidths=0.75,linestyles='--')
        
        if step<=0.001:
            ax.clabel(contours,fmt='%1.4f')
        elif step<0.01:
            ax.clabel(contours,fmt='%1.3f')
        elif 0.01<=step<=1:
            ax.clabel(contours,fmt='%1.2f')
        elif step>=1:
            ax.clabel(contours,fmt='%1.1f')
        
        if self.n_layers()==1:
            ax.text(0.97,0.14,r'$\sigma_I$ = '+str(round(dpx/1E6,2))+' MPa ; '+r'$\sigma_{II}$ = '+str(round(dpy/1E6,2))+' MPa\n'
                    +r'$\alpha$ = '+str(angle)+'° ; '+ r'$\rho$ = '+str(self.layer_values(self.layers_d,'layers_d')[0])+' kg m$^3$'
                    +'\nC = '+str(self.layer_values(self.layers_c,'layers_c')[0]/1e6)+' MPa ; '+r'$\mu$ = '+str(self.layer_values(self.layers_mu,'layers_mu')[0])+'; '
                    +r'$\theta_{opt}$ = '+str(round(90-np.degrees(self.layer_values(self.layers_coula,'layers_coula')[0]),2))+'°',
                    bbox=dict(fc='white', ec='black'),
                    horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes,fontsize=11, zorder=5)
        else:
            ax.text(0.98,0.14,r'$\sigma_I$ = '+str(round(dpx/1E6,2))+' MPa\n'+r'$\sigma_{II}$ = '+str(round(dpy/1E6,2))+' MPa\n'
                    +r'$\alpha$ = '+str(angle)+'°'
                    +'\nC = '+str(np.asarray(self.layers_c,dtype=float)/1e6)+' MPa ; '+r'$\mu$ = '+str(np.asarray(self.layers_mu,dtype=float))+'; '
                    +r'$\theta_{opt}$ = '+str(np.round(90-np.degrees(self.layers_coula),2))+'°',
                    bbox=dict(fc='white', ec='black'),
                    horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes,fontsize=11, zorder=5)
                
        return figure