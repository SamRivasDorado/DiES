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
from Helper_functions import linear
from Helper_functions import boundaries_contours

# Container of the computed fields (Phase 1, Nov 2026)
from results import DiESResults

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

########################### Geometry of the crack: local frame and tripolar coordinates ###########################

# The analytical solutions below (Pollard and Segall, 1987; Pollard and Martel, 2020)
# are written for a straight crack lying along the axis of its own local frame.
# Every calculation therefore follows three steps:
#
#   1. Build the local frame of the crack (crack_frame): its center, a unit vector t
#      along the crack and a unit vector n normal to it.
#   2. Express each point of interest in that frame (local_coordinates), and from
#      there obtain the signed tripolar coordinates r, theta, r1, theta1, r2, theta2,
#      R, THETA (tripolar_coordinates).
#   3. Evaluate the equations in the local frame, where they are exact, and rotate the
#      resulting displacement vector and stress tensor back to the global X-Y axes
#      (rotate_vector_to_global, rotate_stress_to_global).
#
# See docs/crack_coordinates_and_rotation.md for the derivation and the conventions.

def crack_frame(upper_tip,lower_tip):
    """
    Local reference frame of a straight 2D crack.

    Parameters
    ----------
    upper_tip, lower_tip: the (x, y) coordinates of the two tips of the crack, in m.
    If the 'upper' tip is actually deeper than the 'lower' one, they are swapped,
    so that the frame does not depend on the order in which the tips are given.
    For a horizontal crack, the tip with the smaller x is taken as the 'upper' one.

    Returns
    ----------
    center: np.array (xc, yc), the midpoint of the crack, in m.
    t: np.array (tx, ty), unit vector along the crack, pointing from the upper tip
       to the lower tip (down-dip).
    n: np.array (nx, ny), unit vector normal to the crack, equal to t rotated
       +90 degrees (counterclockwise): n = (-ty, tx). (t, n) is therefore a
       right-handed frame, the same handedness as (X, Y). Because t points
       down-dip, n always has nx >= 0: for a vertical crack, n = (1, 0).
    a: the half-length of the crack, in m.
    h: +1 or -1, the orientation factor of the shear driving stresses. It is +1
       when n points into the hanging wall (the block above the crack, ny > 0)
       or, for a vertical crack, into the block at x > 0; and -1 when n points
       into the footwall (cracks dipping towards -x). Multiplying dpy and dpz by
       h makes their sign convention independent of the dip direction
       (see displacements()).
    """
    up=np.asarray(upper_tip,dtype=float) ; lo=np.asarray(lower_tip,dtype=float)

    # The upper tip is the shallower one; for a horizontal crack, the one with smaller x
    if up[1]<lo[1] or (up[1]==lo[1] and up[0]>lo[0]):
        up, lo = lo, up

    length=np.hypot(lo[0]-up[0],lo[1]-up[1])
    if length==0:
        raise InputError('The two tips of the crack are the same point')

    center=(up+lo)/2
    t=(lo-up)/length                 # along the crack, down-dip
    n=np.array([-t[1],t[0]])         # t rotated +90 degrees

    # n points into the hanging wall when ny = tx > 0; for a vertical crack
    # (tx = 0) n = (1, 0) and the block at x > 0 is taken as the hanging wall
    h=-1.0 if t[0]<0 else 1.0

    return center, t, n, length/2, h

def local_coordinates(x,y,center,t,n):
    """
    Coordinates of points in the local frame of the crack.

    Parameters
    ----------
    x, y: floats or arrays of the same shape, the global coordinates of the points, in m.
    center, t, n: the center and the unit vectors of the crack, from crack_frame().

    Returns
    ----------
    xi: the coordinate along the crack (along t), measured from its center, in m.
        The upper tip is at xi = -a and the lower tip at xi = +a.
    eta: the coordinate normal to the crack (along n), in m. eta > 0 on the side
        towards which n points, eta < 0 on the other side.

    Points exactly on the crack plane (eta = 0, including -0.0) are assigned
    eta = +0.0, so that points on the crack take the values of its n-side wall.
    """
    dx=np.asarray(x,dtype=float)-center[0]
    dy=np.asarray(y,dtype=float)-center[1]

    # Projections of the position vector onto t and n
    xi=dx*t[0]+dy*t[1]
    eta=dx*n[0]+dy*n[1]

    # -0.0 == 0 is True, so this also turns -0.0 into +0.0
    eta=np.where(eta==0,0.0,eta)

    return xi, eta

def tripolar_coordinates(xi,eta,a):
    """
    Signed tripolar coordinates of points around a crack of half-length a
    lying along the xi axis, from xi = -a to xi = +a.

    Parameters
    ----------
    xi, eta: local coordinates, from local_coordinates(), in m.
    a: the half-length of the crack, in m.

    Returns
    ----------
    r, theta: distance (m) and angle (rad) from the center of the crack.
    r1, theta1: distance and angle from the lower tip (xi = +a).
    r2, theta2: distance and angle from the upper tip (xi = -a).
    R, THETA: R = (r1*r2)**0.5 and THETA = (theta1 + theta2)/2.

    All angles are measured from the +xi direction (along t, down-dip) and are
    positive towards +eta (towards n), with np.arctan2, so they range from -pi to pi.
    The sign is what tells the two sides of the crack apart: theta, theta1 and
    theta2 are > 0 on the n side and < 0 on the other side. With these angles
    the complex quantities of the solution take the principal branch, e.g.
    (z**2 - a**2)**0.5 = R*exp(i*THETA) and z/(z**2 - a**2)**0.5 = (r/R)*exp(i*(theta-THETA)),
    with z = xi + i*eta, which is continuous everywhere except across the crack.
    """
    r=np.hypot(xi,eta) ; theta=np.arctan2(eta,xi)
    r1=np.hypot(xi-a,eta) ; theta1=np.arctan2(eta,xi-a)
    r2=np.hypot(xi+a,eta) ; theta2=np.arctan2(eta,xi+a)

    R=(r1*r2)**0.5
    THETA=(theta1+theta2)/2

    return r, theta, r1, theta1, r2, theta2, R, THETA

def rotate_vector_to_global(v_t,v_n,t,n):
    """
    Rotates a vector from the local frame of the crack to the global X-Y axes.

    v_t, v_n: the components of the vector along t and along n.
    t, n: the unit vectors of the crack, from crack_frame().

    Returns vx, vy: v = v_t*t + v_n*n.
    """
    return v_t*t[0]+v_n*n[0], v_t*t[1]+v_n*n[1]

def rotate_stress_to_global(s_tt,s_nn,s_tn,t,n):
    """
    Rotates an in-plane stress tensor from the local frame of the crack
    to the global X-Y axes.

    s_tt: normal stress on planes normal to t (acting along the crack),
    s_nn: normal stress on planes normal to n (acting across the crack),
    s_tn: shear stress on those planes.
    t, n: the unit vectors of the crack, from crack_frame().

    Returns sxx, syy, sxy, from S = s_tt*(t t) + s_nn*(n n) + s_tn*(t n + n t),
    where (t n) is the outer product of t and n.
    """
    sxx=s_tt*t[0]**2+s_nn*n[0]**2+2*s_tn*t[0]*n[0]
    syy=s_tt*t[1]**2+s_nn*n[1]**2+2*s_tn*t[1]*n[1]
    sxy=s_tt*t[0]*t[1]+s_nn*n[0]*n[1]+s_tn*(t[0]*n[1]+n[0]*t[1])

    return sxx, syy, sxy

def check_crack_geometry(upper_tip,crack_center,lower_tip,center,a,a_input=None):
    """
    Checks that the crack_center (and a, if given) passed to displacements()
    or stresses() are consistent with the tips, since the solution requires
    the center to be the midpoint of the crack and a its half-length.
    """
    tolerance=1e-6*max(a,1)

    if np.hypot(crack_center[0]-center[0],crack_center[1]-center[1])>tolerance:
        raise InputError('crack_center '+str(tuple(crack_center))+' is not the midpoint of the tips '
                         +str(tuple(center)))

    if a_input is not None and abs(a_input-a)>tolerance:
        raise InputError('a = '+str(a_input)+' m is not the half-length of the crack, '+str(a)+' m')

########################### Displacement field equations ###########################

def displacements(dpx,dpy,dpz,nu,E,upper_tip,crack_center,lower_tip,point):
    """
    Displacements around a straight 2D crack loaded by uniform driving stresses
    on its walls, in plane strain (Pollard and Segall, 1987; Pollard and Martel, 2020).

    Parameters
    ---------------
    dpx: driving stress normal to the crack walls, in Pa. > 0 opens the crack
         (mode I). For a vertical crack it acts along x.
    dpy: driving stress parallel to the crack walls, in the X-Y plane, in Pa
         (mode II). > 0 makes the hanging wall slide down-dip relative to the
         footwall (normal sense) and < 0 up-dip (reverse sense). For a vertical
         crack, > 0 makes the block at x > 0 move down relative to the block at x < 0.
    dpz: driving stress parallel to the crack walls, out of the X-Y plane, in Pa
         (mode III). > 0 makes the hanging wall (the block at x > 0 for a
         vertical crack) move towards +z relative to the footwall.
    nu: Poissons ratio (unitless),
    E: Youngs Modulus in Pa,
    upper_tip, crack_center, lower_tip: the (x, y) coordinates of the upper tip,
    the center and the lower tip of the crack, in m. crack_center must be the
    midpoint of the tips. The crack may have any orientation.
    point: the coordinates (x, y) of the point(s) of interest, in m. x and y may be
    floats or arrays of the same shape (e.g. the whole grid at once).

    Returns
    ---------------
    ux, uy, uz: the displacements at the point(s) of interest along X, Y, and Z, in m.

    The solution is evaluated in the local frame of the crack (t along the crack,
    n normal to it, see crack_frame) and rotated to the global axes:
        u_n: displacement normal to the crack,
        u_t: displacement parallel to the crack,
        (ux, uy) = u_t*t + u_n*n.
    Points exactly on the crack take the displacement of its n-side wall.
    The solution is singular at the tips.
    """
    center, t, n, a, h = crack_frame(upper_tip,lower_tip)
    check_crack_geometry(upper_tip,crack_center,lower_tip,center,a)

    xi, eta = local_coordinates(point[0],point[1],center,t,n)
    r, theta, r1, theta1, r2, theta2, R, THETA = tripolar_coordinates(xi,eta,a)

    # Shear driving stresses with a sign convention independent of the dip direction
    s=h*dpy ; s3=h*dpz

    shear=G(E,nu)

    # Terms shared by the equations, with z = xi + i*eta:
    # Im[(z**2-a**2)**0.5 - z] and Re[(z**2-a**2)**0.5 - z]
    im_Z=R*np.sin(THETA)-r*np.sin(theta)
    re_Z=R*np.cos(THETA)-r*np.cos(theta)
    # eta*Re[z/(z**2-a**2)**0.5 - 1] and eta*Im[z/(z**2-a**2)**0.5]
    eta_re=r*np.sin(theta)*((r/R)*np.cos(theta-THETA)-1)
    eta_im=r*np.sin(theta)*((r/R)*np.sin(theta-THETA))

    # Displacement normal to the crack (along n)
    u_n=(dpx*(2*(1-nu)*im_Z-eta_re)-s*((1-2*nu)*re_Z+eta_im))/(2*shear)

    # Displacement parallel to the crack (along t)
    u_t=(s*(2*(1-nu)*im_Z+eta_re)+dpx*((1-2*nu)*re_Z-eta_im))/(2*shear)

    # Out-of-plane displacement (Pollard and Segall, 1987)
    uz=(2*s3*im_Z)/(2*shear)

    # Rotation to the global axes
    ux, uy = rotate_vector_to_global(u_t,u_n,t,n)

    return ux,uy,uz

########################### Traction boundary value problem, stress field equations ###########################

# Relative distance to a crack tip (in units of the half-length a) below which a point
# is treated as lying on the tip, where the stress solution is singular
TIP_TOLERANCE=1e-6

def stresses(dpx,dpy,dpz,rxx,ryy,rxy,rxz,ryz,nu,E,upper_tip,crack_center,lower_tip,a,point):
    """
    Stresses around a straight 2D crack loaded by uniform driving stresses on its
    walls, in plane strain (Pollard and Segall, 1987; Pollard and Martel, 2020),
    plus uniform remote stresses. Tension positive.

    Parameters
    ---------------
    dpx: driving stress normal to the crack walls (mode I), in Pa,
    dpy: driving stress parallel to the crack walls in the X-Y plane (mode II), in Pa,
    dpz: driving stress parallel to the crack walls out of the X-Y plane (mode III), in Pa,
    with the same sign conventions as in displacements().
    rxx: Remote stress in Pa, normal to the X plane,
    ryy: Remote stress in Pa, normal to the Y plane,
    rxy: Shear stress on plane X in the direction of the Y axis,
    rxz: Shear stress on plane X in the direction of the Z axis,
    ryz: Shear stress on plane Y in the direction of the Z axis,
    all in Pa, in the global X-Y axes. They may be floats or arrays with the shape of x and y.

    nu: Poissons ratio (unitless),
    E: Youngs Modulus in Pa (not used by the stress solution, kept for symmetry
    with displacements()),
    upper_tip, crack_center, lower_tip: the (x, y) coordinates of the crack's
    upper tip, center and lower tip, in m. The crack may have any orientation.
    a: the crack's half-length in m. It must match the distance between the tips.
    point: the coordinates (x, y) of the point(s) of interest, in m. x and y may be
    floats or arrays of the same shape.

    Return
    ---------------
    The stresses at the point(s) of interest, in the global axes,
    sxx, syy, sxy (=syx), sxz, syz, szz and t (the maximum in-plane shear stress),
    in this order, all in Pa.

    The crack-induced stresses are evaluated in the local frame of the crack
    (s_nn across it, s_tt along it, s_tn shear, see crack_frame) and the
    tensor is rotated to the global axes before adding the remote stresses.
    Points exactly on the crack take the values of its n-side wall.
    The solution is singular at the tips: points closer than TIP_TOLERANCE*a to
    a tip (in practice, grid nodes that coincide with a tip) return NaN, and
    evaluate_stress_state classifies them as 'Singular'.
    """
    center, t, n, a_tips, h = crack_frame(upper_tip,lower_tip)
    check_crack_geometry(upper_tip,crack_center,lower_tip,center,a_tips,a)

    xi, eta = local_coordinates(point[0],point[1],center,t,n)
    r, theta, r1, theta1, r2, theta2, R, THETA = tripolar_coordinates(xi,eta,a)

    # Shear driving stresses with a sign convention independent of the dip direction
    s=h*dpy ; s3=h*dpz

    # Terms shared by the equations, with z = xi + i*eta:
    # C = Re[z/(z**2-a**2)**0.5] - 1 and S = Im[z/(z**2-a**2)**0.5]
    C=(r/R)*np.cos(theta-THETA)-1
    S=(r/R)*np.sin(theta-THETA)
    # q = a**2*eta/R**3
    q=(a**2)*(r/(R**3))*np.sin(theta)

    # Normal stress across the crack (on planes parallel to it)
    s_nn=dpx*(C+q*np.sin(3*THETA))+s*(q*np.cos(3*THETA))

    # Normal stress along the crack (on planes normal to it)
    s_tt=dpx*(C-q*np.sin(3*THETA))+s*(2*S-q*np.cos(3*THETA))

    # In-plane shear stress
    s_tn=s*(C-q*np.sin(3*THETA))+dpx*(q*np.cos(3*THETA))

    # Out-of-plane shear stresses on planes parallel (s_nz) and normal (s_tz) to the crack
    s_nz=s3*C
    s_tz=s3*S

    # Rotation to the global axes and remote stresses
    sxx, syy, sxy = rotate_stress_to_global(s_tt,s_nn,s_tn,t,n)
    sxz, syz = rotate_vector_to_global(s_tz,s_nz,t,n)

    sxx=rxx+sxx ; syy=ryy+syy ; sxy=rxy+sxy
    sxz=rxz+sxz ; syz=ryz+syz

    # Calculating sigma_zz, plane strain
    szz=nu*(sxx+syy)

    # Calculating t, the maximum in-plane shear stress
    t_max=0.5*(((syy-sxx)**2)+4*(sxy**2))**0.5
    
    # Singular points: on a tip, rounding leaves r1 or r2 ~1e-13 m and stresses of
    # ~1e14 Pa instead of NaN. They are set to NaN explicitly (since Nov 2026).
    singular=np.minimum(r1,r2)<TIP_TOLERANCE*a
    if np.any(singular):
        sxx, syy, sxy, sxz, syz, szz, t_max = [np.where(singular,np.nan,v) for v in
                                               np.broadcast_arrays(sxx,syy,sxy,sxz,syz,szz,t_max)]
        if np.ndim(singular)==0:
            sxx, syy, sxy, sxz, syz, szz, t_max = [float(v) for v in (sxx,syy,sxy,sxz,syz,szz,t_max)]

    return sxx,syy,sxy,sxz,syz,szz,t_max

########################### Model class ###########################

########################### Principal stresses and stresses on the optimal planes ###########################

def principal_from_stresses(sxx,syy,sxy):
    """
    Principal stresses and their orientations from the in-plane stress
    components (Phase 1, Nov 2026: extracted from model.principal_stresses so
    that they can be computed from stresses that are already available).
    
    Parameters
    ----------
    sxx, syy, sxy: floats or arrays, in Pa, TENSION positive (as returned by
    model.model_stresses and stresses()).
    
    Returns
    -------
    s1, s3: maximum and minimum compressive principal stresses, in Pa,
    COMPRESSION positive.
    s1a, s3a: orientations of sigma1 and sigma3, in degrees counterclockwise
    from +x, in (-90, 90]. See docs/crack_coordinates_and_rotation.md, Section 10.
    """
    # Compression positive
    sxx=-np.asarray(sxx) ; syy=-np.asarray(syy) ; sxy=-np.asarray(sxy)
                            
    s1=(0.5*(sxx+syy)+(((sxx-syy)*0.5)**2+sxy**2)**0.5)
    s3=(0.5*(sxx+syy)-(((sxx-syy)*0.5)**2+sxy**2)**0.5)
    
    # Orientation of sigma1 (Jaeger et al., 2007): tan(2a) = 2sxy/(sxx-syy).
    # arctan2 picks the quadrant of 2a from the signs of sxy and sxx-syy,
    # so a is the direction of the MAXIMUM compressive stress, in (-90, 90],
    # with no division by zero. For example, with sxy = 0 and sxx < syy
    # (vertical sigma1, as in the lithostatic state) it gives 90.
    #
    # Until Nov 2026 the angle was arctan(2sxy/((sxx-syy)+R)), where the
    # denominator was replaced by 0.001 when it was 0: that returned 0
    # (horizontal) instead of 90 wherever sigma1 was vertical and sxy = 0.
    # The whole grid was also forced to 90 when dpx = 0 and s11 = 0, which
    # ignored faults loaded only in shear.
    s1a=np.degrees(0.5*np.arctan2(2*sxy,sxx-syy))
    s1a=np.where(s1a<=-90,s1a+180,s1a)
    
    # sigma3 is normal to sigma1, also in (-90, 90]
    s3a=np.where(s1a>0,s1a-90,s1a+90)
    
    return s1, s3, s1a, s3a

def optimal_plane_stresses(s1,s3,coula):
    """
    Normal and shear stresses on the optimal shear planes, which lie at coula
    (radians) from sigma1 (Pollard and Martel, 2020). s1 and s3 compression
    positive, in Pa. Returns sn, ss, in Pa, compression positive.
    """
    sn=0.5*(s1+s3)+0.5*(s1-s3)*np.cos(2*coula)
    ss=0.5*(s1-s3)*np.sin(2*coula)
    
    return sn, ss

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
        Return the driving stress normal to the crack walls (mode I, > 0 opens it).
        For a vertical crack it acts along x.
        """
        return self.dpx
    
    def y(self):
        """
        Return the driving stress parallel to the crack walls in the X-Y plane
        (mode II, > 0 normal sense: the hanging wall slides down-dip).
        For a vertical crack it acts along y.
        """
        return self.dpy
    
    def z(self):
        """
        Return the driving stress parallel to the crack walls out of the X-Y plane
        (mode III).
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
    def __init__(self, x_limits,y_limits,x_cellsize,y_cellsize,y_position_layers,layers_E,layers_v,layers_d,layers_cfi,layers_T, g):
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
        
        layers_cfi: an array-like object with floats/ints defining the coefficient of 
        internal friction of each of the layers, unitless
        
        layers_T: an array-like object with floats/ints defining the tensile
        strength of each of the layers, in Pa. Negative, because tension is
        negative on the Mohr diagram (e.g. -1.5e6).
        
        g: gravitational acceleration in ms-2 for this model.
        
        The cohesion of each layer is not an input: it is implicit in the
        modified Griffith criterion, C = 2|T| (Rivas-Dorado et al., 2023,
        eq. 12b), and is stored in layers_c for the grid and the labels.
        """
        
        self.x_limits=x_limits ; self.y_limits=y_limits
        self.x_cellsize=x_cellsize ; self.y_cellsize=y_cellsize
        self.y_position_layers=y_position_layers
        self.layers_E=layers_E ; self.layers_v=layers_v ; self.layers_d=layers_d
        self.layers_cfi=layers_cfi; self.layers_T=layers_T
        
        # Cohesion implicit in the modified Griffith criterion: C = 2|T|
        self.layers_c=2*np.abs(np.asarray(self.layers_T,dtype=float))
        
        # The coulomb angle is defined implicitly after mu for each layer
        self.layers_coula=0.5*np.arctan(1/np.asarray(self.layers_cfi,dtype=float))
    
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
                    (self.layers_cfi,'layers_cfi'),(self.layers_coula,'layers_coula'),
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
    
    def metadata(self, crack=None, rxx=0, ryy=0, litho=None):
        """
        Description of the model (and optionally of the crack and the loading)
        as a JSON-serialisable dict, stored with the results.
        """
        def as_list(values):
            return np.atleast_1d(np.asarray(values,dtype=float)).tolist()
        
        info={'x_limits':as_list(self.x_limits), 'y_limits':as_list(self.y_limits),
              'x_cellsize':float(self.x_cellsize), 'y_cellsize':float(self.y_cellsize),
              'n_layers':self.n_layers(), 'y_position_layers':as_list(self.y_position_layers),
              'layers_E':as_list(self.layers_E), 'layers_v':as_list(self.layers_v),
              'layers_d':as_list(self.layers_d), 'layers_c':as_list(self.layers_c),
              'layers_cfi':as_list(self.layers_cfi), 'layers_coula':as_list(self.layers_coula),
              'layers_T':as_list(self.layers_T), 'g':float(self.g)}
        if crack is not None:
            info['crack']={'upper_tip':as_list(crack.up_tip()), 'lower_tip':as_list(crack.low_tip()),
                           'center':as_list(crack.c()), 'length':float(crack.extent()),
                           'dpx':float(crack.x()), 'dpy':float(crack.y()), 'dpz':float(crack.z())}
            info['remote']={'rxx':as_list(rxx), 'ryy':as_list(ryy)}
            info['litho']=litho
        return info
    
    def compute(self, crack, rxx=0, ryy=0, litho=True, displacements=True, stresses=True):
        """
        Computes every field of the model for a crack, once (Phase 1, Nov 2026).
        
        Parameters
        ----------
        crack: a crack object,
        rxx, ryy: remote stresses normal to the x and y planes, in Pa (floats,
        or arrays with the shape of the grid). Both may be 0.
        litho: a bool. If True, the lithostatic stresses are included.
        displacements, stresses: bools, to compute only part of the fields
        (e.g. displacements=True, stresses=False for a displacement figure).
        
        Returns
        -------
        A DiESResults object (results.py) with the material at each node
        (E, nu, density, cohesion, cfi, coula, T) and, as requested,
        ux, uy, uz; sxx, syy, sav, sxy, sxz, szz, tmax (tension positive);
        s1, s3, s1a, s3a and sn, ss (compression positive).
        The values are the same as those of model_displacements,
        model_stresses, principal_stresses and plane_stresses.
        """
        litho=require_bool(litho,'litho')
        grid=self.grid()
        fields={'E':grid[2], 'nu':grid[3], 'density':grid[4], 'cohesion':grid[5],
                'cfi':grid[6], 'coula':grid[7], 'T':grid[8]}
        
        if displacements:
            fields['ux'], fields['uy'], fields['uz'] = self.model_displacements(crack)
        
        if stresses:
            sxx, syy, sav, sxy, sxz, szz, tmax = self.model_stresses(crack, rxx, ryy, litho)
            fields.update({'sxx':sxx, 'syy':syy, 'sav':sav, 'sxy':sxy, 'sxz':sxz, 'szz':szz, 'tmax':tmax})
            fields['s1'], fields['s3'], fields['s1a'], fields['s3a'] = principal_from_stresses(sxx,syy,sxy)
            fields['sn'], fields['ss'] = optimal_plane_stresses(fields['s1'],fields['s3'],grid[7])
        
        return DiESResults(grid[0], grid[1], fields, self.metadata(crack, rxx, ryy, litho))
    
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
                    +'\n'+r"$\mu'$ = "+str(self.layer_values(self.layers_cfi,'layers_cfi')[0]),
                    bbox=dict(fc='white', ec='black'),
                    horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes,fontsize=14, zorder=5)
        
        else:
            i=0
            for y,E,v,d,c,m in zip(self.y_position_layers[1:],self.layers_E, self.layers_v, self.layers_d, self.layers_c, self.layers_cfi):
                x=self.x_limits[-1]-0.05*points_distance((self.x_limits[0],0),(self.x_limits[-1],0))
                y=y-0.05*points_distance((self.x_limits[-1],self.y_position_layers[i]),(self.x_limits[-1],y))
                i=i+1
                ax.text(x,y,
                        'E = '+str(E/1E9)+' GPa\n'+'v = '+str(v)+'\n'
                        +r'$\rho$ = '+str(d)+r' kg/m$^3$'
                        +'\n'+r'C$_o$ = '+str(c/1e6)+' MPa'
                        +'\n'+r"$\mu'$ = "+str(m),
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
            # Magnitude of the in-plane displacement vector
            dp=dx ; ds=dy ; dt=np.hypot(dx,dy) ; plot_params=boundaries_contours(dt, step) ; label=r'|u$_{xy}$| (m)'
        
        else:
            raise InputError('Invalid displacement. Input "ux", "uy", or "uxy" instead.')
            
        if option=='vectors':
            # The arrows are scaled with the largest magnitude of the plotted vector
            # (|ux|, |uy| or |u|, depending on the selected displacement)
            plot=ax.quiver(x,y,dp,ds,dt, width=0.0050, scale=np.nanmax(np.hypot(dp,ds))/0.1, cmap='viridis')
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
            raise InputError('Invalid option. Input "vectors" or "contours" instead.')
                
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
        s1a: orientation of the sigma1 direction, in degrees, measured
        counterclockwise from the +x axis, in (-90, 90]. 90 is vertical.
        s3a: orientation of the sigma3 direction, in degrees, measured in the
        same way: s1a +/- 90, in (-90, 90].
        A direction at angle a is the unit vector (cos a, sin a); a and a+180
        are the same line, so the orientations are given modulo 180.
        
        See docs/crack_coordinates_and_rotation.md, Section 10.
        """
        
        s=self.model_stresses(crack, s11, s22, boolean)
        
        # sxx, syy and sxy, tension positive (sxy is the 4th output of model_stresses)
        return principal_from_stresses(s[0],s[1],s[3])
    
    def plot_principal_stresses(self, crack, stress, s11, s22, boolean, step):
        """

        Parameters
        ----------
        crack: a crack object,
        stress: str, specifies the stress to be plotted, 
            'principal' for the maximum compressive stress, 
            'secondary', for the minimum compressive stress,
            'angle', for the orientation of sigma1, in degrees counterclockwise
            from the +x axis, in (-90, 90] (90 = vertical, see principal_stresses),
        s11 : array-like, the remote stress acting along the x axis, in Pa,
        s11 : array-like, the remote stress acting along the y axis, in Pa,
        boolean: a bool. True, to indicate that the lithostatic stress must
        be calculated, False, for the opposite. Not a str.
        step: int, the desired contour interval in MPa

        Returns
        -------
        A figure object plotting the desired principal stress in MPa, or the
        angle of sigma1 in degrees

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
            label=r'$\sigma_1$ angle from +x (°)'
        
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
        planes: a bool. If True, plots the optimal shear planes defined by cfi.
        If False, it does not do this. It must be a real boolean, not a str.
        s11 : array-like, the remote stress acting along the x axis, in Pa,
        s11 : array-like, the remote stress acting along the y axis, in Pa,
        boolean: a bool. True, to indicate that the lithostatic stress must
        be calculated, False, for the opposite. Not a str.

        Returns
        -------
        A figure plotting the trajectories of the principal stresses and
        optionally the orientation of the optimal shear planes defined by cfi.
        
        """

        # Failing early, before the stresses are calculated, if the input is invalid
        planes=require_bool(planes,'planes')
        
        grid=self.grid()
        x, y = grid[0], grid[1]
        s, cfi, coula = grid[5], grid[6], grid[7]
        
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
                
        # Orientations are counterclockwise from +x: a direction at angle a is (cos a, sin a).
        # Until Nov 2026 they were drawn as (cos a, -sin a), i.e. mirrored about the horizontal.
        plot1=ax.quiver(x,y,np.cos(p_angles),np.sin(p_angles),p,headlength=0, pivot='middle', scale=0.0090, units='xy', width=10, headwidth=1, cmap='copper_r')
        plot2=ax.quiver(x,y,np.cos(s_angles),np.sin(s_angles),s,headlength=0, pivot='middle', scale=0.0090, units='xy', width=10, headwidth=1, cmap='cool')
        
        # plot1=ax.quiver(x,y,sins,cosins, p, headlength=0, pivot='middle', scale=0.0030, units='xy', width=30, headwidth=1, cmap='copper_r')
        # plot2=ax.quiver(x,y,sins_ortho,cosins_ortho,s,headlength=0, pivot='middle', scale=0.0045, units='xy', width=30, headwidth=1, cmap='cool')
        
        cbar=figure.colorbar(plot1,extend='both',shrink=0.60, aspect=30, pad=0.05)
        cbar.set_label(r'$\sigma_1$ (MPa)', rotation=270, labelpad=12, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        cbar=figure.colorbar(plot2,extend='both',shrink=0.60, aspect=30)
        cbar.set_label(r'$\sigma_3$ (MPa)', rotation=270, labelpad=16, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        if planes:
            # The two conjugate optimal shear planes lie at +/- coula from sigma1,
            # coula = 0.5*arctan(1/mu) being the angle between sigma1 and the planes.
            # beta and beta_p are their orientations, counterclockwise from +x
            beta=p_angles-coula
            beta_p=2*p_angles-beta
            
            # '#FF1493', #3FFF00
            ax.quiver(x,y,np.cos(beta),np.sin(beta),headlength=0, pivot='middle', scale=0.0090, units='xy', width=10, headwidth=1, color='#4DAD4B',alpha=0.8)
            ax.quiver(x,y,np.cos(beta_p),np.sin(beta_p),headlength=0, pivot='middle', scale=0.0090, units='xy', width=10, headwidth=1, color='#FA422B',alpha=0.8)
            
        # ax.legend(loc=1)
        
        # ax.text(0.96,0.14,r'$\sigma_I$ = '+str(round(dpx/1E6,2))+' MPa; '+r'$\sigma{^r_x}$ = '+str(round(s11/1e6,0))+' MPa\n'
        #         +r'$\alpha$ = '+str(angle)+'°; '+r'$\rho$ = '+str(self.layers_d)+' kg m$^{-3}$\n'
        #         +r"$\mu'$ = "+str(self.layers_cfi)+'; '+r'$\theta_{opt}$ = '+str(np.round(90-np.degrees(self.layers_coula),2))+'°',
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
        s, cfi, coula = grid[5], grid[6], grid[7]
        
        
        ########### With the principal stresses ###########
        s=self.principal_stresses(crack, s11, s22, boolean)
        s1=s[0] ; s3=s[1]
        
        # Pollard and Martel, 2020
        return optimal_plane_stresses(s1,s3,coula)
    
    
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
        s, cfi, coula = grid[5], grid[6], grid[7]
        
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
                    +'\nC = '+str(self.layer_values(self.layers_c,'layers_c')[0]/1e6)+' MPa ; '+r"$\mu'$ = "+str(self.layer_values(self.layers_cfi,'layers_cfi')[0])+'; '
                    +r'$\theta_{opt}$ = '+str(round(90-np.degrees(self.layer_values(self.layers_coula,'layers_coula')[0]),2))+'°',
                    bbox=dict(fc='white', ec='black'),
                    horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes,fontsize=11, zorder=5)
        else:
            ax.text(0.98,0.14,r'$\sigma_I$ = '+str(round(dpx/1E6,2))+' MPa\n'+r'$\sigma_{II}$ = '+str(round(dpy/1E6,2))+' MPa\n'
                    +r'$\alpha$ = '+str(angle)+'°'
                    +'\nC = '+str(np.asarray(self.layers_c,dtype=float)/1e6)+' MPa ; '+r"$\mu'$ = "+str(np.asarray(self.layers_cfi,dtype=float))+'; '
                    +r'$\theta_{opt}$ = '+str(np.round(90-np.degrees(self.layers_coula),2))+'°',
                    bbox=dict(fc='white', ec='black'),
                    horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes,fontsize=11, zorder=5)
                
        return figure