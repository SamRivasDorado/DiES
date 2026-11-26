import sys
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import (MultipleLocator)

########################### Geomechanical parameters ###########################

def G(E,v):
    """
    Takes E, Youngs Modulus in Pa, and
    v, Poissons ratio (unitless).
    
    Returns shear modulus, G, or rigidity, in Pa
    """
    
    return (E/(2*(1+v)))

def S(E,v):
    """
    Takes E, Youngs Modulus in Pa, and
    v, Poissons ratio (unitless).
    
    Returns stiffness, S, in Pa
    """
    
    return (E/(2*(1-v**2)))

def K(E,v):
    """
    Takes E, Youngs Modulus in Pa, and
    v, Poissons ratio (unitless).
    
    Returns incompressibility, K, or bulk modulus, in Pa
    """
    
    return (E/(3*(1-2*v)))

########################### Geometry functions ###########################
    
def points_distance(point1,point2):
    """
    Takes the coordinates of two points,1 and 2,
    which are in the form of a tuple/list/array,
    
    Returns the distance between them in the same
    unit input.
    """
    
    d=((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)**0.5
    
    return d

def slope(point1,point2):
    """
    Takes the coordinates of two points,1 and 2,
    which are in the form of a tuple/list/array.
    
    Returns the slope of the line between the two.
    """
    
    upper_term=point2[1]-point1[1]
    lower_term=point2[0]-point1[0]
    
    return upper_term/lower_term

def angle_between_lines(point1,point2,point3,point4):
    """
    Takes the coordinates of four points defining two lines
    in the form of a tuple/list/array.
    point1 and point2 define the first line, a,
    point3 and point3 define the second line,a
    The dot product is used to calculate the angle between lines.
    
    Returns the angle between lines in degrees.
    """
    
    dxa = point2[0]-point1[0]
    dya = point2[1]-point1[1]
    dxb = point4[0]-point3[0]
    dyb = point4[1]-point3[1]
    
    d = dxa*dxb + dya*dyb                                      #Dot product of the 2 vectors
    l = (((dxa)**2+(dya)**2)**0.5)*(((dxb)**2+(dyb)**2))**0.5 #Product of the squared lengths
    
    # Making sure that l is never 0
    if l==0:
        l=0.01
    
    # Making sure that only valid values for arcos are passed
    value=d/l
    if value<-1:
        value=-1
    elif value>1:
        value=1
    
    angle = np.arccos(value)
        
    return np.degrees(angle)

def circle_eq_y(h,r,x):
    """

    Parameters
    ----------
    h : x coordinate of the center of a circle
    x : an x coordinate to calculate the corresponding y
    r : circle radius
    All in the same unit

    Returns
    -------
    The corresponding y coordinates of a circle where  k = 0

    """

    y=(abs(r**2-(x-h)**2))**0.5
    
    return y, -y


def linear(x,m,n):
    """
    Parameters
    ------------------
    x: an array-like object, float, or int,'
    m: slope of the linear function
    n: intercept of the linear function
    
    Returns
    ------------------
    y: an array-like object of the y values as evaluated by the function
    """
    
    # Lists and tuples are evaluated element by element and returned as an array;
    # ints, floats (including numpy scalars) and numpy arrays directly
    if isinstance(x,(list,tuple)):
        return np.asarray([value*m+n for value in x])
    
    return x*m+n
    
def power_law(x,C,m):
    'Takes in C, the constant in a power law function'
    'm, the exponential factor in a power law function'
    'a list or tuple of X values, or a single int/float X value'
    'or a numpy array'
    
    # Lists and tuples are evaluated element by element and returned as an array;
    # ints, floats (including numpy scalars) and numpy arrays directly.
    # Until Nov 2026 the type checks were exact (type(x)==float), so numpy
    # scalars such as np.float64 silently returned an empty list.
    if isinstance(x,(list,tuple)):
        return np.asarray([C*(value**m) for value in x])
    
    y=C*(x**m)
    
    'Returns either a list of y values corresponding to the x values, if the input is a list or tuple'
    'or a single number if the input is an int or a float'
    'in a y=C*(X^m) equation'    
    
    return y

def shoelace (points):
    """"
    Parameters
    ----------
    points: array-like object formed by tuples/lists of [x,y] pairs of values 
    that define a polygon, ordered clockwise. x and y should be in the same
    units.
    
    Returns
    ----------
    The area of the polygon, in the square units of x and y
    """
    
    term1=0 ; term3=0
    
    for i in range (len(points)-1):
        term1=term1+points[i][0]*points[i+1][1]
        term3=term3+points[i+1][0]*points[i][1]
            
    term2=points[len(points)-1][0]*points[0][1]
    term4=points[0][0]*points[len(points)-1][1]  
    
    return (0.5*abs(term1+term2-term3-term4))

########################### Generic functions ###########################

def true_min_max(array):
    """
    Parameters
    ----------
    array : a 2D array 

    Returns
    -------
    Floats, the true minimum and maximum values in the array, after 
    ignoring invalid values
    """
    # Copy of each row without its NaN values
    new_array=[]
    
    for row in array:
        new_row=[]
        for value in row:
            if str(value)!='nan':
                new_row.append(value)
        new_array.append(new_row)

    # If every value is NaN there is nothing to compare: (0, 1) is returned,
    # so that a colour scale can still be built
    try:
        minimum, maximum = min(np.concatenate(new_array)), max(np.concatenate(new_array)) 
    
    except:
        minimum, maximum = 0, 1
    
    return minimum, maximum


def downsample_2darray(array,rate):
    """
    Parameters
    ----------
    array : a 2D array 
    rate : an int, used for the downsampling of the rows and columns of the array. 
    For example, a rate = 2 will show 1 out of every 2 values in each
    row and column, rate = 5 will show 1 out of every 5 values in each
    row and column

    Returns
    -------
    array: a downsampled array with the eliminated values as nans
    """
    resampled_array=[]
    
    # i and j count rows and columns from 1; a value is kept only where both
    # are multiples of rate, and the rest become NaN (not drawn by quiver)
    i=1
    for row in array:
        j=1
        new_row=[]
        
        if i%rate==0:
            for value in row:
                if j%rate==0:
                    new_row.append(value)
                elif j%rate!=0:
                    new_row.append(float('NaN'))
                j=j+1
            
            
        elif i%rate!=0:
            for value in row:
                new_row.append(float('NaN'))
                j=j+1
            
        resampled_array.append(new_row)
        i=i+1
        
    resampled_array=np.asarray(resampled_array)
    
    return resampled_array
                  
########################### Plotting functions ###########################

def create_figure(x_limits,y_limits):
    """
    Empty figure with the model limits, axis labels in m and ticks at round
    intervals (model_locators), used by the dynamic model figures.
    """
    
    figure=plt.figure(figsize=(6,4),tight_layout=True)
    ax=figure.add_subplot(111, xlim=x_limits, ylim=y_limits)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.tick_params(labelsize=12, axis='both')
    
    # Major ticks that do not overlap, whatever the size of the model
    model_locators(ax,x_limits,y_limits)
    
    return figure

def scale_arrows(dpx,dpy,dx,dy):
    """
    Parameters
    ----------
    dpx: driving pressure along x, 
    dpy: driving pressure along y, both in MPa, 
    dx: the displacements in the x direction,
    dy: the displacements in the y direction, both as two 2D arrays,
    the outputs of the 'displacements' function.
    
    
    Returns
    -------
    The two scaling parameters needed for an appropriate
    visualization using the quiver method in matplotlib, i.e., 
    the arguments for the 'scale' arg in the 'quiver' method.
    """
    
    dpx=float(dpx) ; dpy=float(dpy)
    
    if not (np.isfinite(dpx) and np.isfinite(dpy)):
        raise ValueError('scale_arrows needs finite driving pressures, got dpx='+str(dpx)+', dpy='+str(dpy))
    
    if dpx>dpy:
        maximum=np.amax(abs(dx))
        denom=0.05
    
    elif dpx<dpy:
        if dpy<=0:
            # Both compressive
            maximum=np.amax(abs(dy))
            denom=0.025
        elif dpx>0:
            # Both tensile
            maximum=np.amax(abs(dy))
            denom=0.05
        else:
            # dpx<=0<dpy
            maximum=np.amax(abs(dx))
            denom=0.05
    
    else:
        maximum=np.amax(abs(dx))
        denom=0.04
            
    return maximum,denom

def boundaries_contours(values,step=None):
    """
    Takes a set of values, which can represent anything, but should usually
    be 2D arrays representing the values of a 2D mesh, and 
    step: a float representing the desired contour interval, or None to
    choose it automatically (auto_contour_step).
    
    Returns the p1 and p99 of said set of values, and an array with the
    values at which to plot contours between them. This array should be used
    with the contours method within matplotlib.
    
    Since Nov 2026 (Phase 2, figures): NaN values (e.g. nodes on a crack tip)
    are ignored, and the contour levels are multiples of the step. Until then
    they started at p1 (steps < 1) or at p1 rounded down to a multiple of 5.
    """
    
    # Flattened, without NaN (nodes on the crack tips)
    values=np.asarray(values,dtype=float).ravel()
    values=values[np.isfinite(values)]
    if values.size==0:
        return 0.0, 1.0, np.array([])
        
    # The 1st and 99th percentiles are used as colour limits, so that the
    # extreme values next to the crack tips do not saturate the colour scale
    p99=np.percentile(values,99)
    p1=np.percentile(values,1)
    
    if step is None:
        step=auto_contour_step(values)
    if step is None:
        # Constant field: no contours
        return p1, p99, np.array([])
        
    # Contour levels at the multiples of step between the limits
    first=np.ceil(p1/step)*step
    new_contours=np.arange(first,p99+step*1e-9,step)
    
    return p1,p99,new_contours


def nice_step(span,max_count):
    """
    The smallest 'round' step (1, 2, 2.5 or 5 times a power of 10) that
    divides span into at most max_count intervals. None if span is not > 0.
    Used for contour intervals and axis ticks (Phase 2, Nov 2026).
    """
    if not np.isfinite(span) or span<=0:
        return None
    # Power of ten just below the step that would give exactly max_count intervals
    magnitude=10**np.floor(np.log10(span/max_count))
    for factor in (1,2,2.5,5,10):
        if span/(factor*magnitude)<=max_count:
            return float(factor*magnitude)
    return float(10*magnitude)


def auto_contour_step(values,target=12):
    """
    Automatic contour interval for a field (Phase 2, Nov 2026): a round step
    (1, 2, 2.5 or 5 times a power of 10) giving at most `target` contours
    between the 1st and 99th percentiles of the values. Returns None for a
    constant field (or one without finite values). In the dynamic models the
    values of the final step are used, so that the final figure is the one
    that looks right. It is a starting point: every figure accepts a step.
    """
    values=np.asarray(values,dtype=float).ravel()
    values=values[np.isfinite(values)]
    if values.size==0:
        return None
    span=np.percentile(values,99)-np.percentile(values,1)
    return nice_step(span,target)


def model_locators(ax,x_limits,y_limits):
    """
    Major ticks at round intervals, at most 6 labels along x and 13 along y
    (so that the labels of a 10 km wide model do not overlap), and automatic
    minor ticks. Until Nov 2026 the major ticks were every 1000 m (x) and
    500 m (y) whatever the size of the model.
    """
    from matplotlib.ticker import AutoMinorLocator
    x_step=nice_step(abs(x_limits[1]-x_limits[0]),6) or 1000
    y_step=nice_step(abs(y_limits[1]-y_limits[0]),13) or 500
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

########################### Geology functions ###########################
    

def profile_building(X,Y):
    """"
    Parameters
    ----------
    X: 1D array with the x coordinates of a set of points
    Y: 1D array with the y coordinates of a set of points
    both of the same length.
    
    Returns
    -------
    1D array of the same length where each element is the 
    distance between the first point and each point of the coordinate set
    """
    
    x1=X[0] ; y1=Y[0]
    
    profile_points=[]
    
    i=0
    for x in X:
        profile_points.append(((x-x1)**2+(Y[i]-y1)**2)**0.5)
        i=i+1
        
    profile_points=np.asarray(profile_points)
    
    return profile_points

def project_points(x, y, z, angle):
    """
    Parameters
    ----------
    x : 1D array-like object containing the x coordinates of a set of points 
    which form a line
    y : 1D array-like object containing the y coordinates of a set of points
    which form a line
    z: 1D array-like object containing the z (depth or elevation) coordinates 
    for each of the points
    angle: an int/float with the clockwise angle between the input line 
    and the line onto which the points are to be projected
    
    Returns
    -------
    Two array-like objects, 
    one with the distance from the first point along the line of projection, 
    another with the depth/elevation of each point.

    """
    
    distance = profile_building(x, y)
    
    projected_points=[]
    
    for point in distance:
        projected_points.append(np.cos(np.deg2rad(angle))*point)
        
    projected_points=np.asarray(projected_points)
    
    return projected_points, z

def normalize_lengths_frac(lengths):
    """
    Takes an 1D-like array set of distance measurements, 
    which sum gives the total length of a fracture. 
    
    Returns an array of the same measurements normalized 
    to the center of the fracture
    """
    
    if 'ndarray' not in str(type(lengths)):
        lengths=np.asarray(lengths)
        
    # Cumulative distance of each measurement and the midpoint of the fracture
    cumlengths=np.cumsum(lengths)
    center=np.average((0,cumlengths[-1]))
    
    # Distances relative to the midpoint: negative before it, positive after it
    normalized_lengths=[]
    
    for cl in cumlengths:
        if cl<center:
            normalized_lengths.append(-1*(center-cl))

        elif cl>center:
            normalized_lengths.append((cl-center))
    
    normalized_lengths=np.asarray(normalized_lengths)
    
    return normalized_lengths


def normalize_lengths_graben(distances,elevations):
    """
    Takes an 1D-like array set of distance measurements along a cross section,
    and a 1D-like array of the elevations for each point.
    
    Returns an array of the same measurements normalized 
    to the center of the graben, which is taken as the deepest 
    point in the cross section.
    """
    
    if 'ndarray' not in str(type(distances)):
        distances=np.asarray(distances)
            
    # The centre of the graben is the distance of the deepest point
    center_elev=min(elevations)
    for d,elev in zip(distances,elevations):
        if elev==center_elev:
            center_distance=d
            
    # Distances relative to that centre: negative on one side, positive on the other
    normalized_lengths=[]
    
    for distance in distances:
        if distance<center_distance:
            normalized_lengths.append(0-(center_distance-distance))

        elif distance>=center_distance:
            normalized_lengths.append(0+(distance-center_distance))
    
    normalized_lengths=np.asarray(normalized_lengths)
    
    return normalized_lengths

def normalize_elevations(distances, elevations, ranges):
    """
    Parameters
    -----------------
    distances: 1D-like array of distance points which defines a cross section
    elevations: 1D-like array set of elevation points, 
    range: a two-item tuple with a range that is contained in distances,
    with the first being the smallest and the second the largest
    
    Normalizes the elevations so that the highest point between the specified
    range si placed at0 m depth, and the rest are placed relative to this. 
    
    Returns
    A 1D-array of depths relative to the highest elevtion point.
    """
    normalized_elevations=[]
    
    elevations_in_range=[]
    
    for d,e in zip(distances, elevations):
        if d>ranges[0] and d<ranges[1]:
            elevations_in_range.append(e)
    
    for elevation in elevations:
        normalized_elevations.append(max(elevations_in_range)-elevation)
    
    normalized_elevations=np.asarray(normalized_elevations)
    
    return normalized_elevations

def get_topography_v1(header_skips, footer_skips, file):
    """
    
    Parameters
    ----------
    header_skips : array-like object specifying the genfromtxt header skips 
    of the input file
    footer_skips : array-like object specifying the genfromtxt footer skips 
    of the input file
    file : the file, coma separated, in which each column contains the x, y
    and z values for each point, as well as the ID for each point.

    Returns
    -------
    An array-like object in each which each element contains the 
    distance along profile, the elevation, and ID, for the points 
    of the cross sections selected. 
    The h and f skips must separate the d and z points which belong 
    to different sections in the original file.
    
    """
    all_profiles=[]
    
    for h,f,fl in zip(header_skips,footer_skips,file):
        x,y,z=np.genfromtxt(fl, delimiter=';', usecols=(0,1,2), 
                            skip_header=h, skip_footer=f, unpack=True)
        ID=np.genfromtxt(fl, delimiter=';', usecols=(3), dtype='str',
                            skip_header=h, skip_footer=f, unpack=True)
        
        all_profiles.append([x,y,z, ID[0]])
        
    return all_profiles


def get_topography_v2(header_skips, footer_skips, file):
    """
    
    Parameters
    ----------
    header_skips : array-like object specifying the genfromtxt header skips 
    of the input file
    footer_skips : array-like object specifying the genfromtxt footer skips 
    of the input file
    file : the file, coma separated, which contains the input parameters
    for the gryke function. Each line contains the gryke inputs for one cross
    section

    Returns
    -------
    An array-like object in each which each element contains the 
    distance along profile, the elevation, and ID, for the points 
    of the cross sections selected. 
    The h and f skips must separate the d and z points which belong 
    to different sections in the original file.
    
    """
    all_profiles=[]
    
    for h,f in zip(header_skips,footer_skips):
        d,z=np.genfromtxt(file, delimiter=';', usecols=(0,1), 
                            skip_header=h, skip_footer=f, unpack=True)
        ID=np.genfromtxt(file, delimiter=';', usecols=(2),dtype='str', 
                            skip_header=h, skip_footer=f, unpack=True)
        
        all_profiles.append([d,z, ID[0]])
        
    return all_profiles


def faults_above_crack(grid, stress, crack):
    """
    Takes a model grid as a 2D-array,
    the shear stresses produced by a crack (e.g. a dike) as 
    calculated by the 'stresses' function,  
    and a crack object instance.
    
    Returns two 2-D arrays representing the coordinates of the 
    faults at either side of the crack based on the points of 
    maximum shear stress above its upper tip.
    
    Written for the graben above the Martian dikes of the thesis
    (Rivas-Dorado et al., 2023). Named faults_above_dike until Nov 2026;
    that name is kept as an alias.
    """
    
    # x and y coordinates of the points of each fault
    left_fault=[[],[]] ; right_fault=[[],[]]
    
    for row_x,row_y,row_stress in zip(grid[0],grid[1], stress):
        # |shear stress| -> (x, y) of the nodes left and right of the crack in this row
        left={}
        right={}
        
        # Rows where the stress is zero everywhere are skipped
        i=0
        for stress in row_stress:
            if stress==0:
                i=i+1
        
        if len(row_stress)==i:
            continue
        
        # Only the rows above the upper tip: in each one, the node of maximum
        # |shear stress| on each side is taken as a point of the fault
        elif row_y[0]>crack.up_tip()[-1]:
            for x,y,stress in zip(row_x,row_y,row_stress):
                    if x<crack.up_tip()[0]:
                        left[abs(stress)]= x, y
                    elif x>crack.up_tip()[0]:
                        right[abs(stress)]= x, y
                        
            left_fault[0].append(left[max(left)][0])
            left_fault[1].append(left[max(left)][1])
            
            right_fault[0].append(right[max(right)][0])
            right_fault[1].append(right[max(right)][1])
                    
        else:
            continue
                    
    left_fault=np.asarray(left_fault)
    right_fault=np.asarray(right_fault)

    return left_fault, right_fault       

def compare_outputs_and_faults(delta,beta,grid,crack,fault1,fault_angle,elevation):
    """
    
    Parameters
    ----------
    delta : 2D array, the delta values resulting from the evaluate_parameters method
    beta: 2D array, the beta values resulting from the evaluate_parameters method
    grid : the grid of the inspected model
    crack : the crack object inspected (a dike, in the thesis)
    fault1 : a tuple of tuples, containing the x,y values of the upper and lower
    fault_angle: a float/int indicating the fault angle in degrees
    fault tips for the left fault, as in ((x1,y1)(x2,y2))
    
    Only valid for vertical dikes opened exclusively by magma pressure
    with no shearing components. This is because this function assumes 
    symetry on the delta values between both sides of the dike.
    
    Thesis function (Rivas-Dorado et al., 2023), not called anywhere in DiES.
    Notes for its use (24 Nov 2026):
    - delta and beta come from an 'evaluate_parameters' method that is no
      longer in the code (neither in main nor in this branch), so they have to
      be computed separately.
    - opt_beta = 90 - degrees(beta) assumes the angle convention of that older
      code; check it against docs/crack_coordinates_and_rotation.md, Section 10
      (orientations counterclockwise from +x since 24 Nov 2026), before use.
    - Only the left side of the crack (x < center) is inspected, and the row
      used is the first one strictly above investigation_depth.
    It reads the crack tip and center with up_tip() and c(), equivalent to the
    upper_tip and center attributes it used before (both exist in the class).
    
    Returns
    -------
    investigation_depth: the depth at which the calculation is performed
    x_delta : the x position of the maximum delta value at the investigation depth 
    x_fault: the x position of fault1 at the investigation depth
    max_delta: float, the maximum delta value at the investigation depth, in Pa
    opt_beta: float, the angle of the optimal failure planes relative to X, in degrees
    d: float, the difference between the x_delta and x_fault_value, in m
    D: float, the difference between the modeled fault angle and the angle of opt_beta, in degrees

    """
    
    # Identifying the position of the horizontal section to investigate
    investigation_depth=crack.up_tip()[1]+elevation

    i=0
    while grid[1][i][0]<=investigation_depth:
        i=i+1
    
    # Getting the maximum delta value at that elevation, in P
    # for the delta values left of the crack center, its position along the x axis
    # and the angle of the optimal shearing plane relative to the vertical
        
    new_delta=[] ; row_x=[] ; row_beta=[]
    
    for d,x,b in zip(delta[i],grid[0][0],beta[i]):
        if x<crack.c()[0]:
            new_delta.append(d)
            row_x.append(x)
            row_beta.append(b)
    
    new_delta=np.asarray(new_delta) ; row_x=np.asarray(row_x) ; row_beta=np.asarray(row_beta)
    
    max_delta=max(new_delta)
    
    j=0
    
    for d,x,b in zip(new_delta,row_x, row_beta):
        if d==max_delta:
            x_delta=x
            opt_beta=90-np.degrees(b)
        else:
            continue
           
    # Getting the x position of the faults at this depth
    fault1=np.polyfit((fault1[0][0],fault1[1][0]),(fault1[0][1],fault1[1][1]),1)
  
    x_fault_value_at_id=(investigation_depth-fault1[1])/fault1[0]
        
    # Calculating the horizontal distance between the point of max delta and the fault
    d = abs(x_delta-x_fault_value_at_id)
    
    # Calculating the difference between the optimal shearing angle and that of the modeled fault
    D = (fault_angle)- abs(opt_beta)

    return investigation_depth, x_delta, x_fault_value_at_id, max_delta, opt_beta, d, D


# Former name of faults_above_crack, kept for the thesis scripts
faults_above_dike=faults_above_crack

def point_side_of_line(A, B, P):
    """
    Determine which side of a 2D line a point lies on.

    The line is defined by two points A and B. The function computes the sign
    of the 2D cross product between vectors AB and AP to determine the relative
    position of point P.

    Parameters
    ----------
    A : tuple(float, float)
        First point defining the line (x, y).
    B : tuple(float, float)
        Second point defining the line (x, y).
    P : tuple(float, float)
        Point to test (x, y).

    Returns
    -------
    int
        1  : P lies on the left side of the line from A → B
        -1 : P lies on the right side of the line from A → B
        0  : P lies exactly on the line

    Notes
    -----
    The result depends on the direction of the line (A → B). Reversing the
    order of A and B will flip the sign of the result.
    """

    x1, y1 = A
    x2, y2 = B
    xp, yp = P

    value = (x2 - x1) * (yp - y1) - (y2 - y1) * (xp - x1)

    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0