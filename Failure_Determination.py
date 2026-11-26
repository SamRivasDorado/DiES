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

# Error class shared with the core
from DiES_core import InputError

def evaluate_stress_state(p,s,friction,T,output):
    """
    Evaluates a state of stress against the Griffith-Coulomb failure criterion.
    
    Parameters
    ----------
    p : int/float, the maximum compressive principal stress in Pa (p >= s)
    s : int/float, the minimum compressive principal stress in Pa
    friction : int/float, coefficient of internal friction (unitless)
    T : negative int/float, tensile strength in Pa. The Mohr diagram uses the
        compression-positive convention, so tension plots on the negative
        side of the normal-stress axis and T must always be < 0 (e.g. -1.5e6).
        The Griffith parabola has its vertex at sigma_n = T and the Coulomb
        line intersects the shear-stress axis at -2T = 2|T| (the cohesion).
    output: a str, 'plot' or 'values'

    The cohesion is not an input: the Coulomb envelope is taken to intersect
    the shear stress axis at 2|T| (the Griffith-Coulomb criterion), so the
    cohesion is implicitly 2|T|.
    
    Method (since Nov 2026, Phase 2, point 3B; see docs/failure_criterion.md):
    the Mohr circle of centre c = (p+s)/2 and radius r = (p-s)/2 touches the
    envelope when r equals the distance from its centre to the envelope, the
    critical radius r_c (failure_contact). The state is at failure when
    |r/r_c - 1| <= FAILURE_TOLERANCE, below failure when r is smaller, and
    above failure when r is larger or when s < T (tension beyond the tensile
    strength). The point of the envelope closest to the centre gives sigma_n,
    tau and 2theta. Until Nov 2026 the circle was sampled instead (see
    evaluate_stress_state_sampling).

    Returns
    -------
    If output = 'plot', returns a Mohr circle plot of the given stress state
    and the Griffith and Coulomb criteria (drawn by plotting.plot_mohr_circle
    since Nov 2026).
    If output = 'values', returns a tuple with the normal stress, shear stress
    (MPa), failure angle (2theta, degrees), and the failure mode, as a str:
        'Below': the circle is below failure
        'Above': the circle is above failure
        'Mode I': tensional failure: the circle touches the vertex of the
                  parabola (sigma3 = T; only for c < |T|)
        'Mixed-mode I-II': shear failure on the Griffith parabola (sigma_n < 0)
        'Mode II - tension-compression': shear failure on the Coulomb line,
                  with sigma3 < 0
        'Mode II - compression': shear failure on the Coulomb line, sigma3 >= 0
        'Singular': p or s is not finite (NaN/inf), e.g. at a node lying
                    exactly on a crack tip, where the elastic solution is singular.
                    The values are NaN and no failure plane is defined.
    The values are NaN unless the state is at failure.
    For an isotropic state (p == s) the Mohr circle is a point: it is 'Below'
    if p > T and 'Above' if p <= T (isotropic tension beyond the tensile
    strength), with NaN values, since no failure plane is defined.
    p < s, or a non-negative T, raise InputError.

    Arrays (since Nov 2026, Phase 2): with output = 'values', p, s, friction and T
    may be arrays (or a mix of arrays and numbers, broadcast together, e.g. the
    stresses of a grid and the friction and T of each node). The result is then four
    arrays of that shape: sn, ss and 2theta (float, NaN where there is no
    failure) and the mode (str). See evaluate_stress_states.
    """
    
    if output not in ('values','plot'):
        raise InputError("output must be 'values' or 'plot', got "+repr(output))
    
    # Arrays: every node of a grid in one call
    if output=='values' and any(np.ndim(v)>0 for v in (p,s,friction,T)):
        return evaluate_stress_states(p,s,friction,T)
    
    # The Mohr diagram is drawn by plotting.plot_mohr_circle, which takes the
    # values from this function (imported here to avoid a circular import)
    if output=='plot':
        # Same input checks as for the values, with the messages of a drawing
        if not T<0:
            raise InputError('T must be negative (tension is negative on the Mohr diagram), got '+str(T)+' Pa')
        if not (np.isfinite(p) and np.isfinite(s)):
            raise InputError('Cannot draw a Mohr circle with non-finite stresses (p = '+str(p)+', s = '+str(s)+')')
        if p<s:
            raise InputError('p must be the maximum compressive principal stress, but p = '
                             +str(p)+' Pa < s = '+str(s)+' Pa')
        if p==s:
            raise InputError('p = s: the state of stress is isotropic and its Mohr circle is a point')
        import plotting
        return plotting.plot_mohr_circle(p,s,friction,T)
    
    # Numbers: evaluated as one-node arrays, returned as plain numbers and a str
    sn, ss, two_theta, mode = evaluate_stress_states(p,s,friction,T)
    return float(sn), float(ss), float(two_theta), str(mode)


# Half-width of the band of r/r_c in which a state of stress is taken to be at
# failure (Phase 2, point 3B, Nov 2026): |r/r_c - 1| <= FAILURE_TOLERANCE.
# 0.035 reproduces the band of the sampling algorithm used until then
# (measured: r/r_c from about 0.965 to 1.035; see docs/failure_criterion.md).
FAILURE_TOLERANCE = 0.035


def failure_contact(p,s,friction,T):
    """
    Contact of the Mohr circle with the Griffith-Coulomb envelope, computed
    analytically (Phase 2, point 3B, Nov 2026; see docs/failure_criterion.md).
    
    The circle of a state of stress has centre c = (p+s)/2 and radius
    r = (p-s)/2. It touches the envelope when r equals the distance from its
    centre to the envelope: the critical radius r_c. The point of the envelope
    closest to the centre is where it touches: the stresses on the failure plane.
    
    With k = |T| (MPa, compression positive), the envelope is the Griffith
    parabola tau^2 = 4k(sigma_n + k) for -k <= sigma_n < 0 and the Coulomb line
    tau = 2k + friction*sigma_n for sigma_n >= 0. r_c is the smaller of the
    distances from the centre to the two:
      - parabola: tangent where sigma_n = c - 2k, with r_c = sqrt(4kc), if that
        point is on it (k <= c <= 2k); if c < k, the closest point is the vertex
        (-k, 0): r_c = c + k ('Mode I'); if c > 2k, the end (0, 2k).
      - line: r_c = (2k + friction*c)/sqrt(1 + friction^2), tangent where
        sigma_n = c - r_c*friction/sqrt(1 + friction^2), if that is >= 0; otherwise the
        end (0, 2k).
    
    Parameters
    ----------
    p, s : arrays or numbers, maximum and minimum compressive principal
           stresses in Pa (p >= s)
    friction : array or number, coefficient of internal friction
    T : array or number, tensile strength in Pa (negative)
    All are broadcast together; no checks are made here (see evaluate_stress_states).
    
    Returns
    -------
    ratio : r/r_c (inf where the centre is beyond the vertex, c <= T, and r > 0)
    sn, ss : normal and shear stress at the contact point, in MPa
    two_theta : angle at the centre between sigma1 and the contact point, in
                degrees (180 for 'Mode I')
    mode : str array, the failure mode the contact corresponds to
           ('Mode I', 'Mixed-mode I-II', 'Mode II - tension-compression' or
           'Mode II - compression'), whatever the ratio
    """
    # Everything in MPa, compression positive; k = |T|
    p=np.asarray(p,dtype=float)/1e6 ; s=np.asarray(s,dtype=float)/1e6
    k=-np.asarray(T,dtype=float)/1e6 ; mu=np.asarray(friction,dtype=float)
    p,s,k,mu=np.broadcast_arrays(p,s,k,mu)
    
    # Centre and radius of the Mohr circle
    c=(p+s)/2 ; r=(p-s)/2
    root=np.sqrt(1+mu**2)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        # 1) Griffith parabola: unconstrained tangent point sigma_n = c - 2k
        sg=c-2*k
        vertex=sg<-k            # c < k: the closest point is the vertex (-k, 0)
        end_g=sg>0              # c > 2k: the closest point is the end (0, 2k)
        d_g=np.where(vertex, c+k, np.where(end_g, np.hypot(c,2*k), np.sqrt(4*k*np.clip(c,0,None))))
        sn_g=np.where(vertex, -k, np.where(end_g, 0.0, sg))
        tau_g=np.where(vertex, 0.0, np.where(end_g, 2*k, np.sqrt(4*k*np.clip(sg+k,0,None))))
        
        # 2) Coulomb line: foot of the perpendicular from the centre
        d_line=(2*k+mu*c)/root
        sn_line=c-d_line*mu/root
        on_line=sn_line>=0      # otherwise the closest point is the end (0, 2k)
        d_c=np.where(on_line, d_line, np.hypot(c,2*k))
        sn_c=np.where(on_line, sn_line, 0.0)
        tau_c=np.where(on_line, 2*k+mu*sn_line, 2*k)
        
        # 3) The closer of the two is where the circle touches first
        griffith=d_g<=d_c
        r_c=np.where(griffith, d_g, d_c)
        sn=np.where(griffith, sn_g, sn_c)
        ss=np.where(griffith, tau_g, tau_c)
        
        # r/r_c; r_c <= 0 means the centre is at or beyond the vertex (c <= T)
        ratio=np.where(r_c>0, r/np.where(r_c>0, r_c, 1.0), np.where(r>0, np.inf, 0.0))
        
        # 2theta: angle at the centre from sigma1 (towards +sigma_n) to the contact point
        two_theta=np.degrees(np.arctan2(ss, sn-c))
    
    # Failure mode of the contact: the vertex is Mode I; the parabola, mixed
    # mode; the line (or its end at sigma_n = 0), Mode II, with or without
    # tension depending on the sign of sigma3
    mode=np.where(s<0, 'Mode II - tension-compression', 'Mode II - compression')
    mode=np.where(griffith & ~vertex & ~end_g, 'Mixed-mode I-II', mode)
    mode=np.where(griffith & vertex, 'Mode I', mode)
    
    return ratio, sn, ss, two_theta, mode


def evaluate_stress_states(p,s,friction,T):
    """
    evaluate_stress_state for arrays: evaluates the Griffith-Coulomb criterion
    at every node of a grid in one call (Phase 2: interface for arrays in
    point 3A; analytical criterion, without loops, in point 3B, Nov 2026).

    Parameters
    ----------
    p, s : arrays (or numbers) of the maximum and minimum compressive principal
           stresses in Pa, p >= s at every node
    friction : array or number, coefficient of internal friction of each node
    T : array or number, tensile strength of each node in Pa (negative)
    All are broadcast together to a common shape.

    Returns
    -------
    sn, ss : float arrays, normal and shear stress on the failure plane (MPa):
             the point of the Mohr circle at the angle 2theta of the contact
    two_theta : float array, the failure angle 2theta (degrees)
    mode : str array, the failure mode of each node (see evaluate_stress_state)
    sn, ss and two_theta are NaN where there is no failure.

    Raises InputError if any T >= 0 or if p < s at any node with finite stresses.
    """
    # Common shape of the inputs (e.g. a grid with one friction and T per node)
    p,s,friction,T=np.broadcast_arrays(np.asarray(p,dtype=float),np.asarray(s,dtype=float),
                                  np.asarray(friction,dtype=float),np.asarray(T,dtype=float))
    
    # T must be negative: it is the tensile strength on the compression-positive
    # normal-stress axis of the Mohr diagram
    if np.any(~(T<0)):
        bad=T[~(T<0)].flat[0]
        raise InputError('T must be negative (tension is negative on the Mohr diagram), got '+str(bad)+' Pa')
    
    # Non-finite stresses: a node on a crack tip, where the solution is singular
    finite=np.isfinite(p) & np.isfinite(s)
    
    # p must be the maximum compressive principal stress
    wrong=finite & (p<s)
    if np.any(wrong):
        i=np.argwhere(wrong)[0] if wrong.ndim else ()
        raise InputError('p must be the maximum compressive principal stress, but p = '
                         +str(p[tuple(i)])+' Pa < s = '+str(s[tuple(i)])+' Pa')
    
    ratio, _, _, two_theta, contact_mode = failure_contact(p,s,friction,T)
    
    # Stresses on the failure plane: the point of the Mohr circle in the direction
    # 2theta of the contact. At r = r_c it is the contact point itself; within
    # the band (r slightly different from r_c) it is the point of the circle, i.e.
    # the stresses actually acting on that plane, as the sampling algorithm gave
    # (inf/NaN stresses of singular nodes would warn here; they are discarded below)
    with np.errstate(invalid='ignore'):
        c=(p+s)/2e6 ; r=(p-s)/2e6
        sn=c+r*np.cos(np.radians(two_theta)) ; ss=r*np.sin(np.radians(two_theta))
    
    # Classification by r/r_c: at failure within the band, above beyond it, and
    # always above when sigma3 is beyond the tensile strength (s < T)
    at_failure=np.abs(ratio-1)<=FAILURE_TOLERANCE
    above=(ratio>1+FAILURE_TOLERANCE) | (s<T)
    
    mode=np.full(p.shape, 'Below', dtype='<U29')
    mode=np.where(at_failure, contact_mode, mode)
    mode=np.where(above, 'Above', mode)
    
    # Isotropic states: the circle is a point, 'Below' unless it is beyond T
    isotropic=finite & (p==s)
    mode=np.where(isotropic, np.where(p>T, 'Below', 'Above'), mode)
    
    # Singular nodes
    mode=np.where(finite, mode, 'Singular')
    
    # Values only where the state is at failure
    failed=np.isin(mode, ('Mode I','Mixed-mode I-II','Mode II - tension-compression','Mode II - compression'))
    sn=np.where(failed, sn, np.nan) ; ss=np.where(failed, ss, np.nan)
    two_theta=np.where(failed, two_theta, np.nan)
    
    # A str array as wide as the longest label found
    mode=np.asarray(mode.tolist()) if mode.ndim else np.asarray(str(mode))
    
    return sn, ss, two_theta, mode


def evaluate_stress_state_sampling(p,s,friction,T,output='values'):
    """
    The failure criterion used until Nov 2026: it samples the Mohr circle and
    looks for points where the circle is tangent to the Griffith parabola or
    the Coulomb line, within a 5 % tolerance in slope and intercept.
    
    Replaced in evaluate_stress_state by the analytical criterion (Phase 2,
    point 3B; see failure_contact and docs/failure_criterion.md). Kept, with its
    logic unchanged, to compare old and new results. Only output = 'values'
    and scalar inputs. Known bug, not fixed here: when a sample of the circle
    falls exactly on sigma_n = 0 (e.g. s = 0), no branch assigns the results
    and it raises UnboundLocalError.
    
    Parameters
    ----------
    p : int/float, the maximum compressive principal stress in Pa (p > s)
    s : int/float, the minimum compressive principal stress in Pa
    friction : int/float, coefficient of internal friction (unitless)
    T : negative int/float, tensile strength in Pa. The Mohr diagram uses the
        compression-positive convention, so tension plots on the negative
        side of the normal-stress axis and T must always be < 0 (e.g. -1.5e6).
        The Griffith parabola has its vertex at sigma_n = T and the Coulomb
        line intersects the shear-stress axis at -2T = 2|T| (the cohesion).
    output: a str, 'plot' or 'values'

    The cohesion is not an input: the Coulomb envelope is taken to intersect
    the shear stress axis at 2|T| (the Griffith-Coulomb criterion), so the
    cohesion is implicitly 2|T|.

    Returns
    -------
    If output = 'plot', returns a Mohr circle plot of the given stress state
    and the Griffith and Coulomb criteria (drawn by plotting.plot_mohr_circle
    since Nov 2026).
    If output = 'values', returns a tuple with the normal stress, shear stress,
    failure angle (2theta), and the failure mode, as a str:
        'Below': the circle is below failure
        'Above': the circle is above failure
        'Mode I': the circle is at tensional failure
        'Mode II - tension-compression': shear failure, with sigma_n > 0
        'Mixed-mode I-II': shear failure, with sigma_n < 0
        'Mode II - compression': shear failure under compression only
        'Singular': p or s is not finite (NaN/inf), e.g. at a node lying
                    exactly on a crack tip, where the elastic solution is singular.
                    The values are NaN and no failure plane is defined.
    For an isotropic state (p == s) the Mohr circle is a point: it is 'Below'
    if p > T and 'Above' if p <= T (isotropic tension beyond the tensile
    strength), with NaN values, since no failure plane is defined.
    p < s, or a non-negative T, raise InputError.

    Arrays (since Nov 2026, Phase 2): with output = 'values', p, s, friction and T
    may be arrays (or a mix of arrays and numbers, broadcast together, e.g. the
    stresses of a grid and the friction and T of each node). The result is then four
    arrays of that shape: sn, ss and 2theta (float, NaN where there is no
    failure) and the mode (str), node by node identical to the scalar call.
    See evaluate_stress_states.

    """
    
    # T must be negative: it is the tensile strength on the compression-positive
    # normal-stress axis of the Mohr diagram
    if not T<0:
        raise InputError('T must be negative (tension is negative on the Mohr diagram), got '+str(T)+' Pa')
    
    # Until Nov 2026, p < s and p == s printed a message and called sys.exit(),
    # which would stop a whole model (or an application) because of a single node.
    
    # Non-finite stresses: a node on a crack tip, where the solution is singular
    if not (np.isfinite(p) and np.isfinite(s)):
        if output=='values':
            return float('NaN'), float('NaN'), float('NaN'), 'Singular'
        raise InputError('Cannot draw a Mohr circle with non-finite stresses (p = '+str(p)+', s = '+str(s)+')')
    
    if p<s:
        raise InputError('p must be the maximum compressive principal stress, but p = '
                         +str(p)+' Pa < s = '+str(s)+' Pa')
    
    # Isotropic state: the Mohr circle is a single point on the normal-stress axis
    if p==s:
        if output=='values':
            mode='Below' if p>T else 'Above'
            return float('NaN'), float('NaN'), float('NaN'), mode
        raise InputError('p = s: the state of stress is isotropic and its Mohr circle is a point')
        
    # elif p<0 and s<0:
    #     print ('This is a biaxial tension situation.')
    #     print ('This algorithm is not suited to evaluate this scenario.')
    #     print ('Please revise your values and try again.')
    
    if output not in ('values','plot'):
        raise InputError("output must be 'values' or 'plot', got "+repr(output))
    
    # Since Nov 2026 (Phase 1) the Mohr diagram is drawn by plotting.plot_mohr_circle,
    # which takes the values from this function. Imported here to avoid a circular import.
    if output=='plot':
        raise InputError("evaluate_stress_state_sampling only returns values; use evaluate_stress_state(..., 'plot')")
    
    # Redefining the inputs to MPa. T keeps its sign: everything below
    # (x<T, the parabola vertex at T, the Coulomb intercept -2T) is written for T<0.
    # Before Nov 2026 this line was T=(-T)/1e6, which flipped the envelope into
    # compression and classified most stable states as 'Above'.
    p=p/1e6 ; s=s/1e6 ; T=T/1e6
    
    # Mohr circle radius # Mohr circle x coordinate
    r = abs(p - s)/2 ; x_center = p-(p-s)/2
                                    
    # The points around the Mohr circle
    circle_points = np.linspace(s,p,int(r)*50)
    
    # Calculating the slope and intercept of the Mohr-Coulomb failure line
    slope_l = friction ; intercept_l = -2*T
    
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
            sn=float('NaN') ; ss=float('NaN') ; angle=float('NaN') ; mode='Above'
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
                sn=float('NaN') ; ss=float('NaN') ; angle=float('NaN'); mode='Above'
                break
            
            # 2.1.3 ) If all points except the first are below failure, 
            # the circle is under tensional failure.
            elif 0.95<len(circle_points)/j<1.05:
                i=1
            
                slope_p = slope_c
                intercept_p = T - slope_p*T 
                
                circle_tangent_params.append((slope_c, intercept_c))
                parabole_tangent_params.append((slope_p, intercept_p))
                
                sn=T ; ss=0 ; angle=180 ; mode='Mode I'
                
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
                        mode='Mode II - tension-compression'
                    elif sn<0:
                        mode='Mixed-mode I-II'
                elif s>0 and p>0:
                    mode='Mode II - compression'
            
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
                y_line=-2*T+friction*x
                
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
            sn=float('NaN') ; ss=float('NaN') ; angle=float('NaN'); mode='Below'
            
   
        elif k>=1:
            i=2
            # print ('The circle is above failure')
            sn=float('NaN'); ss=float('NaN') ; angle=float('NaN'); mode='Above'
                    
    return sn, ss, angle, mode


def failure_plane_orientation(s1a,theta,mode,side):
    """
    Orientation of the failure plane formed at a point, in the 2D section.
    
    Parameters
    ----------
    s1a : float, orientation of sigma1 in degrees, counterclockwise from the
          +x axis (as returned by model.principal_stresses), modulo 180.
    theta : float, in degrees, half of the 2theta angle returned by
          evaluate_stress_state: the angle between sigma1 and the NORMAL to the
          failure plane. The plane itself is at delta = 90 - theta from sigma1.
    mode : str, the failure mode returned by evaluate_stress_state.
    side : -1 if the point is on the left of the crack (x <= x of its center),
           +1 if it is on the right.
    
    Returns
    -------
    The orientation of the failure plane in radians, counterclockwise from +x
    (the plane trace is the direction (cos b, sin b)), or NaN if there is no
    failure plane (modes 'Below' and 'Above', or theta = NaN).
    
    - Mode I: the plane contains sigma1 (theta = 90), so b = s1a + theta - 90 = s1a.
    - Shear modes ('Mixed-mode I-II', 'Mode II - tension-compression',
      'Mode II - compression'): there are two conjugate planes, at s1a + delta
      and s1a - delta. The one that dips towards the crack is selected, as in
      the original code: on the left of the crack the plane dipping towards +x,
      on the right the plane dipping towards -x. Above a dike these are the
      graben-forming normal faults.
      A plane trace at orientation b (modulo 180) dips towards +x when
      90 < b < 180, i.e. when sin(2b) < 0, and towards -x when sin(2b) > 0;
      the candidate with the most negative (left) or most positive (right)
      sin(2b) is chosen, which also works when sigma1 is far from vertical.
      If sigma1 is tilted so much that both conjugates dip the same way
      (e.g. next to an inclined fault), the one closest to dipping towards
      the crack (the steeper one, on that side) is chosen.
    
    Arrays (since Nov 2026, Phase 2): s1a, theta, mode and side may be arrays
    (broadcast together); the result is then a float array of that shape,
    node by node identical to the scalar call.
    
    Until Nov 2026 this was done with four hand-tuned branches depending on the
    side and on the sign of s1a, which reduce to b = s1a - delta (left) and
    b = s1a + delta (right), and the traces were drawn as (cos b, -sin b). That
    gave the right plane only where sigma1 was exactly vertical; elsewhere the
    error was twice the tilt of sigma1. See docs/crack_coordinates_and_rotation.md,
    Section 10.
    """
    # Arrays: same rules, applied to every node at once
    if any(np.ndim(v)>0 for v in (s1a,theta,mode,side)):
        return _failure_plane_orientations(s1a,theta,mode,side)
    
    # No failure plane without failure
    if not np.isfinite(theta) or mode in ('Below','Above'):
        return float('NaN')
    
    # Mode I: the plane contains sigma1
    if mode=='Mode I':
        return np.deg2rad(s1a+theta-90)
    
    # Shear modes: the two conjugate planes, at +/- delta from sigma1
    delta=90-theta
    candidates=(s1a+delta, s1a-delta)
    # sin(2b) < 0: the plane dips towards +x; sin(2b) > 0: towards -x
    dip_sense=[np.sin(np.deg2rad(2*c)) for c in candidates]
    
    # Left of the crack, the plane dipping most towards +x (towards the crack);
    # right of it, the one dipping most towards -x
    if side<0:
        chosen=candidates[int(np.argmin(dip_sense))]
    else:
        chosen=candidates[int(np.argmax(dip_sense))]
    
    return np.deg2rad(chosen)


def _failure_plane_orientations(s1a,theta,mode,side):
    """
    failure_plane_orientation for arrays (Phase 2, point 3A, Nov 2026). Same
    rules, written with np.where; see failure_plane_orientation.
    """
    # Common shape of the inputs; the mode as a str array
    s1a,theta,side=np.broadcast_arrays(np.asarray(s1a,dtype=float),np.asarray(theta,dtype=float),
                                       np.asarray(side,dtype=float))
    mode=np.broadcast_to(np.asarray(mode).astype(str),s1a.shape)
    
    # Shear modes: the two conjugate planes, at +/- delta from sigma1, and the
    # sense of dip of each (sin(2b) < 0: towards +x; sin(2b) > 0: towards -x)
    delta=90-theta
    plus=s1a+delta ; minus=s1a-delta
    dip_plus=np.sin(np.deg2rad(2*plus)) ; dip_minus=np.sin(np.deg2rad(2*minus))
    
    # Left of the crack, the plane dipping most towards +x; right of it, the one
    # dipping most towards -x. On a tie the first candidate (s1a + delta) is kept,
    # as np.argmin/np.argmax do in the scalar version.
    chosen=np.where(side<0,
                    np.where(dip_plus<=dip_minus,plus,minus),
                    np.where(dip_plus>=dip_minus,plus,minus))
    
    # Mode I: the plane contains sigma1
    orientation=np.where(mode=='Mode I',np.deg2rad(s1a+theta-90),np.deg2rad(chosen))
    
    # No failure plane without failure
    no_plane=~np.isfinite(theta) | (mode=='Below') | (mode=='Above')
    return np.where(no_plane,np.nan,orientation)
