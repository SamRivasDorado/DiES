"""
Independent reference solution for the tests of DiES_core.

Plane-strain crack of half-length a with uniformly loaded walls (normal driving
stress p, mode I; in-plane shear driving stress s, mode II), written with the
Westergaard complex potential Z(z) = z/(z**2 - a**2)**0.5 - 1 instead of tripolar
coordinates, and rotated to the orientation of the crack. Tension positive.

It shares no code with DiES_core: only numpy complex arithmetic. The local frame
follows the same definition (t from the upper to the lower tip, n = t rotated +90
degrees), so the shear stress s here is the local one, i.e. h*dpy in DiES.

Reference: Westergaard (1939); Pollard and Segall (1987).
"""

import numpy as np


def frame(upper_tip, lower_tip):
    up = np.asarray(upper_tip, float) ; lo = np.asarray(lower_tip, float)
    center = (up + lo)/2
    t = (lo - up)/np.linalg.norm(lo - up)       # x' along the crack, from the upper tip
    n = np.array([-t[1], t[0]])                 # y' = t rotated +90 degrees
    return center, t, n, np.linalg.norm(lo - up)/2


def field(p, s, nu, E, upper_tip, lower_tip, X, Y):
    """
    Returns sxx, syy, sxy, ux, uy in the global axes at the points (X, Y),
    for the crack between upper_tip and lower_tip (the upper tip must be the shallower one).
    """
    c, t, n, a = frame(upper_tip, lower_tip)
    G = E/(2*(1 + nu)) ; k = 3 - 4*nu                     # Kolosov constant, plane strain
    dx = np.asarray(X, float) - c[0] ; dy = np.asarray(Y, float) - c[1]
    xl = dx*t[0] + dy*t[1] ; yl = dx*n[0] + dy*n[1]
    z = xl + 1j*yl
    sq = np.sqrt(z - a)*np.sqrt(z + a)                    # branch cut along the crack
    Z = z/sq - 1                                          # stress potential
    Zp = -a**2/sq**3                                      # dZ/dz
    Zh = sq - z                                           # integral of Z, -> 0 far away
    # Mode I: crack along x', loaded on the faces normal to y'
    sxx = p*(Z.real - yl*Zp.imag) ; syy = p*(Z.real + yl*Zp.imag) ; sxy = p*(-yl*Zp.real)
    ux = p*((k - 1)/2*Zh.real - yl*Z.imag)/(2*G) ; uy = p*((k + 1)/2*Zh.imag - yl*Z.real)/(2*G)
    # Mode II
    sxx += s*(2*Z.imag + yl*Zp.real) ; syy += s*(-yl*Zp.real) ; sxy += s*(Z.real - yl*Zp.imag)
    ux += s*((k + 1)/2*Zh.imag + yl*Z.real)/(2*G) ; uy += s*(-(k - 1)/2*Zh.real - yl*Z.imag)/(2*G)
    # Rotation from (x', y') = (t, n) to the global axes
    Sxx = sxx*t[0]**2 + syy*n[0]**2 + 2*sxy*t[0]*n[0]
    Syy = sxx*t[1]**2 + syy*n[1]**2 + 2*sxy*t[1]*n[1]
    Sxy = sxx*t[0]*t[1] + syy*n[0]*n[1] + sxy*(t[0]*n[1] + n[0]*t[1])
    Ux = ux*t[0] + uy*n[0] ; Uy = ux*t[1] + uy*n[1]
    return Sxx, Syy, Sxy, Ux, Uy
