"""
Tests of the analytical core (displacements, stresses) of DiES_core.

Run with pytest, or directly: python tests/test_core_against_reference.py
See docs/crack_coordinates_and_rotation.md, Section 8.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from DiES_core import displacements, stresses, crack_frame, InputError
from reference_westergaard import field

warnings.filterwarnings('ignore')

E, NU = 10e9, 0.25
G = E/(2*(1 + NU)) ; LAM = E*NU/((1 + NU)*(1 - 2*NU))

GEOMETRIES = {
    'vertical': ((0, -1000), (0, -5000)),
    'dip 60 towards +x': ((-500, -1000), (500, -2732.0508)),
    'dip 60 towards -x': ((500, -1000), (-500, -2732.0508)),
    'dip 30 towards +x': ((-1500, -1000), (1500, -2732.0508)),
    'horizontal': ((-1500, -2000), (1500, -2000)),
    'tips in reverse order': ((500, -2732.0508), (-500, -1000)),
}
LOADS = {'mode I': (2.67e6, 0), 'mode II': (0, 5e6), 'mixed': (2e6, -3e6)}


def center_and_half_length(up, lo):
    return ((up[0] + lo[0])/2, (up[1] + lo[1])/2), np.hypot(up[0] - lo[0], up[1] - lo[1])/2


def dies_fields(p, s, up, lo, X, Y, s3=0):
    c, a = center_and_half_length(up, lo)
    S = stresses(p, s, s3, 0, 0, 0, 0, 0, NU, E, up, c, lo, a, (X, Y))
    U = displacements(p, s, s3, NU, E, up, c, lo, (X, Y))
    return np.array(S[:3]), np.array(U[:2]), S, U


def test_against_westergaard():
    rng = np.random.default_rng(1)
    for gname, (up, lo) in GEOMETRIES.items():
        X, Y = rng.uniform([-4000, -6000], [4000, -50], size=(2000, 2)).T
        center, t, n, a, h = crack_frame(up, lo)
        upper, lower = center - a*t, center + a*t
        for lname, (p, s) in LOADS.items():
            S, U, _, _ = dies_fields(p, s, up, lo, X, Y)
            W = np.array(field(p, h*s, NU, E, upper, lower, X, Y))
            err_s = np.abs(S - W[:3]).max()/np.abs(W[:3]).max()
            err_u = np.abs(U - W[3:]).max()/np.abs(W[3:]).max()
            assert err_s < 1e-9 and err_u < 1e-9, (gname, lname, err_s, err_u)


def wall_checks(p, s, up, lo):
    center, t, n, a, h = crack_frame(up, lo)
    e = 0.5
    Up = np.array(dies_fields(p, s, up, lo, *(center + e*n))[1])
    Um = np.array(dies_fields(p, s, up, lo, *(center - e*n))[1])
    jump = Up - Um
    q = center + 0.25*a*t + e*n
    S = dies_fields(p, s, up, lo, *q)[0]
    traction = np.array([[S[0], S[2]], [S[2], S[1]]]) @ n
    return jump @ n, jump @ t, traction @ n, traction @ t, a, h


def test_walls_and_jumps():
    for gname, (up, lo) in GEOMETRIES.items():
        for p, s in ((2.67e6, 0), (0, 5e6)):
            jn, jt, tn, tt, a, h = wall_checks(p, s, up, lo)
            # Displacement jump at the center: opening normal, slip parallel to the crack
            assert np.isclose(jn, 2*(1 - NU)*p*a/G, rtol=2e-3, atol=1e-3), (gname, jn)
            assert np.isclose(jt, 2*(1 - NU)*h*s*a/G, rtol=2e-3, atol=1e-3), (gname, jt)
            # Tractions on the n-side wall cancel the driving stresses
            assert np.isclose(tn, -p, rtol=2e-3, atol=2e3), (gname, tn)
            assert np.isclose(tt, -h*s, rtol=2e-3, atol=2e3), (gname, tt)


def hooke_from_displacements(p, s, up, lo, P, s3=0, step=0.5):
    c, a = center_and_half_length(up, lo)
    u = lambda X, Y: np.array(displacements(p, s, s3, NU, E, up, c, lo, (X, Y)))
    du_dx = (u(P[0] + step, P[1]) - u(P[0] - step, P[1]))/(2*step)
    du_dy = (u(P[0], P[1] + step) - u(P[0], P[1] - step))/(2*step)
    exx, eyy, exy = du_dx[0], du_dy[1], 0.5*(du_dy[0] + du_dx[1])
    return np.array([LAM*(exx + eyy) + 2*G*exx, LAM*(exx + eyy) + 2*G*eyy, 2*G*exy,
                     G*du_dx[2], G*du_dy[2]])


def test_hooke_consistency():
    for gname, (up, lo) in GEOMETRIES.items():
        for P in ((800, -1500), (-900, -2600), (300, -4500)):
            H = hooke_from_displacements(2e6, -3e6, up, lo, P, s3=1.5e6)
            S = dies_fields(2e6, -3e6, up, lo, *P, s3=1.5e6)[2]
            assert np.allclose(H, [S[0], S[1], S[2], S[3], S[4]], atol=5e2), (gname, P)


def test_shear_sense():
    # dpy > 0: the hanging wall (x > 0 block if vertical) slides down-dip
    for gname in ('vertical', 'dip 60 towards +x', 'dip 60 towards -x'):
        up, lo = GEOMETRIES[gname]
        center, t, n, a, h = crack_frame(up, lo)
        hanging_wall = h*n
        U = lambda P: np.array(dies_fields(0, 5e6, up, lo, *P)[1])
        jump = U(center + 0.5*hanging_wall) - U(center - 0.5*hanging_wall)
        assert jump @ t > 0 and jump[1] < 0, (gname, jump)


def test_far_field_vanishes():
    for gname, (up, lo) in GEOMETRIES.items():
        center, t, n, a, h = crack_frame(up, lo)
        S = dies_fields(2e6, 3e6, up, lo, *(center + 200*a*n))[0]
        assert np.abs(S).max() < 1e-4*2e6, gname


def test_vectorised_equals_pointwise():
    up, lo = GEOMETRIES['dip 60 towards +x']
    c, a = center_and_half_length(up, lo)
    X, Y = np.meshgrid(np.arange(-3950, 4000, 300), np.arange(-6000, -10, 300))
    V = np.array(stresses(2e6, 3e6, 0, 0, 0, 0, 0, 0, NU, E, up, c, lo, a, (X, Y))[:3])
    P = np.array([[stresses(2e6, 3e6, 0, 0, 0, 0, 0, 0, NU, E, up, c, lo, a, (x, y))[:3]
                   for x, y in zip(rx, ry)] for rx, ry in zip(X, Y)]).transpose(2, 0, 1)
    assert np.allclose(V, P, rtol=1e-12, atol=1e-6)


def test_inconsistent_geometry_raises():
    up, lo = GEOMETRIES['vertical']
    for args in ((up, (0, -2000), lo, 2000), (up, (0, -3000), lo, 1000)):
        try:
            stresses(1e6, 0, 0, 0, 0, 0, 0, 0, NU, E, args[0], args[1], args[2], args[3], (100, -100))
        except InputError:
            continue
        raise AssertionError('no InputError for ' + str(args))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('OK  ', test.__name__)
    print(len(tests), 'tests passed')
