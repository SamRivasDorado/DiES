"""
Tests of the analytical Griffith-Coulomb criterion (Phase 2, point 3B, Nov 2026):
failure_contact, evaluate_stress_state(s) and the dynamic model with nodes that
jump over the failure band. The method is explained in docs/failure_criterion.md.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')

from Failure_Determination import (FAILURE_TOLERANCE, evaluate_stress_state, evaluate_stress_state_sampling,
                                   failure_contact)
from Modeling_functions import compute_dynamic_model

CFI, T = 0.57, -1.5e6
K = 1.5   # |T| in MPa


def envelope(sn, cfi=CFI, k=K):
    """Shear stress of the Griffith-Coulomb envelope at sigma_n (MPa)."""
    return np.sqrt(4*k*(sn + k)) if sn < 0 else 2*k + cfi*sn


def critical_state(c, cfi=CFI, t=T, factor=1.0):
    """(p, s) in Pa of the circle of centre c (MPa) and radius factor*r_c."""
    big = 1e6   # any circle gives r_c: take r = 1 MPa just to call failure_contact
    ratio = float(failure_contact((c + 1)*big, (c - 1)*big, cfi, t)[0])
    r_c = 1/ratio
    return (c + factor*r_c)*big, (c - factor*r_c)*big


def test_contact_point_is_tangent():
    """At r = r_c the contact point is on the circle and on the envelope, and the
    radius to it is normal to the envelope (where the envelope is smooth)."""
    for cfi, t in ((0.57, -1.5e6), (0.8, -3e6), (0.3, -0.5e6)):
        k = -t/1e6
        for c in np.linspace(-0.9*k, 40, 157):
            p, s = critical_state(c, cfi, t)
            ratio, sn, ss, two_theta, mode = (np.asarray(v).item() for v in failure_contact(p, s, cfi, t))
            r = (p - s)/2e6
            assert abs(ratio - 1) < 1e-9
            assert abs(np.hypot(sn - c, ss) - r) < 1e-9*max(1, r), (c, cfi)          # on the circle
            assert abs(ss - envelope(sn, cfi, k)) < 1e-9*max(1, ss), (c, cfi)         # on the envelope
            # Normal to the envelope at the contact: compare with the direction of the radius
            if mode == 'Mixed-mode I-II':
                slope = 2*k/ss                         # d tau / d sigma_n of the parabola
            elif mode.startswith('Mode II') and sn > 1e-9:
                slope = cfi
            else:
                continue                               # vertex or kink: not smooth
            radius = np.array([sn - c, ss]) / r
            tangent = np.array([1.0, slope]) / np.hypot(1, slope)
            assert abs(radius @ tangent) < 1e-9, (c, cfi, mode)


def test_modes_by_zone():
    # c < |T|: the circle touches the vertex: Mode I, sigma_n = T, tau = 0, 2theta = 180
    # (at 0.999 r_c: at exactly r_c, s = T, and rounding may leave s a hair below T, 'Above')
    for c in (-1.0, 0.0, 1.0, 1.45):
        sn, ss, two_theta, mode = evaluate_stress_state(*critical_state(c, factor=0.999), CFI, T, 'values')
        assert mode == 'Mode I' and abs(sn + K) < 0.01 and abs(ss) < 1e-9 and np.isclose(two_theta, 180), c
    # |T| <= c up to where the line takes over (~2.4 MPa here): parabola, mixed mode
    sn, ss, two_theta, mode = evaluate_stress_state(*critical_state(2.2), CFI, T, 'values')
    assert mode == 'Mixed-mode I-II' and np.isclose(sn, 2.2 - 2*K) and sn < 0
    # Coulomb line: 2theta = 90 + phi, with tan(phi) = cfi; with or without tension
    phi = np.degrees(np.arctan(CFI))
    p, s = critical_state(4.0)
    assert s < 0 and evaluate_stress_state(p, s, CFI, T, 'values')[3] == 'Mode II - tension-compression'
    for c in (4.0, 10.0, 30.0):
        sn, ss, two_theta, mode = evaluate_stress_state(*critical_state(c), CFI, T, 'values')
        assert np.isclose(two_theta, 90 + phi), c
    assert evaluate_stress_state(*critical_state(10.0), CFI, T, 'values')[3] == 'Mode II - compression'


def test_failure_band():
    tol = FAILURE_TOLERANCE
    for c in (2.2, 4.0, 10.0, 30.0):
        below = evaluate_stress_state(*critical_state(c, factor=1 - tol - 1e-3), CFI, T, 'values')[3]
        lower = evaluate_stress_state(*critical_state(c, factor=1 - tol + 1e-3), CFI, T, 'values')[3]
        upper = evaluate_stress_state(*critical_state(c, factor=1 + tol - 1e-3), CFI, T, 'values')[3]
        above = evaluate_stress_state(*critical_state(c, factor=1 + tol + 1e-3), CFI, T, 'values')[3]
        assert below == 'Below' and above == 'Above', c
        assert lower not in ('Below', 'Above'), c
        # Near the vertex a circle slightly beyond r_c reaches s < T: always 'Above'
        p, s = critical_state(c, factor=1 + tol - 1e-3)
        assert upper == ('Above' if s < T else upper) and (s < T or upper not in ('Below', 'Above')), c
    # Within the band the stresses are those of the circle, at the angle of the contact
    p, s = critical_state(10.0, factor=1.02)
    sn, ss, two_theta, mode = evaluate_stress_state(p, s, CFI, T, 'values')
    c, r = (p + s)/2e6, (p - s)/2e6
    assert np.isclose(sn, c + r*np.cos(np.radians(two_theta))) and np.isclose(ss, r*np.sin(np.radians(two_theta)))


def test_states_that_broke_the_sampling_algorithm():
    # A sample on sigma_n = 0 made it crash (s = 0, or linspace through 0)
    for p, s in ((8e6, 0.0), (0.9e6, -1.2e6), (3e6, 0.0)):
        evaluate_stress_state(p, s, CFI, T, 'values')
    # Circles with r < 1 MPa had no samples and were always 'Below', even in
    # biaxial tension beyond T
    assert evaluate_stress_state(-2.3e6, -4e6, CFI, T, 'values')[3] == 'Above'
    assert evaluate_stress_state_sampling(-2.3e6, -4e6, CFI, T)[3] == 'Below'   # the old behaviour
    # Isotropic and singular states, as before
    assert evaluate_stress_state(5e6, 5e6, CFI, T, 'values')[3] == 'Below'
    assert evaluate_stress_state(-2e6, -2e6, CFI, T, 'values')[3] == 'Above'
    assert evaluate_stress_state(np.nan, 1e6, CFI, T, 'values')[3] == 'Singular'


def test_agrees_with_the_sampling_algorithm_on_the_coulomb_line():
    """Where both criteria find Mode II failure away from the kink at sigma_n = 0,
    the stresses on the plane agree within the resolution of the sampling algorithm:
    it averaged the samples found within its band, which spans ~2% of the radius,
    so the differences grow with the circle (0.11 MPa and 0.8 deg for r = 8 MPa)."""
    checked = 0
    for c in np.arange(4, 30, 1.7):
        for factor in (0.98, 1.0, 1.02):
            p, s = critical_state(c, factor=factor)
            new = evaluate_stress_state(p, s, CFI, T, 'values')
            old = evaluate_stress_state_sampling(p, s, CFI, T)
            if new[3] == old[3] and new[3].startswith('Mode II'):
                r = (p - s)/2e6
                assert abs(new[0] - old[0]) < 0.02*r and abs(new[1] - old[1]) < 0.02*r, (c, factor, new, old)
                assert abs(new[2] - old[2]) < 1.0, (c, factor, new, old)
                checked += 1
    assert checked >= 30


def test_dynamic_model_records_nodes_that_jump_the_band():
    ci = [2, 0, -1000, 0, -5000]
    model_inputs = [30e9, 0.25, 2900, CFI, T]
    args = (ci, model_inputs, 0, 0, 0, 0, 0, 3.71, True)
    grid = ([-3000, 3000], [-3000, -10], [250, 250])
    # Two large steps: many nodes go from below to above failure in one step
    coarse = compute_dynamic_model(*args, [0.5, 1.0], *grid)
    last = coarse[-1]
    jumped = last['failure_jumped'] == 1
    assert jumped.any()
    # They have a failure mode, stresses and (except Mode I) a plane, not 'Above'
    modes = last['failure_mode'][jumped]
    assert not np.isin(modes, ('Below', 'Above', 'Singular')).any()
    assert np.isfinite(last['failure_sn'][jumped]).all()
    assert coarse.metadata['failure_criterion'] == 'analytical'
    assert coarse.metadata['failure_tolerance'] == FAILURE_TOLERANCE
    # With small steps fewer nodes jump
    fine = compute_dynamic_model(*args, [i/16 for i in range(1, 17)], *grid)
    assert (fine[-1]['failure_jumped'] == 1).sum() < jumped.sum()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('OK  ', test.__name__)
    print(len(tests), 'tests passed')
