"""
Tests of the principal stress orientations, the failure planes and the sign of T.

Run with pytest, or directly: python tests/test_orientations_and_failure.py
See docs/crack_coordinates_and_rotation.md, Section 10.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from DiES_core import model, crack, InputError
from Failure_Determination import evaluate_stress_state, failure_plane_orientation
from Helper_functions import G

warnings.filterwarnings('ignore')

E, NU, RHO, CFI, T = 30e9, 0.25, 2900, 0.57, -1.5e6
# Earth gravity for the tests since 25 Nov 2026 (Sam's convention); Mars (3.71) until then
G_EARTH = 9.81
CASES = {
    'vertical dike': ((0, -1000), (0, -5000), 2, 0),          # aperture 2 m
    'fault dipping 60 towards +x': ((-500, -1000), (500, -2732.0508), 0, 15e6),
    'fault dipping 60 towards -x': ((500, -1000), (-500, -2732.0508), 0, 15e6),
}


def setup(up, lo, aperture, dy, cell=(200, 150)):
    b = np.hypot(up[0] - lo[0], up[1] - lo[1])
    px = aperture*G(E, NU)/(b*(1 - NU)) if aperture else 0
    disc = crack(up, lo, px, dy, 0)
    m = model((-5000, 5000), (-6000, -10), cell[0], cell[1], 500, E, NU, RHO, CFI, T, G_EARTH)
    S = m.model_stresses(disc, 0, 0, True)
    sxx, syy, sxy = -S[0], -S[1], -S[3]                     # compression positive
    tensor = np.stack([np.stack([sxx, sxy], -1), np.stack([sxy, syy], -1)], -2)
    return disc, m, tensor


def axial_misfit(a, b):
    return np.degrees(np.arccos(np.clip(np.abs((a*b).sum(-1)), 0, 1)))


def test_principal_orientations_match_eigenvectors():
    for name, (up, lo, aperture, dy) in CASES.items():
        disc, m, tensor = setup(up, lo, aperture, dy)
        s1, s3, s1a, s3a = m.principal_stresses(disc, 0, 0, True)
        w, v = np.linalg.eigh(tensor)
        d1 = np.stack([np.cos(np.deg2rad(s1a)), np.sin(np.deg2rad(s1a))], -1)
        d3 = np.stack([np.cos(np.deg2rad(s3a)), np.sin(np.deg2rad(s3a))], -1)
        assert np.nanmax(axial_misfit(d1, v[..., :, 1])) < 1e-4, name
        assert np.nanmax(axial_misfit(d3, v[..., :, 0])) < 1e-4, name
        assert np.all((s1a[np.isfinite(s1a)] > -90) & (s1a[np.isfinite(s1a)] <= 90)), name


def test_lithostatic_sigma1_is_vertical():
    # No crack loading and no remote stresses: sigma1 must be vertical (90) everywhere
    disc = crack((0, -1000), (0, -5000), 0, 0, 0)
    m = model((-5000, 5000), (-6000, -10), 500, 500, 500, E, NU, RHO, CFI, T, G_EARTH)
    s1, s3, s1a, s3a = m.principal_stresses(disc, 0, 0, True)
    assert np.allclose(s1a, 90) and np.allclose(s3a, 0)


def test_failure_plane_orientation_rules():
    # sigma1 vertical, theta = 60 (plane at 30 from sigma1): left 120 (dips to +x), right 60 (dips to -x)
    assert np.isclose(np.degrees(failure_plane_orientation(90, 60, 'Mode II - compression', -1)) % 180, 120)
    assert np.isclose(np.degrees(failure_plane_orientation(90, 60, 'Mode II - compression', 1)) % 180, 60)
    # Mode I: the plane contains sigma1
    assert np.isclose(np.degrees(failure_plane_orientation(73, 90, 'Mode I', -1)), 73)
    # No plane for stable or past-failure states
    assert np.isnan(failure_plane_orientation(90, 60, 'Below', 1))
    assert np.isnan(failure_plane_orientation(90, float('nan'), 'Above', 1))


def test_failure_planes_resolve_to_the_mohr_tangent_point():
    for name, (up, lo, aperture, dy) in CASES.items():
        disc, m, tensor = setup(up, lo, aperture, dy, cell=(250, 200))
        s1, s3, s1a, s3a = m.principal_stresses(disc, 0, 0, True)
        x = m.grid()[0] ; c = disc.c() ; checked = 0
        for i, j in zip(*np.nonzero(np.isfinite(s1))):
            sn, ss, two_theta, mode = evaluate_stress_state(s1[i, j], s3[i, j], CFI, T, 'values')
            if mode in ('Below', 'Above', 'Mode I'):
                continue
            side = -1 if x[i, j] <= c[0] else 1
            b = failure_plane_orientation(s1a[i, j], two_theta/2, mode, side)
            normal = np.array([-np.sin(b), np.cos(b)])
            traction = tensor[i, j] @ normal
            sn_plane = normal @ traction
            ss_plane = np.linalg.norm(traction - sn_plane*normal)
            assert abs(sn_plane/1e6 - sn) < 0.02 and abs(ss_plane/1e6 - ss) < 0.02, (name, i, j, mode)
            # Of the two conjugates, the chosen one is the closest to dipping towards the crack
            other = np.deg2rad(2*s1a[i, j]) - b
            assert np.sin(2*b)*side >= np.sin(2*other)*side - 1e-12, (name, i, j)
            checked += 1
        assert checked > 0, name


def test_tensile_strength_sign():
    # Stable lithostatic states must be below failure with T < 0.
    # With v = 0.25 the lithostatic state has s3 = s1/3, so r = c/2 while the critical
    # radius on the Coulomb line tends to cfi*c/sqrt(1 + cfi^2) = 0.495c: r/r_c tends to
    # ~1.01 with depth. With Earth gravity it reaches the lower edge of the failure band
    # (0.965) at ~6.0 km (on Mars, at ~16 km), so depths up to 5 km are checked.
    for depth in (300, 1000, 3000, 5000):
        s1 = RHO*G_EARTH*depth ; s3 = s1*NU/(1 - NU)
        assert evaluate_stress_state(s1, s3, CFI, T, 'values')[3] == 'Below', depth
    # Tension beyond |T| fails; tension below |T| with a small circle does not
    assert evaluate_stress_state(5e6, -1.6e6, CFI, T, 'values')[3] == 'Above'
    assert evaluate_stress_state(1e6, -0.5e6, CFI, T, 'values')[3] == 'Below'
    # T must be negative
    try:
        evaluate_stress_state(5e6, 1e6, CFI, 1.5e6, 'values')
    except InputError:
        return
    raise AssertionError('no InputError for T > 0')



def test_arrays_match_scalar_calls():
    """evaluate_stress_state and failure_plane_orientation with arrays (Phase 2, point 3A)
    give, node by node, the same values as the scalar calls, bit by bit."""
    rng = np.random.default_rng(0)
    # States from tension to compression, with p >= s; some isotropic, some singular
    a, b = rng.uniform(-3e6, 25e6, (2, 12, 9))
    p, s = np.maximum(a, b), np.minimum(a, b)
    p[0, 0] = s[0, 0] = 4e6
    p[0, 1] = np.nan
    # cfi and T per node, as in a grid with layers
    cfi = np.where(np.arange(12)[:, None] < 6, 0.57, 0.6) * np.ones((12, 9))
    t = np.where(np.arange(12)[:, None] < 6, -1.5e6, -2e6) * np.ones((12, 9))
    sn, ss, two_theta, mode = evaluate_stress_state(p, s, cfi, t, 'values')
    assert sn.shape == ss.shape == two_theta.shape == mode.shape == p.shape
    for idx in np.ndindex(p.shape):
        ref = evaluate_stress_state(p[idx], s[idx], cfi[idx], t[idx], 'values')
        assert mode[idx] == ref[3], idx
        for value, expected in zip((sn[idx], ss[idx], two_theta[idx]), ref[:3]):
            assert np.array_equal(value, expected, equal_nan=True), idx
    # Failure planes: every mode, both sides, ties between the two conjugates (s1a = 45)
    modes = ['Mode I', 'Mixed-mode I-II', 'Mode II - tension-compression',
             'Mode II - compression', 'Below', 'Above', 'Singular']
    s1a, theta, m, side = np.meshgrid(np.arange(-90, 90.1, 7.5), [45.0, 60.0, 72.5, 90.0, np.nan],
                                      modes, [-1, 1], indexing='ij')
    planes = failure_plane_orientation(s1a, theta, m, side)
    for idx in np.ndindex(s1a.shape):
        expected = failure_plane_orientation(s1a[idx], theta[idx], m[idx], side[idx])
        assert np.array_equal(planes[idx], expected, equal_nan=True), (idx, planes[idx], expected)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('OK  ', test.__name__)
    print(len(tests), 'tests passed')
