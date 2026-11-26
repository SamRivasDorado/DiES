"""
Tests of plotting.py and of the model.plot_* wrappers (Phase 1, 2026-11-24).

The figures drawn from DiESResults must be identical, pixel by pixel, to those
of the wrappers, and the wrappers must compute the model only once.
Run with pytest, or directly: python tests/test_plotting.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import plotting
from DiES_core import model, crack, InputError
from Helper_functions import G

warnings.filterwarnings('ignore')

E, NU, RHO, FRICTION, T = 30e9, 0.25, 2900, 0.57, -1.5e6


def small_case():
    m = model((-2000, 2000), (-3000, -10), 250, 250, 500, E, NU, RHO, FRICTION, T, 9.81)
    disc = crack((0, -500), (0, -2000), 2*G(E, NU)/(1500*(1 - NU)), 1e6, 0)
    return m, disc


def pixels(fig):
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return image


def test_wrappers_and_plotting_draw_the_same_figures():
    m, disc = small_case()
    res = m.compute(disc, 0, 0, True)
    pairs = [
        (m.plot_displacement(disc, 'uxy', 'vectors', 0.25), plotting.plot_displacement(res, 'uxy', 'vectors', 0.25)),
        (m.plot_displacement(disc, 'ux', 'contours', 0.1), plotting.plot_displacement(res, 'ux', 'contours', 0.1)),
        (m.plot_stresses(disc, 'sxya', 0, 0, True, 1), plotting.plot_stresses(res, 'sxya', 1)),
        (m.plot_stresses(disc, 'ta', 0, 0, True, 1), plotting.plot_stresses(res, 'tmax', 1)),
        (m.plot_principal_stresses(disc, 'angle', 0, 0, True, 10), plotting.plot_principal_stresses(res, 'angle', 10)),
        (m.plot_stress_trajectories(disc, True, 0, 0, True), plotting.plot_stress_trajectories(res, True)),
        (m.plot_plane_stresses(disc, 'shear', 0, 0, 60, True, 0.5), plotting.plot_plane_stresses(res, 'shear', 60, 0.5)),
        (disc.plot_crack_displacements(NU, E), plotting.plot_crack_displacements(disc, NU, E)),
        # Former names, kept for existing scripts
        (disc.plot_mode_I_crack_displacements(NU, E), plotting.plot_crack_opening(disc, NU, E)),
    ]
    for k, (a, b) in enumerate(pairs):
        assert np.array_equal(pixels(a), pixels(b)), k


def test_wrappers_compute_once():
    m, disc = small_case()
    calls = {'n': 0}
    original = m.model_stresses

    def counting(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)

    m.model_stresses = counting
    plt.close(m.plot_principal_stresses(disc, 'principal', 0, 0, True, 5))
    assert calls['n'] == 1, calls                 # it used to be 2 or 3
    calls['n'] = 0
    plt.close(m.plot_plane_stresses(disc, 'normal', 0, 0, 60, True, 5))
    assert calls['n'] == 1, calls


def test_invalid_requests():
    m, disc = small_case()
    res = m.compute(disc, 0, 0, True)
    for call in (lambda: plotting.plot_stresses(res, 'bad', 1),
                 lambda: plotting.plot_displacement(res, 'uz', 'vectors', 1),
                 lambda: plotting.plot_displacement(res, 'ux', 'arrows', 1),
                 lambda: plotting.plot_principal_stresses(res, 'tertiary', 1),
                 lambda: plotting.plot_plane_stresses(res, 'bad', 60, 1),
                 lambda: plotting.plot_stress_trajectories(res, 'True')):
        try:
            call()
        except plotting.PlotError:
            continue
        raise AssertionError('no PlotError')
    # Missing fields: displacements only
    res_u = m.compute(disc, displacements=True, stresses=False)
    try:
        plotting.plot_stresses(res_u, 'sxx', 1)
    except plotting.PlotError:
        pass
    else:
        raise AssertionError('missing field accepted')
    # The wrappers reject invalid keys before computing
    for call in (lambda: m.plot_stresses(disc, 'bad', 0, 0, True, 1),
                 lambda: m.plot_principal_stresses(disc, 'bad', 0, 0, True, 1),
                 lambda: m.plot_plane_stresses(disc, 'bad', 0, 0, 60, True, 1),
                 lambda: m.plot_stress_trajectories(disc, 'True', 0, 0, True)):
        try:
            call()
        except InputError:
            continue
        raise AssertionError('no InputError')
    plt.close('all')


def test_figures_from_saved_results():
    # Results saved to disk and reopened draw the same figure
    import tempfile
    from results import DiESResults
    m, disc = small_case()
    res = m.compute(disc, 0, 0, True)
    path = os.path.join(tempfile.mkdtemp(), 'r.npz')
    res.save(path)
    back = DiESResults.load(path)
    assert np.array_equal(pixels(plotting.plot_plane_stresses(res, 'normal', 60, 5)),
                          pixels(plotting.plot_plane_stresses(back, 'normal', 60, 5)))



def test_mohr_circle():
    from Failure_Determination import evaluate_stress_state
    T = -1.5e6
    # Below, Above, Mode I, Mixed-mode I-II, Mode II tension-compression, Mode II compression
    for p, s in ((5.4e6, 1.8e6), (5e6, -1.6e6), (4e6, -1.48e6), (5.71e6, -1.45e6), (9.68e6, -0.2e6), (13.5e6, 1.2e6)):
        assert np.array_equal(pixels(evaluate_stress_state(p, s, FRICTION, T, 'plot')),
                              pixels(plotting.plot_mohr_circle(p, s, FRICTION, T))), (p, s)
    for call in (lambda: evaluate_stress_state(5e6, 1e6, FRICTION, T, 'figure'),
                 lambda: evaluate_stress_state(5e6, 5e6, FRICTION, T, 'plot')):
        try:
            call()
        except InputError:
            continue
        raise AssertionError('no InputError')
    try:
        plotting.plot_mohr_circle(float('nan'), 1e6, FRICTION, T)
    except plotting.PlotError:
        pass
    else:
        raise AssertionError('no PlotError for a singular state')



def test_wall_displacements_match_the_analytical_profiles():
    """crack.wall_displacements (core solution off each wall, Nov 2026) against Eq. 7.21
    of Pollard and Martel (mode I) and the elliptical slip of mode II, vertical and inclined."""
    E, nu = 30e9, 0.25
    dike = crack((0, -1000), (0, -5000), 8e6, 0, 0)
    s, walls = dike.wall_displacements(nu, E)
    ux_pos, ux_neg, uy_pos, uy_neg = dike.mode_I_crack_displacements(nu, E)
    peak = ux_pos.max()
    assert np.max(np.abs(walls['positive']['normal'] - ux_pos)) < 1e-3*peak
    assert np.max(np.abs(walls['negative']['normal'] - ux_neg)) < 1e-3*peak
    # Eq. 7.21 measures along the crack in the same sense as s
    assert np.max(np.abs(walls['positive']['tangential'] - uy_pos)) < 1e-3*peak
    for lower in ((0, -5000), (500, -1000 - 1000*np.tan(np.radians(60)))):
        fault = crack((0 if lower[0] == 0 else -500, -1000), lower, 0, 15e6, 0)
        s, walls = fault.wall_displacements(nu, E)
        half = fault.extent()/2
        slip = (1 - nu)*15e6*np.sqrt(half**2 - s**2)/G(E, nu)
        assert np.max(np.abs(np.abs(walls['positive']['tangential']) - slip)) < 1e-3*slip.max()
        # Opposite walls slide in opposite senses; the normal component is the same on both
        assert np.all(walls['positive']['tangential'][1:-1]*walls['negative']['tangential'][1:-1] < 0)
        assert np.allclose(walls['positive']['normal'], walls['negative']['normal'], atol=1e-3*slip.max())
    # The figure of a fault is no longer flat
    fig = plotting.plot_crack_displacements(fault, nu, E)
    assert any(np.nanmax(np.abs(np.asarray(c.get_offsets())[:, 1])) > 0.5 for c in fig.axes[0].collections)
    plt.close(fig)


def test_automatic_contour_steps_and_constant_fields():
    from Helper_functions import nice_step, auto_contour_step, boundaries_contours
    assert nice_step(150, 12) == 20 and nice_step(37, 12) == 5 and nice_step(0.6, 12) == 0.05
    assert nice_step(0, 12) is None
    values = np.linspace(12, 148, 100)
    p1, p99, levels = boundaries_contours(values)        # step None: automatic
    assert auto_contour_step(values) == 20 and np.allclose(levels % 20, 0) and 3 <= len(levels) <= 12
    assert np.allclose(boundaries_contours(values, 25)[2], [25, 50, 75, 100, 125])   # a step can still be forced
    # A field that is zero everywhere gets a note instead of an empty colour scale
    m, disc = small_case()
    res = m.compute(disc, 0, 0, False)
    fig = plotting.plot_stresses(res, 'sxza')
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any('everywhere' in t for t in texts), texts
    plt.close(fig)
    # The sigma1 angle uses a cyclic colour scale from -90 to 90
    fig = plotting.plot_principal_stresses(res, 'angle')
    mesh = [c for c in fig.axes[0].collections if hasattr(c, 'get_cmap')][0]
    assert mesh.get_cmap().name == 'twilight' and mesh.get_clim() == (-90, 90)
    plt.close(fig)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('OK  ', test.__name__)
    print(len(tests), 'tests passed')
