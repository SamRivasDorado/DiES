"""
Tests of the drawing options of plotting.py and of the failure state of a single model
(Phase 4, Nov 2026): the options change what they say and nothing else, their defaults
give the figures of before (also checked for 129 figures by tests/compare_figures.py),
invalid options are rejected, and compute_failure_state agrees with the failure criterion.
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import plotting
from plotting import PlotError
from DiES_core import model, crack, InputError
from Helper_functions import G
from Modeling_functions import compute_failure_state
from Failure_Determination import evaluate_stress_states, failure_contact, FAILURE_TOLERANCE

warnings.filterwarnings('ignore')

E, NU, RHO, FRICTION, T = 30e9, 0.25, 2900, 0.57, -1.5e6


@pytest.fixture(scope='module')
def res():
    # Vertical dike with 2 m of aperture and some shear, on a coarse grid, Earth gravity
    m = model((-2000, 2000), (-3000, -10), 250, 250, 500, E, NU, RHO, FRICTION, T, 9.81)
    disc = crack((0, -500), (0, -2000), 2*G(E, NU)/(1500*(1 - NU)), 1e6, 0)
    return m.compute(disc, 0, 0, True)


def pixels(fig):
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return image


def mesh(fig):
    # The colour map of a figure (its first QuadMesh)
    from matplotlib.collections import QuadMesh
    return [c for c in fig.axes[0].collections if isinstance(c, QuadMesh)][0]


def test_explicit_defaults_give_the_same_figures(res):
    """Passing every option with its default value draws the figure of before, pixel by pixel."""
    scalar = dict(cmap=None, limits=None, center_zero=False, contours=True, contour_labels=True)
    pairs = [
        (plotting.plot_stresses(res, 'sxx'), plotting.plot_stresses(res, 'sxx', **scalar)),
        (plotting.plot_principal_stresses(res, 'principal'), plotting.plot_principal_stresses(res, 'principal', **scalar)),
        (plotting.plot_principal_stresses(res, 'angle'), plotting.plot_principal_stresses(res, 'angle', cmap=None)),
        (plotting.plot_displacement(res, 'uxy', 'vectors'),
         plotting.plot_displacement(res, 'uxy', 'vectors', info_box=True, arrow_scale=1.0, **scalar)),
        (plotting.plot_displacement(res, 'ux', 'contours'),
         plotting.plot_displacement(res, 'ux', 'contours', info_box=True, **scalar)),
        (plotting.plot_stress_trajectories(res, True), plotting.plot_stress_trajectories(res, True, arrow_scale=1)),
        (plotting.plot_plane_stresses(res, 'normal', 60), plotting.plot_plane_stresses(res, 'normal', 60, info_box=True,
                                                                                       **scalar)),
    ]
    for k, (a, b) in enumerate(pairs):
        assert np.array_equal(pixels(a), pixels(b)), k


def test_colour_options(res):
    """cmap, center_zero and limits change the colour map as asked."""
    fig = plotting.plot_stresses(res, 'sxx', cmap='RdBu_r', center_zero=True)
    quad = mesh(fig)
    vmin, vmax = quad.get_clim()
    assert quad.get_cmap().name == 'RdBu_r' and vmin == -vmax
    plt.close(fig)
    fig = plotting.plot_stresses(res, 'sxx', limits=(-10, 5), center_zero=True)    # limits win
    assert mesh(fig).get_clim() == (-10, 5)
    plt.close(fig)
    fig = plotting.plot_principal_stresses(res, 'angle', cmap='hsv', limits=(0, 1))  # angle: only cmap
    assert mesh(fig).get_cmap().name == 'hsv' and mesh(fig).get_clim() == (-90, 90)
    plt.close(fig)
    fig = plotting.plot_displacement(res, 'uxy', 'vectors', cmap='plasma', limits=(0, 0.5))
    quiver = fig.axes[0].collections[-1]
    assert quiver.get_cmap().name == 'plasma' and quiver.get_clim() == (0, 0.5)
    plt.close(fig)


def test_contour_and_box_options(res):
    """contours, contour_labels and info_box remove what they say."""
    def texts(fig):
        return len(fig.axes[0].texts)
    full = plotting.plot_plane_stresses(res, 'normal', 60)
    no_labels = plotting.plot_plane_stresses(res, 'normal', 60, contour_labels=False)
    no_box = plotting.plot_plane_stresses(res, 'normal', 60, info_box=False)
    no_contours = plotting.plot_plane_stresses(res, 'normal', 60, contours=False, info_box=False)
    assert texts(full) > texts(no_labels) == 1          # only the box is left
    assert texts(no_box) == texts(full) - 1
    assert texts(no_contours) == 0
    assert len(no_contours.axes[0].collections) < len(full.axes[0].collections)
    for fig in (full, no_labels, no_box, no_contours):
        plt.close(fig)
    fig = plotting.plot_displacement(res, 'ux', 'contours', info_box=False)
    assert all('sigma' not in t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)


def test_arrow_scale(res):
    """arrow_scale > 1 draws longer arrows and segments (smaller quiver scale)."""
    a = plotting.plot_displacement(res, 'uxy', 'vectors')
    b = plotting.plot_displacement(res, 'uxy', 'vectors', arrow_scale=2)
    assert b.axes[0].collections[-1].scale == pytest.approx(a.axes[0].collections[-1].scale/2)
    c = plotting.plot_stress_trajectories(res, False, arrow_scale=0.5)
    d = plotting.plot_stress_trajectories(res, False)
    assert c.axes[0].collections[0].scale == pytest.approx(d.axes[0].collections[0].scale*2)
    for fig in (a, b, c, d):
        plt.close(fig)


@pytest.mark.parametrize('call, message', [
    (lambda r: plotting.plot_stresses(r, 'sxx', colour='red'), 'Unknown drawing option'),
    (lambda r: plotting.plot_stresses(r, 'sxx', info_box=False), 'Unknown drawing option'),
    (lambda r: plotting.plot_stresses(r, 'sxx', cmap='not_a_map'), 'Unknown colour map'),
    (lambda r: plotting.plot_stresses(r, 'sxx', limits=(5, 1)), 'vmin < vmax'),
    (lambda r: plotting.plot_stresses(r, 'sxx', limits='wide'), 'two numbers'),
    (lambda r: plotting.plot_stresses(r, 'sxx', contours='yes'), 'True or False'),
    (lambda r: plotting.plot_stress_trajectories(r, True, arrow_scale=0), 'positive number'),
])
def test_invalid_options(res, call, message):
    with pytest.raises(PlotError, match=message):
        call(res)
    plt.close('all')


def test_failure_state_agrees_with_the_criterion(res):
    """compute_failure_state: the modes of evaluate_stress_states and r/r_c of failure_contact."""
    state = compute_failure_state(res)
    sn, ss, two_theta, mode = evaluate_stress_states(res.s1, res.s3, res.friction, res.T)
    ratio, _, _, _, contact = failure_contact(res.s1, res.s3, res.friction, res.T)
    finite = np.isfinite(res.s1) & np.isfinite(res.s3)
    assert np.array_equal(state['failure_mode'], mode)
    np.testing.assert_array_equal(state['failure_ratio'][finite], ratio[finite])
    np.testing.assert_array_equal(state['failure_theta'], two_theta/2)
    assert np.array_equal(state['failure_contact_mode'][finite], contact[finite])
    # The classification follows the ratio: within the band, at failure; beyond it, above
    at = np.abs(state['failure_ratio'] - 1) <= FAILURE_TOLERANCE
    assert np.all(np.isin(state['failure_mode'][at & finite & (res.s3 >= res.T)],
                          ['Mode I', 'Mixed-mode I-II', 'Mode II - tension-compression', 'Mode II - compression']))
    assert np.all(state['failure_mode'][state['failure_ratio'] > 1 + FAILURE_TOLERANCE] == 'Above')
    assert state.metadata['failure_tolerance'] == FAILURE_TOLERANCE
    assert state.metadata['crack'] == res.metadata['crack']


def test_failure_state_needs_the_stresses(res):
    from results import DiESResults
    with pytest.raises(InputError, match='s1'):
        compute_failure_state(DiESResults(res.x, res.y, {'ux': res.ux}, res.metadata))


def test_failure_figures(res):
    """The failure-mode map lists the states found with their number of nodes; the ratio map
    has the band edges; both draw the crack."""
    state = compute_failure_state(res)
    fig = plotting.plot_failure_modes(state)
    legend = fig.axes[0].get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    # Under the map, without 'Below', each state with its share of the nodes that fail
    modes, counts = np.unique(state['failure_mode'], return_counts=True)
    failing = sum(c for m, c in zip(modes, counts) if m not in ('Below', 'Singular'))
    expected = ['%s (%.1f %%)' % (m, 100.0*c/failing) for m, c in zip(modes, counts) if m not in ('Below', 'Singular')]
    assert sorted(labels) == sorted(expected)
    assert legend.get_title().get_text() == 'Share of the nodes that fail'
    plt.close(fig)
    fig = plotting.plot_failure_modes(state, legend=False)
    assert fig.axes[0].get_legend() is None
    plt.close(fig)
    fig = plotting.plot_failure_ratio(state, cmap='coolwarm', limits=(0, 3))
    assert mesh(fig).get_clim() == (0, 3) and mesh(fig).get_cmap().name == 'coolwarm'
    plt.close(fig)
    # The failure mode of a dynamic step is drawn the same way
    from results import DiESResults
    step = DiESResults(res.x, res.y, {'failure_mode': np.where(res.y > -1000, 'Above', 'Below')}, res.metadata)
    fig = plotting.plot_failure_modes(step)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ['Above (100.0 %)']
    plt.close(fig)
    # Nothing fails: a note instead of the legend
    calm = DiESResults(res.x, res.y, {'failure_mode': np.full(res.x.shape, 'Below')}, res.metadata)
    fig = plotting.plot_failure_modes(calm)
    assert fig.axes[0].get_legend() is None and fig.axes[0].texts[-1].get_text() == 'No node fails'
    plt.close(fig)


def test_failure_mode_shares():
    """Shares relative to the nodes that fail: 'Below' and 'Singular' are counted but get no share."""
    modes = ['Below']*6 + ['Above']*1 + ['Mode I']*3 + ['Singular']*2
    assert plotting.failure_mode_shares(modes) == [('Below', 6, None), ('Above', 1, 25.0), ('Mode I', 3, 75.0),
                                                   ('Singular', 2, None)]
    assert plotting.failure_mode_shares(['Below', 'Below']) == [('Below', 2, None)]


# ------------------------------------------------------------- dynamic figures (Phase 4, step 3)

def _dynamic():
    import dies
    return dies.load_scenario(os.path.join(ROOT, 'examples', 'dike_example.yaml')).run().dynamic


def test_dynamic_loads_and_first_failure():
    dyn = _dynamic()
    loads = dyn.step_loads()
    assert len(loads) == len(dyn) and np.isclose(loads[-1], 8e6) and np.all(np.diff(loads) > 0)
    first = dyn.failure_step()
    load = dyn.failure_load()
    assert np.all(np.isnan(load[first == 0])) and np.allclose(load[first > 0], loads[first[first > 0] - 1])
    # Up to step 10: the later failures are left out
    upto = dyn.failure_load(10)
    assert np.all(np.isnan(upto[first > 10])) and np.allclose(upto[(first > 0) & (first <= 10)],
                                                               load[(first > 0) & (first <= 10)])


def test_dynamic_figures_take_their_options():
    dyn = _dynamic()
    color_range = plotting.dynamic_color_range(dyn)
    for figure in (plotting.plot_dynamic_step(dyn[10], 2, color_range, cmap='magma', above=False, traces=False,
                                              info_box=False, limits=(0, 50)),
                   plotting.plot_field(dyn[-1], 'failure_ss', factor=1.0, cmap='viridis'),
                   plotting.plot_first_failure(dyn, 12, 'percent', cmap='plasma'),
                   plotting.plot_first_failure(dyn, unit='step'),
                   plotting.plot_failure_evolution(dyn, current=5, share=True)):
        assert figure.axes
        plt.close(figure)
    with pytest.raises(plotting.PlotError):
        plotting.plot_first_failure(dyn, unit='km')
    with pytest.raises(plotting.PlotError):
        plotting.plot_dynamic_step(dyn[0], 1, color_range, above='yes')
