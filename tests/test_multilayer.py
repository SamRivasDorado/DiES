"""
Multilayer models in single_model, dynamic_model and compute_dynamic_model (Phase 2,
Nov 2026): model_inputs with one value per layer and layer_tops, from the top down.
"""

import os
import sys
import warnings

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from DiES_core import model, crack, InputError
from Helper_functions import G
from Modeling_functions import layered_inputs, single_model, compute_dynamic_model, dynamic_model

G_EARTH = 9.81
XLIMS, YLIMS, RES = [-3000, 3000], [-4000, -10], [250, 250]
DIKE = [2, 0, -500, 0, -2500]
ONE_LAYER = [30e9, 0.25, 2900, 0.57, -1.5e6]


def test_layered_inputs_converts_top_down_layers():
    bounds, E, v, d, cfi, T = layered_inputs([[20e9, 50e9], 0.25, [2500, 2900], 0.6, [-1e6, -3e6]], [0, -2000])
    # Ascending boundaries, values from the deepest layer up
    assert list(bounds[1:]) == [-2000, 0] and bounds[0] < -2000
    assert list(E) == [50e9, 20e9] and list(d) == [2900, 2500] and list(T) == [-3e6, -1e6]
    assert list(v) == [0.25, 0.25] and list(cfi) == [0.6, 0.6]      # one number: the same in every layer
    m = model(XLIMS, YLIMS, RES[0], RES[1], bounds, E, v, d, cfi, T, G_EARTH)
    grid = m.grid()
    assert np.all(grid[2][grid[1] >= -2000] == 20e9) and np.all(grid[2][grid[1] < -2000] == 50e9)
    # One layer: plain numbers, as until Nov 2026
    assert layered_inputs(ONE_LAYER) == (0, *ONE_LAYER)
    with pytest.raises(InputError):
        layered_inputs([[20e9, 50e9], 0.25, 2900, 0.6, -1e6])            # lists without layer_tops
    with pytest.raises(InputError):
        layered_inputs(ONE_LAYER, [-2000, 0])                            # tops not from the top down
    with pytest.raises(InputError):
        layered_inputs([[20e9, 50e9, 60e9], 0.25, 2900, 0.6, -1e6], [0, -2000])   # 3 values, 2 layers


def test_two_identical_layers_equal_one_layer():
    """Splitting a homogeneous model into two identical layers changes nothing."""
    one = compute_dynamic_model(DIKE, ONE_LAYER, 0, 0, 0, 0, 0, G_EARTH, True, [0.5, 1.0], XLIMS, YLIMS, RES)
    # The boundary at -1700 m crosses the dike: it warns, and with identical layers it does not matter
    with pytest.warns(UserWarning, match='crosses a layer boundary'):
        two = compute_dynamic_model(DIKE, ONE_LAYER, 0, 0, 0, 0, 0, G_EARTH, True, [0.5, 1.0], XLIMS, YLIMS, RES,
                                    layer_tops=[0, -1700])
    assert np.array_equal(one.final_mode(), two.final_mode())
    assert np.allclose(one[-1]['failure_sn'], two[-1]['failure_sn'], equal_nan=True, rtol=1e-12, atol=1e-9)


def test_aperture_uses_the_layer_at_the_crack_centre():
    E = [20e9, 50e9] ; nu = [0.25, 0.3]
    inputs = [E, nu, [2500, 2900], 0.6, [-1e6, -3e6]]
    b = 2000
    # Centre at y = -1500, in the upper layer (top 0, base -2000): its E and v. The lower tip
    # (-2500) is in the other layer, so it also warns
    with pytest.warns(UserWarning, match='crosses a layer boundary'):
        dyn = compute_dynamic_model(DIKE, inputs, 0, 0, 0, 0, 0, G_EARTH, True, [1.0], XLIMS, YLIMS, RES,
                                    layer_tops=[0, -2000])
    expected = DIKE[0]*G(E[0], nu[0])/(b*(1 - nu[0]))
    assert np.isclose(dyn[-1].metadata['dx'], expected)
    # Centre below a boundary at -1000 m: the lower layer's E and v
    with pytest.warns(UserWarning, match='crosses a layer boundary'):
        dyn = compute_dynamic_model(DIKE, inputs, 0, 0, 0, 0, 0, G_EARTH, True, [1.0], XLIMS, YLIMS, RES,
                                    layer_tops=[0, -1000])
    assert np.isclose(dyn[-1].metadata['dx'], DIKE[0]*G(E[1], nu[1])/(b*(1 - nu[1])))
    # A crack inside one layer: no warning
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compute_dynamic_model(DIKE, inputs, 0, 0, 0, 0, 0, G_EARTH, True, [1.0], XLIMS, YLIMS, RES,
                              layer_tops=[0, -3000])


def test_single_and_dynamic_model_figures_with_layers():
    inputs = [[20e9, 50e9], [0.25, 0.3], [2500, 2900], [0.57, 0.6], [-1.5e6, -3e6]]
    warnings.filterwarnings('ignore')
    for parameter in ('sxy', 's1', 'st', 'sn', 'ux'):
        fig = single_model(DIKE, inputs, 0, 0, 0, 0, 0, G_EARTH, True, 60, XLIMS, YLIMS, RES, parameter,
                           layer_tops=[0, -1800])
        plt.close(fig)
    modes = dynamic_model(DIKE, inputs, 0, 0, 0, 0, 0, G_EARTH, True, [0.5, 1.0], XLIMS, YLIMS, RES, 2, False, '',
                          layer_tops=[0, -1800])
    assert modes.shape == (len(np.arange(*YLIMS, RES[1])), len(np.arange(*XLIMS, RES[0])))
