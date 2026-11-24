"""
Tests of the model class (grid, layers, lithostatic stresses) and of the helper
functions linear, power_law and scale_arrows.

Ported on 2026-11-24 from the 84-check verification script of the Phase 0
session (2026-11-23). Run with pytest, or directly:
python tests/test_model_and_helpers.py
"""

import itertools
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from DiES_core import model, InputError
from Helper_functions import linear, power_law, scale_arrows

warnings.filterwarnings('ignore')

G_MARS = 3.71
LAYERS = dict(ypos=(-1100., -600., -300., 0.), E=[10e9, 20e9, 30e9], v=[0.25, 0.3, 0.35],
              d=[2500, 2600, 2700], cfi=[0.6, 0.7, 0.8], T=[-1e6, -2e6, -3e6])


def three_layer_model():
    L = LAYERS
    return model((0, 1000), (-1000, 0), 100, 100, L['ypos'], L['E'], L['v'], L['d'], L['cfi'], L['T'], G_MARS)


def expected_layer(y):
    # Layer bases at -1100, -600, -300; a node at a base belongs to that layer
    return np.where(y < -600, 0, np.where(y < -300, 1, 2))


def test_multilayer_grid():
    m = three_layer_model()
    grid = m.grid()
    assert grid.shape == (9, 10, 10)
    y = grid[1][:, 0]
    layer = expected_layer(y)
    L = LAYERS
    expected = {2: L['E'], 3: L['v'], 4: L['d'], 5: 2*np.abs(L['T']), 6: L['cfi'],
                7: 0.5*np.arctan(1/np.array(L['cfi'])), 8: L['T']}
    for row, values in expected.items():
        assert np.allclose(grid[row][:, 0], np.asarray(values)[layer]), row
        assert np.all(grid[row] == grid[row][:, [0]]), row        # constant along each row
    assert m.n_layers() == 3


def test_multilayer_lithostatic_stresses():
    m = three_layer_model()
    syy, sxx = m.stresses_at_depth()
    y = m.grid()[1][:, 0]

    def litho(yv):
        depth = -yv
        if yv >= -300:
            return 2700*G_MARS*depth
        if yv >= -600:
            return 2700*G_MARS*300 + 2600*G_MARS*(depth - 300)
        return 2700*G_MARS*300 + 2600*G_MARS*300 + 2500*G_MARS*(depth - 600)

    expected = np.array([litho(v) for v in y])
    nu = np.array(LAYERS['v'])[expected_layer(y)]
    assert np.allclose(-syy[:, 0], expected)
    assert np.allclose(-sxx[:, 0], expected*nu/(1 - nu))
    assert np.all(np.diff(-syy[:, 0]) < 0)                          # continuous, increasing with depth


def test_single_layer_lithostatic_and_tuple_input():
    m1 = model((0, 1000), (-1000, 0), 100, 100, 0.0, 10e9, 0.25, 2500, 0.6, -1e6, G_MARS)
    m2 = model((0, 1000), (-1000, 0), 100, 100, (0.0,), (10e9,), (0.25,), (2500,), (0.6,), (-1e6,), G_MARS)
    assert np.array_equal(m1.grid(), m2.grid())
    syy, sxx = m1.stresses_at_depth()
    y = m1.grid()[1]
    assert np.allclose(-syy, 2500*G_MARS*np.abs(y))
    assert np.allclose(sxx, syy*0.25/0.75)
    assert all(np.array_equal(a, b) for a, b in zip(m1.stresses_at_depth(), m2.stresses_at_depth()))
    assert np.allclose(m1.grid()[5], 2e6)                           # implicit cohesion 2|T|


def test_invalid_layers_raise():
    for ypos, n in (((-1100., -600., 0.), 3), ((-300., -600., 0.), 2)):
        try:
            model((0, 1000), (-1000, 0), 100, 100, ypos, [1e9]*n, [0.25]*n, [2500]*n, [0.6]*n, [-1e6]*n,
                  G_MARS).grid()
        except InputError:
            continue
        raise AssertionError('no InputError for ' + str(ypos))


def test_linear_and_power_law():
    assert np.array_equal(linear([1, 2, 3], 2, 1), [3, 5, 7])
    assert np.array_equal(linear((1., 2.), 2, 1), [3., 5.])
    assert linear(2, 2, 1) == 5 and linear(np.float64(2), 2, 1) == 5
    assert np.array_equal(linear(np.array([1, 2]), 2, 1), [3, 5])
    assert np.array_equal(power_law([1, 2], 2, 2), [2, 8])
    assert np.array_equal(power_law((1., 3.), 2, 2), [2., 18.])
    assert power_law(3, 2, 2) == 18 and power_law(np.float64(3), 2, 2) == 18
    assert np.array_equal(power_law(np.array([1, 2]), 2, 2), [2, 8])


def test_scale_arrows():
    dx = np.array([[1., -3.], [2., 0.5]]) ; dy = np.array([[4., -1.], [0.2, 6.]])
    mx, my = np.amax(abs(dx)), np.amax(abs(dy))
    for dpx, dpy in itertools.product([-3, -1, 0, 1, 3, -2.5e6, 4e6], repeat=2):
        if dpx > dpy:
            expected = (mx, 0.05)
        elif dpx < dpy:
            expected = (my, 0.025) if dpy <= 0 else ((my, 0.05) if dpx > 0 else (mx, 0.05))
        else:
            expected = (mx, 0.04)
        assert scale_arrows(dpx, dpy, dx, dy) == expected, (dpx, dpy)
    try:
        scale_arrows(float('nan'), 1, dx, dy)
    except ValueError:
        return
    raise AssertionError('no ValueError for NaN')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('OK  ', test.__name__)
    print(len(tests), 'tests passed')
