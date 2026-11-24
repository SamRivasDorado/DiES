"""
Tests of single_model, dynamic_model and the robustness of evaluate_stress_state.

Rebuilt on 2026-11-24 from the description of the 37-check verification script
of the refactoring session (bool validation, aperture formula against the
displacement field, scaling of dx/dy/dz per step for a dike and the three fault
types, one PNG per step with plotting=True, single_model with litho True/False),
plus the Phase 0 changes of 2026-11-24 ('Singular' tips, isotropic states,
errors instead of sys.exit()). Run with pytest, or directly:
python tests/test_modeling_functions.py
"""

import glob
import os
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import Modeling_functions
from DiES_core import crack, displacements, InputError, require_bool
from Failure_Determination import evaluate_stress_state
from Helper_functions import G
from Modeling_functions import single_model, dynamic_model

warnings.filterwarnings('ignore')

E, NU, RHO, CFI, T = 30e9, 0.25, 2900, 0.57, -1.5e6
MODEL_INPUTS = [E, NU, RHO, CFI, T]
G_MARS = 3.71
# Small grid to keep the tests fast
XLIMS, YLIMS, RES = [-1500, 1500], [-2500, -10], [250, 250]
DIKE = [2, 0, -500, 0, -2000]                  # aperture 2 m, vertical
FAULT = [0, -300, -500, 300, -1539.2305]       # aperture 0, dipping 60 degrees
PARAMETERS = ['ux', 'uy', 'uxy', 'sx', 'sy', 'sxy', 'sav', 't', 's1', 's3', 'st', 'sn', 'ss']


def test_bools_must_be_real_bools():
    assert require_bool(True, 'x') is True and require_bool(False, 'x') is False
    for value in ('True', 'False', 1, None):
        try:
            require_bool(value, 'x')
        except InputError:
            continue
        raise AssertionError('no InputError for ' + repr(value))
    try:
        single_model(DIKE, MODEL_INPUTS, 0, 0, 0, 0, 0, G_MARS, 'True', 60, XLIMS, YLIMS, RES, 'sxy')
    except InputError:
        return
    raise AssertionError("single_model accepted litho='True'")


def test_model_inputs_need_five_values():
    try:
        single_model(DIKE, [E, NU, RHO, 3e6, CFI, T], 0, 0, 0, 0, 0, G_MARS, True, 60, XLIMS, YLIMS, RES, 'sxy')
    except InputError:
        return
    raise AssertionError('6 model_inputs accepted')


def test_aperture_formula_gives_the_total_aperture():
    a = 2.0 ; up, lo = (0, -500), (0, -2000) ; b = 1500
    dx = a*G(E, NU)/(b*(1 - NU))
    c = (0, -1250)
    opening = displacements(dx, 0, 0, NU, E, up, c, lo, (0.01, -1250))[0] - \
        displacements(dx, 0, 0, NU, E, up, c, lo, (-0.01, -1250))[0]
    assert np.isclose(opening, a, rtol=1e-3), opening
    profile = crack(up, lo, dx, 0, 0).mode_I_crack_displacements(NU, E)
    assert np.isclose(np.max(profile[0]) - np.min(profile[1]), a, rtol=1e-2)


def test_single_model_all_parameters():
    for litho in (True, False):
        for parameter in PARAMETERS:
            for crack_inputs, dy in ((DIKE, 0), (FAULT, 5e6)):
                fig = single_model(crack_inputs, MODEL_INPUTS, 0, dy, 0, 0, 0, G_MARS, litho, 60,
                                   XLIMS, YLIMS, RES, parameter)
                assert isinstance(fig, matplotlib.figure.Figure), (parameter, litho)
                plt.close(fig)


class RecordingCrack(crack):
    """crack that records the driving stresses of every instance created."""
    created = []

    def __init__(self, up, lo, dx, dy, dz):
        RecordingCrack.created.append((dx, dy, dz))
        super().__init__(up, lo, dx, dy, dz)


def run_recording(crack_inputs, dx, dy, dz, steps):
    RecordingCrack.created = []
    original = Modeling_functions.crack
    Modeling_functions.crack = RecordingCrack
    try:
        modes = dynamic_model(crack_inputs, MODEL_INPUTS, dx, dy, dz, 0, 0, G_MARS, True, steps,
                              XLIMS, YLIMS, RES, 2, False, '.')
    finally:
        Modeling_functions.crack = original
    return np.asarray(RecordingCrack.created), np.asarray(modes).astype(str)


def test_dynamic_model_scales_driving_stresses_per_step():
    steps = [0.25, 0.5, 0.75, 1.0]
    # Dike: dx follows from the aperture (the input dx is ignored); dy and dz scale with the step
    b = 1500
    dx_final = DIKE[0]*G(E, NU)/(b*(1 - NU))
    recorded, modes = run_recording(DIKE, 123.0, 2e6, 1e6, steps)
    assert np.allclose(recorded, [[f*dx_final, f*2e6, f*1e6] for f in steps])
    # Faults: dilatant (dx > 0), compaction band (dx < 0) and pure shear (dx = 0)
    for dx in (2e6, -2e6, 0.0):
        recorded, modes = run_recording(FAULT, dx, 5e6, 0.0, steps)
        assert np.allclose(recorded, [[f*dx, f*5e6, 0.0] for f in steps]), dx
        assert modes.shape == (len(np.arange(*YLIMS, RES[1])), len(np.arange(*XLIMS, RES[0])))


def test_dynamic_model_one_png_per_step():
    folder = tempfile.mkdtemp()
    steps = [0.5, 1.0]
    dynamic_model(DIKE, MODEL_INPUTS, 0, 0, 0, 0, 0, G_MARS, True, steps, XLIMS, YLIMS, RES, 2, True, folder)
    plt.close('all')
    assert len(glob.glob(os.path.join(folder, '*.png'))) == len(steps)


def test_tip_on_a_grid_node_is_singular():
    # Upper tip at (-125, -510): a node of the 250 x 250 m grid, whose x nodes are
    # shifted half a cell (-1375, ..., -125, 125, ...) and whose y nodes start at -2510
    crack_inputs = [0, -125, -510, 375, -1376.0254]
    modes = dynamic_model(crack_inputs, MODEL_INPUTS, 2e6, 5e6, 0, 0, 0, G_MARS, True, [0.5, 1.0],
                          XLIMS, [-2510, -10], RES, 2, False, '.')
    assert 'Singular' in np.asarray(modes).astype(str)


def test_evaluate_stress_state_edge_cases():
    assert evaluate_stress_state(float('nan'), 1e6, CFI, T, 'values')[3] == 'Singular'
    assert evaluate_stress_state(np.inf, 1e6, CFI, T, 'values')[3] == 'Singular'
    assert evaluate_stress_state(5e6, 5e6, CFI, T, 'values')[3] == 'Below'          # isotropic compression
    assert evaluate_stress_state(-2e6, -2e6, CFI, T, 'values')[3] == 'Above'        # isotropic tension > |T|
    for args in ((1e6, 5e6, CFI, T), (5e6, 1e6, CFI, 1.5e6)):                      # p < s ; T > 0
        try:
            evaluate_stress_state(*args, 'values')
        except InputError:
            continue
        raise AssertionError('no InputError for ' + str(args))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('OK  ', test.__name__)
    print(len(tests), 'tests passed')
