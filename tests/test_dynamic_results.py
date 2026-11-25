"""
Tests of compute_dynamic_model, DynamicResults and plotting.plot_dynamic_step
(Phase 1, 2026-11-24).

Run with pytest, or directly: python tests/test_dynamic_results.py
"""

import glob
import io
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

import plotting
from Modeling_functions import compute_dynamic_model, dynamic_model
from results import DynamicResults, FAILED_MODES

warnings.filterwarnings('ignore')

MODEL_INPUTS = [30e9, 0.25, 2900, 0.57, -1.5e6]
XLIMS, YLIMS, RES = [-1500, 1500], [-2500, -10], [250, 250]
DIKE = [2, 0, -500, 0, -2000]
STEPS = [0.25, 0.5, 0.75, 1.0]


def run(plot=False, folder='.'):
    return dynamic_model(DIKE, MODEL_INPUTS, 0, 0, 0, 0, 0, 9.81, True, STEPS, XLIMS, YLIMS, RES, 2, plot, folder)


def compute():
    return compute_dynamic_model(DIKE, MODEL_INPUTS, 0, 0, 0, 0, 0, 9.81, True, STEPS, XLIMS, YLIMS, RES)


def test_compute_matches_dynamic_model():
    results = compute()
    assert isinstance(results, DynamicResults) and len(results) == len(STEPS)
    assert np.array_equal(results.final_mode().astype(str), np.asarray(run()).astype(str))
    for k, step in enumerate(results, start=1):
        meta = step.metadata
        assert meta['step'] == k and meta['n_steps'] == len(STEPS) and meta['length_label'] == 'a'
        assert np.isclose(meta['length'], DIKE[0]*STEPS[k - 1])
        assert step.units['failure_sn'] == 'MPa' and step.units['plane_mode_I'] == 'rad'


def test_first_failure_is_kept_and_failure_step():
    results = compute()
    first = results.failure_step()
    assert first.shape == results[0].shape and first.min() >= 0 and first.max() <= len(results)
    for k, step in enumerate(results, start=1):
        failed = np.isin(step['failure_mode'].astype(str), FAILED_MODES)
        # Nodes that failed by step k have failure_step <= k, and stay failed afterwards
        assert np.all(first[failed] <= k) and np.all(first[failed] > 0)
        if k > 1:
            before = np.isin(results[k - 2]['failure_mode'].astype(str), FAILED_MODES)
            assert np.all(failed[before])
            same = np.array_equal(results[k - 2]['failure_mode'][before], step['failure_mode'][before])
            assert same


def test_save_and_load():
    results = compute()
    path = os.path.join(tempfile.mkdtemp(), 'dynamic.npz')
    results.save(path)
    back = DynamicResults.load(path)
    assert len(back) == len(results) and back.metadata == results.metadata
    for a, b in zip(results, back):
        assert a.metadata == b.metadata and a.names() == b.names()
        for name in a.names():
            assert np.array_equal(a[name], b[name], equal_nan=a[name].dtype.kind == 'f'), name


def test_saved_figures_equal_plot_dynamic_step():
    folder = tempfile.mkdtemp()
    run(plot=True, folder=folder)
    files = sorted(glob.glob(os.path.join(folder, '*.png')), key=lambda p: int(os.path.basename(p).split('_')[-3]))
    assert len(files) == len(STEPS)
    results = compute()
    for path, step in zip(files, results):
        figure = plotting.plot_dynamic_step(step, 2)
        buffer = io.BytesIO()
        figure.savefig(buffer, dpi=300, format='png')
        plt.close(figure)
        buffer.seek(0)
        assert np.array_equal(plt.imread(path), plt.imread(buffer, format='png')), path


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('OK  ', test.__name__)
    print(len(tests), 'tests passed')
