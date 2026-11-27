"""
Tests of the package layer (Phase 3, Nov 2026): the dies facade and pyproject.toml.
The modules at the root keep working as scripts; dies only re-exports them.
"""

import glob
import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')

import dies
import DiES_core, Modeling_functions, Failure_Determination, results, plotting


def test_facade_reexports_the_same_objects():
    """dies.X is the object of the root module, not a copy."""
    pairs = [(dies.model, DiES_core.model), (dies.crack, DiES_core.crack), (dies.InputError, DiES_core.InputError),
             (dies.single_model, Modeling_functions.single_model),
             (dies.dynamic_model, Modeling_functions.dynamic_model),
             (dies.compute_dynamic_model, Modeling_functions.compute_dynamic_model),
             (dies.evaluate_stress_state, Failure_Determination.evaluate_stress_state),
             (dies.failure_contact, Failure_Determination.failure_contact),
             (dies.DiESResults, results.DiESResults), (dies.DynamicResults, results.DynamicResults),
             (dies.plotting, plotting)]
    for a, b in pairs:
        assert a is b
    assert dies.FAILURE_TOLERANCE == Failure_Determination.FAILURE_TOLERANCE
    for name in dies.__all__:
        assert hasattr(dies, name), name


def test_pyproject_lists_every_core_module():
    """Every .py module at the root, except the user's template script, is in py-modules,
    so that a new core module is not left out of the package by mistake."""
    tomllib = pytest.importorskip('tomllib')      # Python >= 3.11
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as f:
        config = tomllib.load(f)
    listed = set(config['tool']['setuptools']['py-modules'])
    root_modules = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ROOT, '*.py'))}
    assert listed == root_modules - {'Running_Models'}
    # Every folder of the dies package with an __init__.py is listed (since Phase 4: dies.app, dies.app.ui)
    packages = sorted(os.path.relpath(os.path.dirname(p), ROOT).replace(os.sep, '.')
                      for p in glob.glob(os.path.join(ROOT, 'dies', '**', '__init__.py'), recursive=True))
    assert sorted(config['tool']['setuptools']['packages']) == packages


def test_facade_runs_a_model():
    m = dies.model([-2000, 2000], [-3000, -10], 250, 250, 500, 30e9, 0.25, 2900, 0.57, -1.5e6, 9.81)
    res = m.compute(dies.crack((0, -500), (0, -2000), 5e6, 0, 0), 0, 0, True)
    assert res.shape == m.grid()[0].shape and 'sxx' in res
