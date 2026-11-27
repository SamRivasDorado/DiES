"""
Tests of the interface of the DiES app with Streamlit's AppTest (Phase 4, Nov 2026):
the app starts, both tabs compute and give the results of Scenario.run(), the problems
of the inputs disable Run, and a scenario already computed comes from the cache.
Skipped if Streamlit is not installed (it is an optional dependency: pip install -e .[app]).
"""

import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

import dies
from dies.app.defaults import DEFAULT_SCENARIO

APP = os.path.join(ROOT, 'dies', 'app', 'streamlit_app.py')


def _start():
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception
    return app


def _key(app, name):
    # Key of a widget in the current version of the form (dies.app.session.key)
    return name + '@' + str(app.session_state['version'])


def test_app_starts_with_the_example():
    app = _start()
    assert [e.value for e in app.error] == []
    assert app.session_state[_key(app, 'gravity')] == 9.81
    assert not app.button(key='run_static').disabled and not app.button(key='run_dynamic').disabled


def test_style_targets_the_box_of_the_models():
    """The CSS of ui/style.py styles the tabs of the box with key 'models', the one around the two
    tabs in streamlit_app.py (checked by eye in a browser: AppTest does not render CSS)."""
    from dies.app.ui import style
    assert '.st-key-models [data-testid="stTab"]' in style.CSS
    with open(APP, encoding='utf-8') as f:
        assert "st.container(border=True, key='models')" in f.read()


def test_single_model_gives_the_results_of_the_scenario():
    """Run single model: the static results are those of Scenario.run(), bit for bit."""
    app = _start()
    app.button(key='run_static').click().run()
    assert not app.exception
    results = app.session_state['static_run']['results']
    raw = dict(DEFAULT_SCENARIO, analysis={'type': 'static', 'plane_angle': '60 deg'})
    raw.pop('output')
    expected = dies.Scenario.from_dict(raw).run()
    for name in ('ux', 'uy', 'sxx', 'syy', 'sxy', 's1', 's3', 'sn', 'ss'):
        np.testing.assert_array_equal(results.static[name], expected.static[name])
    assert results.dynamic is None


def test_dynamic_model_runs_and_is_cached():
    """Run dynamic model: 32 steps (0.25 MPa up to 8 MPa); the same inputs again come from the cache."""
    app = _start()
    app.button(key='run_dynamic').click().run()
    assert not app.exception
    run = app.session_state['dynamic_run']
    assert len(run['results'].dynamic) == 32
    app.button(key='run_dynamic').click().run()
    assert app.session_state['dynamic_run']['cached']


def test_problems_disable_run_and_are_shown():
    """An invalid value shows its problem next to its field and disables Run; fixing it enables Run."""
    app = _start()
    app.number_input(key=_key(app, 'grid.y_top')).set_value(10.0).run()
    messages = ' '.join(e.value for e in app.error)
    assert 'the top of the model must be below the surface' in messages
    assert app.button(key='run_static').disabled and app.button(key='run_dynamic').disabled
    app.number_input(key=_key(app, 'grid.y_top')).set_value(-10.0).run()
    assert not app.button(key='run_static').disabled


def test_empty_gravity_is_missing_and_step_only_blocks_the_dynamic_model():
    app = _start()
    app.number_input(key=_key(app, 'gravity')).set_value(None).run()
    assert any('gravity: missing' in e.value for e in app.error)
    app.number_input(key=_key(app, 'gravity')).set_value(3.71).run()
    app.number_input(key=_key(app, 'dynamic.step')).set_value(0.0).run()
    assert any('step: must be larger than 0' in e.value for e in app.error)
    assert not app.button(key='run_static').disabled and app.button(key='run_dynamic').disabled


def test_changed_inputs_mark_the_results_as_old():
    app = _start()
    app.button(key='run_static').click().run()
    app.number_input(key=_key(app, 'gravity')).set_value(3.71).run()
    assert any('The inputs have changed since the last run' in w.value for w in app.warning)


def test_launcher_and_back_to_the_example():
    """The dies-app entry point exists; 'Back to the example' opens a new version of the form."""
    from dies.app import launcher
    assert callable(launcher.main)
    app = _start()
    version = app.session_state['version']
    app.number_input(key=_key(app, 'gravity')).set_value(3.71).run()
    app.button[[b.label for b in app.button].index('Back to the example')].click().run()
    assert app.session_state['version'] == version + 1
    assert app.session_state[_key(app, 'gravity')] == 9.81
