"""
Tests of the parts of the DiES app that do not need Streamlit (Phase 4, Nov 2026):
the conversion between the forms and the scenario (dies/app/state.py), the default
scenario (dies/app/defaults.py) and the figures of the app (dies/app/figures.py).
"""

import copy
import os
import sys
import warnings

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')

import dies
from dies.app import state, figures
from dies.app.defaults import DEFAULT_SCENARIO

DIKE = os.path.join(ROOT, 'examples', 'dike_example.yaml')
FAULT = os.path.join(ROOT, 'examples', 'fault_example.yaml')


def _read(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _without_output(raw):
    # The scenarios the app computes have no output section
    raw = copy.deepcopy(raw)
    raw.pop('output', None)
    return raw


def test_default_scenario_is_the_dike_example():
    """The scenario the app opens with is examples/dike_example.yaml."""
    assert DEFAULT_SCENARIO == _read(DIKE)


@pytest.mark.parametrize('path', [DIKE, FAULT])
def test_form_round_trip_gives_the_same_scenario(path):
    """Example -> form -> scenario: the same checked scenario (in the units of the core)."""
    raw = _read(path)
    form = state.form_from_raw(raw)
    kind = raw['analysis']['type']
    scenario, problems = state.check(state.raw_from_form(form, kind))
    assert problems == []
    expected = dies.Scenario.from_dict(_without_output(raw))
    assert state.scenario_key(scenario) == state.scenario_key(expected)
    assert scenario.data['name'] == expected.data['name']


def test_static_and_dynamic_scenarios_of_one_form():
    """The same form gives the static scenario and the dynamic one, with its step or its steps."""
    form = state.form_from_raw(_read(DIKE))
    static, problems = state.check(state.raw_from_form(form, 'static'))
    assert problems == [] and static.analysis == 'static'
    dynamic, _ = state.check(state.raw_from_form(form, 'dynamic'))
    assert dynamic.analysis == 'dynamic' and dynamic.data['analysis']['step'] == 0.25e6
    form['dynamic'] = {'increment': 'steps', 'step': 0.25, 'steps': 10}
    dynamic, _ = state.check(state.raw_from_form(form, 'dynamic'))
    assert dynamic.data['analysis']['steps'] == 10 and dynamic.data['analysis']['step'] is None


def test_key_does_not_depend_on_units_name_or_output():
    """'5 km' and '5000 m' give the same cache key; so do a different name or output section."""
    raw = _read(DIKE)
    other = copy.deepcopy(raw)
    other['grid']['x'] = ['-5000 m', '5000 m']
    other['name'] = 'Another name'
    other['output'] = {'results': ['npz']}
    assert state.scenario_key(dies.Scenario.from_dict(raw)) == state.scenario_key(dies.Scenario.from_dict(other))
    other['gravity'] = '3.71 m/s2'
    assert state.scenario_key(dies.Scenario.from_dict(raw)) != state.scenario_key(dies.Scenario.from_dict(other))


def test_any_gravity_value():
    """The gravity is a free value (any planet or an exotic case), 0 included."""
    form = state.form_from_raw(_read(DIKE))
    for g in (0.0, 1.62, 24.79):
        form['gravity'] = g
        scenario, problems = state.check(state.raw_from_form(form))
        assert problems == [] and scenario.data['gravity'] == g


def test_empty_and_invalid_fields_are_reported_in_their_section():
    """Empty fields are reported as missing; each problem goes to the section of its field."""
    form = state.form_from_raw(_read(DIKE))
    form['gravity'] = None
    form['grid']['cell_x'] = None
    form['layers'][0]['E'] = -30.0
    form['crack']['sigma_I'] = 5.0          # aperture and sigma_I: one too many
    form['dynamic']['step'] = 0.0
    scenario, problems = state.check(state.raw_from_form(form, 'dynamic'))
    assert scenario is None
    sections = {state.problem_section(p) for p in problems}
    assert sections == {'gravity', 'grid', 'layers', 'cracks', 'analysis'}
    described = [state.describe_problem(p) for p in problems]
    assert 'gravity: missing' in described
    assert 'cell size: missing' in described
    assert 'Layer 1, E (GPa): must be larger than 0' in described
    assert 'Crack: give the aperture or σI, and 0 in the other one' in described
    assert 'step: must be larger than 0' in described


def test_zero_means_not_given_for_the_opening():
    """In the crack table, 0 in the aperture or in sigma_I means "not given" (Sam, 27 Nov 2026:
    with 0 in one of them, the check took both as given and the error stayed)."""
    form = state.form_from_raw(_read(DIKE))
    assert form['crack']['aperture'] == 2 and form['crack']['sigma_I'] == 0   # the one not given shows 0

    def opening(aperture, sigma_I, sigma_II=0.0):
        form['crack'].update(aperture=aperture, sigma_I=sigma_I, sigma_II=sigma_II)
        scenario, problems = state.check(state.raw_from_form(form, 'static'))
        crack = scenario.data['cracks'][0] if scenario else None
        return crack, [state.describe_problem(p) for p in problems]

    # Sam's sequence: aperture 0 and sigma_I 10, then aperture 2 and sigma_I 0: both valid
    crack, problems = opening(0.0, 10.0)
    assert problems == [] and crack['aperture'] is None and crack['sigma_I'] == 10e6
    crack, problems = opening(2.0, 0.0)
    assert problems == [] and crack['aperture'] == 2.0 and crack['sigma_I'] == 0
    # Empty cells work as 0
    crack, problems = opening(None, 10.0)
    assert problems == [] and crack['sigma_I'] == 10e6
    # Both 0: no opening (a fault with shear only)
    crack, problems = opening(0.0, 0.0, 15.0)
    assert problems == [] and crack['aperture'] is None and crack['sigma_II'] == 15e6
    # Both non-zero, or a negative aperture: reported
    _, problems = opening(2.0, 10.0)
    assert problems == ['Crack: give the aperture or σI, and 0 in the other one']
    _, problems = opening(-1.0, 0.0)
    assert problems[0].startswith('Crack, aperture: must be larger than 0')


def test_two_layers_and_empty_rows():
    """A second layer is read and written back; an empty row (just added in the table) is skipped."""
    raw = _read(DIKE)
    raw['layers'].append({'name': 'basement', 'top': '-3 km', 'E': '50 GPa', 'nu': 0.3, 'density': '2700 kg/m3',
                          'friction': 0.6, 'T': '-3 MPa'})
    form = state.form_from_raw(raw)
    assert form['layers'][1]['top'] == -3000 and form['layers'][1]['E'] == 50
    form['layers'].append({key: None for key, _, _ in state.LAYER_COLUMNS})
    scenario, problems = state.check(state.raw_from_form(form, 'dynamic'))
    assert problems == []
    assert scenario.layer_tops() == [0.0, -3000.0]
    assert state.scenario_key(scenario) == state.scenario_key(dies.Scenario.from_dict(_without_output(raw)))


def test_a_file_with_problems_can_be_opened():
    """What cannot be read is left empty in the form, and reported when checked."""
    raw = _read(DIKE)
    raw['layers'][0]['E'] = 30e9                 # no unit
    raw['grid']['cell'] = ['100 m']              # not a pair
    form = state.form_from_raw(raw)
    assert form['layers'][0]['E'] is None
    assert form['grid']['cell_x'] is None and form['grid']['cell_y'] is None
    assert form['layers'][0]['nu'] == 0.25       # the rest is read
    _, problems = state.check(raw)
    assert any(p.startswith('layers[0].E') for p in problems)


def test_saved_yaml_reads_back_the_same():
    """The YAML saved by the app is a valid scenario equal to the form; a static save drops the
    dynamic figures of the output section (only allowed in a dynamic analysis)."""
    raw = _read(DIKE)
    form = state.form_from_raw(raw)
    for kind in ('static', 'dynamic'):
        text = state.scenario_yaml(state.raw_from_form(form, kind, include_output=True))
        assert '[-5000 m, 5000 m]' in text      # pairs on one line, as in the examples
        back = state.read_scenario_text(text, 'saved.yaml')
        scenario, problems = state.check(back)
        assert problems == [] and scenario.analysis == kind
        assert ('dynamic' in back['output']['figures']) == (kind == 'dynamic')
    # The dynamic save is the example itself
    back = state.read_scenario_text(state.scenario_yaml(state.raw_from_form(form, 'dynamic', True)), 'x.yaml')
    assert dies.Scenario.from_dict(back).to_dict() == dies.Scenario.from_dict(raw).to_dict()


def test_reading_files():
    """YAML and JSON are read; a file that is not a scenario raises ScenarioError."""
    import json
    raw = _read(DIKE)
    assert state.read_scenario_text(json.dumps(raw), 'a.json') == raw
    with pytest.raises(dies.ScenarioError):
        state.read_scenario_text('grid: [unclosed', 'bad.yaml')
    with pytest.raises(dies.ScenarioError):
        state.read_scenario_text('- just\n- a list\n', 'list.yaml')


def test_units_of_the_forms():
    """Numbers of the forms <-> values with their units."""
    assert state.to_quantity(30, 'modulus') == '30 GPa'
    assert state.to_quantity(-1.5, 'stress') == '-1.5 MPa'
    assert state.to_quantity(0.1 + 0.2, 'length') == '0.3 m'     # no float noise
    assert state.to_quantity(1.0, None) == 1 and state.to_quantity(0.57, None) == 0.57
    assert state.to_quantity(float('nan'), 'length') is None
    assert state.to_display('5 km', 'length') == 5000
    assert state.to_display('2.9 g/cm3', 'density') == pytest.approx(2900)
    assert state.to_display('-1.5e6 Pa', 'stress') == -1.5
    assert state.to_display('30e9', 'modulus') is None          # no unit
    assert state.cohesion(-1.5) == 3 and state.cohesion(None) is None


def test_app_figures_render():
    """The sketch of the model and the crack profile are drawn and saved (PNG, PDF, SVG)."""
    raw = _read(DIKE)
    raw['layers'].append({'name': 'basement', 'top': '-3 km', 'E': '50 GPa', 'nu': 0.3, 'density': '2700 kg/m3',
                          'friction': 0.6, 'T': '-3 MPa'})
    scenario = dies.Scenario.from_dict(raw)
    png = figures.render(lambda: figures.plot_model(scenario), 'png', 60)
    assert png.startswith(b'\x89PNG')
    assert figures.render(lambda: figures.plot_model(scenario), 'pdf').startswith(b'%PDF')
    assert b'<svg' in figures.render(lambda: figures.plot_crack(scenario), 'svg')
    import matplotlib.pyplot as plt
    assert plt.get_fignums() == []               # every figure is closed
