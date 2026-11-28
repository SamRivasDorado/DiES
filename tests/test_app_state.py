"""
Tests of the parts of the DiES app that do not need Streamlit (Phase 4, Nov 2026):
the conversion between the forms and the scenario (dies/app/state.py), the default
scenario (dies/app/defaults.py) and the figures of the app (dies/app/figures.py).
"""

import copy
import os
import sys
import warnings

import numpy as np
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


# ----------------------------------------------------------------- views (Phase 4, step 2)

@pytest.fixture(scope='module')
def dike_run():
    raw = _without_output(_read(DIKE))
    raw['analysis'] = {'type': 'static'}
    return dies.Scenario.from_dict(raw).run()


def test_every_figure_of_every_view_is_drawn(dike_run):
    """Each figure of dies.app.views draws with its default options and with every option it takes,
    and the profile has its fields."""
    from dies.app import views
    import matplotlib.pyplot as plt
    changed = {'cmap': 'cividis', 'limits': (-1.0, 1.0), 'center_zero': True, 'contours': False,
               'contour_labels': False, 'info_box': False, 'arrow_scale': 2.0, 'step': 0.5}
    for view, items in views.VIEWS.items():
        for item in items:
            res = item.results(dike_run)
            for options in ({}, changed):
                style, step = views.style_for(item, options)
                assert set(style) <= set(item.options)
                figure = item.draw(res, dike_run, style, step, {'plane_angle': 45.0})
                plt.close(figure)
            data = views.profile(dike_run, item, (-4000, -500), (4000, -500), n=50)
            assert len(data) == 3 + len(item.profile) and all(len(v) == 50 for v in data.values())


def test_style_for_leaves_out_the_defaults(dike_run):
    """Options at their default are not passed, so that the app draws exactly what the scripts draw."""
    from dies.app import views
    item = views.figure('Stresses', 'sxx')
    assert views.style_for(item, {'cmap': None, 'contours': True, 'step': None}) == ({}, None)
    assert views.style_for(item, {'cmap': 'RdBu_r', 'info_box': False, 'step': 2}) == ({'cmap': 'RdBu_r'}, 2)
    angle = views.figure('Principal & trajectories', 'angle')
    assert views.style_for(angle, {'cmap': 'hsv', 'center_zero': True, 'step': 5}) == ({'cmap': 'hsv'}, None)


def test_probe_profile_and_plane_angle(dike_run):
    """The probe gives the values of the nearest node in plotting units; the default profile runs
    along the top row of nodes; the default plane angle is the optimal one."""
    from dies.app import views
    x, y, rows = views.probe(dike_run, 480, -1030)
    expected = dike_run.static.probe(480, -1030)
    assert (x, y) == (expected['x'], expected['y'])
    values = dict(rows)
    assert values['σxx (MPa)'] == pytest.approx(expected['sxx']/1e6)
    assert values['ux (m)'] == expected['ux']
    assert values['Failure state'] in ('Below', 'Above', 'Mode I', 'Mixed-mode I-II',
                                       'Mode II - tension-compression', 'Mode II - compression')
    A, B = views.default_profile(dike_run.static)
    assert A[1] == B[1] == dike_run.static.y_nodes().max()
    assert (A[0], B[0]) == (dike_run.static.x_nodes().min(), dike_run.static.x_nodes().max())
    data = views.profile(dike_run, views.figure('Displacements', 'uy_contours'), A, B)
    # Surface profile above a vertical dike: uy is symmetric about x = 0
    np.testing.assert_allclose(data['uy (m)'], data['uy (m)'][::-1], atol=1e-12)
    assert views.optimal_plane_angle(dike_run) == 59.84


def test_overlays_and_profile_plot(dike_run):
    import matplotlib.pyplot as plt
    import plotting
    import matplotlib.transforms as mtransforms
    probes = [(-50, -600, 'P1'), (950, -600, 'P2'), (4950, -3000, 'P3')]
    profiles = [((-4950, -100), (4950, -100), '1'), ((-4950, -3000), (4950, -3000), '2')]
    fig = figures.add_overlays(plotting.plot_stresses(dike_run.static, 'sxx'), probes, profiles)
    ax = fig.axes[0]
    labels = [t for t in ax.texts if t.get_text() in ('A1', 'B1', 'A2', 'B2', 'P1', 'P2', 'P3')]
    assert sorted(t.get_text() for t in labels) == ['A1', 'A2', 'B1', 'B2', 'P1', 'P2', 'P3']
    # Every label stays inside the map, also those at its edges (Sam, 27 Nov 2026)
    fig.canvas.draw()
    box = ax.get_window_extent()
    for text in labels:
        extent = text.get_window_extent()
        assert box.x0 - 1 <= extent.x0 and extent.x1 <= box.x1 + 1, text.get_text()
        assert box.y0 - 1 <= extent.y0 and extent.y1 <= box.y1 + 1, text.get_text()
    # The marks do not change the limits of the map
    assert ax.get_xlim() == tuple(dike_run.static.metadata['x_limits'])
    plt.close(fig)
    from dies.app import views
    data = views.profile(dike_run, views.figure('Displacements', 'uxy'), (-4000, -100), (4000, -100), n=20)
    fig = figures.plot_profile(data, figures.line_colour(1), title='Profile 2', minor_grid=True, markers=True)
    assert len(fig.axes) == 2 and fig.axes[0].get_lines()[0].get_color() == figures.line_colour(1)
    plt.close(fig)
    fig = figures.plot_profile(data, axis_labels=False)
    assert fig.axes[0].get_ylabel() == '' and fig.axes[0].texts[0].get_text() == 'ux (m)'
    plt.close(fig)


def test_arrow_spacing_and_probe_table(dike_run):
    """Arrow density: the static results at one node every n, centred, without computing again; the
    probe table has one row per probe."""
    from dies.app import views
    res = dike_run.static
    every3 = views.subsample(res, 3)
    rows, columns = res.shape
    assert every3.shape == ((rows - 1)//3 + 1, (columns - 1)//3 + 1)
    i0 = ((rows - 1) % 3)//2 ; j0 = ((columns - 1) % 3)//2
    np.testing.assert_array_equal(every3['ux'], res['ux'][i0::3, j0::3])
    assert views.subsample(res, 1) is res
    item = views.figure('Displacements', 'uxy')
    assert item.results(dike_run, {'arrow_spacing': 3}).shape == every3.shape
    assert item.results(dike_run, {'arrow_spacing': None}).shape == dike_run.coarse().shape
    assert views.style_for(item, {'arrow_spacing': 3, 'arrow_scale': 2.0}) == ({'arrow_scale': 2.0}, None)
    table = views.probe_table(dike_run, [(0, -500, 'P1'), (1000, -2000, 'P2')])
    assert [row['Probe'] for row in table] == ['P1', 'P2']
    assert table[1]['node x (m)'] == res.probe(1000, -2000)['x']


def test_plane_angle_is_not_in_the_key():
    """The plane angle is only a label: changing it does not make the results old."""
    raw = _without_output(_read(DIKE))
    other = copy.deepcopy(raw)
    other['analysis']['plane_angle'] = '45 deg'
    assert state.scenario_key(dies.Scenario.from_dict(raw)) == state.scenario_key(dies.Scenario.from_dict(other))


def test_export_zip(dike_run):
    """The ZIP holds what `dies run` writes for the choices, from the results of the run (not computed
    again): the scenario as YAML (static, with the label angle given), the results and the figures."""
    import io
    import zipfile
    from dies.app import export
    from results import DiESResults
    data, files = export.export_zip(dike_run, 45, ('npz',), ['sxx', 'failure_mode'], 'svg')
    assert files == ['scenario.yaml', 'static_results.npz', 'sxx.svg', 'failure_mode.svg']
    archive = zipfile.ZipFile(io.BytesIO(data))
    folder = export.folder_name(dike_run.scenario.name)
    assert archive.namelist() == [folder + '/' + name for name in files]
    scenario = state.read_scenario_text(archive.read(folder + '/scenario.yaml').decode('utf-8'), 'scenario.yaml')
    assert scenario['analysis'] == {'type': 'static', 'plane_angle': '45 deg'}
    assert scenario['output']['figures']['static'] == ['sxx', 'failure_mode']
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'r.npz')
        with open(path, 'wb') as f:
            f.write(archive.read(folder + '/static_results.npz'))
        np.testing.assert_array_equal(DiESResults.load(path)['sxx'], dike_run.static['sxx'])
    with pytest.raises(ValueError, match='unknown figure'):
        export.export_zip(dike_run, figures=['not_a_figure'])
    assert export.folder_name('Vertical dike, opening only') == 'Vertical_dike_opening_only_results'


def test_probe_table_edits():
    """The edits of the probe table (st.data_editor) applied to the list of probes: edited cells
    (an emptied one keeps its value), deleted rows and added rows (missing values from the default)."""
    from dies.app import views
    probes = [{'x': 0.0, 'y': -500.0, 'show': True, 'id': 1}, {'x': 1000.0, 'y': -500.0, 'show': True, 'id': 2},
              {'x': 2000.0, 'y': -900.0, 'show': True, 'id': 3}]
    counter = iter(range(10, 20))
    default = {'x': -100.0, 'y': -300.0}
    new_entry = lambda values: dict(default, show=True, **values, id=next(counter))
    edits = {'edited_rows': {0: {'x (m)': 2500, 'On the map': False}, '2': {'y (m)': None}},
             'deleted_rows': [1],
             'added_rows': [{'y (m)': -1200}, {}]}
    result = views.apply_table_edits(probes, edits, new_entry)
    assert result == [{'x': 2500.0, 'y': -500.0, 'show': False, 'id': 1},
                      {'x': 2000.0, 'y': -900.0, 'show': True, 'id': 3},
                      {'x': -100.0, 'y': -1200, 'show': True, 'id': 10},
                      {'x': -100.0, 'y': -300.0, 'show': True, 'id': 11}]
    assert probes[0]['x'] == 0.0                     # the list given is not changed
    assert views.apply_table_edits(probes, {}, new_entry) == probes
    # Edits of the computed columns (not editable in the app) are ignored
    assert views.apply_table_edits(probes, {'edited_rows': {0: {'ux (m)': 5}}}, new_entry) == probes


# ------------------------------------------------------- outputs of the player (Phase 4, step 3)

def test_dynamic_views_probes_and_profiles():
    import dies
    from dies.app import dynamic_views
    run = dies.load_scenario(os.path.join(ROOT, 'examples', 'dike_example.yaml')).run()
    dyn = run.dynamic
    # Probes at a step: the failure state and first failure up to that step
    rows = dynamic_views.probe_table(run, 32, [(0.0, -550.0, 'P1')])
    assert rows[0]['Probe'] == 'P1' and rows[0]['Failure state'] != 'Below'
    step = rows[0]['First failure: step']
    early = dynamic_views.probe_table(run, int(step) - 1, [(0.0, -550.0, 'P1')])
    assert early[0]['First failure: step'] == 0 and np.isnan(early[0]['First failure: driving stress (MPa)'])
    # Profiles of every map output, at a step
    for item in dynamic_views.OUTPUTS:
        if item.profile:
            data = dynamic_views.profile(run, 16, item, (-4950, -100), (4950, -100), n=50)
            assert len(data['x (m)']) == 50 and len(data) == 4
    # Options left at their default are not passed to plotting; the options of the app never
    style = dynamic_views.style_for(dynamic_views.output('planes'),
                                    {'cmap': 'magma', 'above': True, 'arrow_spacing': 3, 'same_range': True})
    assert style == {'cmap': 'magma'}
    assert dynamic_views.step_label(dyn, 16).startswith('Step 16 of 32 · 50 %')
