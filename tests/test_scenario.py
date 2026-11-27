"""
Tests of the scenarios (dies/scenario.py, Phase 3, Nov 2026): units, checks of the
format, and results identical, bit for bit, to those of the core functions called with
the same values (model.compute and compute_dynamic_model).
"""

import copy
import glob
import json
import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if folder not in sys.path:
        sys.path.insert(0, folder)
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import dies
from dies.scenario import STATIC_FIGURES
from DiES_core import model, crack, InputError
from Helper_functions import G
from Modeling_functions import compute_dynamic_model, layered_inputs
from results import DiESResults, DynamicResults
from visual_check import MODEL_INPUTS, G_EARTH

EXAMPLES = os.path.join(ROOT, 'examples')
DIKE = os.path.join(EXAMPLES, 'dike_example.yaml')


def small(raw, cell='500 m'):
    """A copy of a scenario dict on a coarse grid, so that the tests run fast."""
    raw = copy.deepcopy(raw)
    raw['grid']['cell'] = [cell, cell]
    return raw


def dike_dict():
    """The dike example as a dict (as read from its YAML file)."""
    return copy.deepcopy(dies.load_scenario(DIKE).raw)


def problems(raw):
    """The list of problems that a scenario dict has (empty if it is valid)."""
    try:
        dies.Scenario.from_dict(raw)
    except dies.ScenarioError as error:
        return error.problems
    return []


# ----------------------------------------------------------------- units

def test_parse_quantity_converts_to_core_units():
    q = dies.parse_quantity
    assert q('30 GPa', 'stress') == 30e9
    assert q('-1.5 MPa', 'stress') == -1.5e6
    assert q('5 km', 'length') == 5000 and q('-10 m', 'length') == -10 and q('2.5e3 m', 'length') == 2500
    assert q('2900 kg/m3', 'density') == 2900 and q('2.9 g/cm3', 'density') == pytest.approx(2900)
    assert q('9.81 m/s2', 'acceleration') == 9.81 and q('3.71 m s-2', 'acceleration') == 3.71
    assert q('60 deg', 'angle') == 60 and q('60°', 'angle') == 60
    assert q('1 rad', 'angle') == pytest.approx(57.29577951)


@pytest.mark.parametrize('value, kind, message', [
    (30e9, 'stress', 'has no unit'),                   # the mistake the units are there to catch
    ('30', 'stress', 'has no unit'),                   # a number as text (YAML reads '30e9' so)
    ('30e9', 'stress', 'has no unit'),
    ('30 m', 'stress', 'a unit of length'),             # a unit of another kind
    ('30 mpa', 'stress', "unknown unit 'mpa'"),          # case sensitive
    ('five km', 'length', 'not a number'),
])
def test_parse_quantity_rejects(value, kind, message):
    with pytest.raises(dies.ScenarioError) as error:
        dies.parse_quantity(value, kind, 'layers[0].E')
    assert message in str(error.value) and 'layers[0].E' in str(error.value)
    # A ScenarioError is an InputError, as the other errors of the inputs of DiES
    assert isinstance(error.value, InputError)


# ------------------------------------------------------------- reading

def test_dike_example_in_core_units():
    s = dies.load_scenario(DIKE)
    d = s.data
    assert d['grid'] == {'x': [-5000, 5000], 'y': [-6000, -10], 'cell': [100, 100]}
    assert d['gravity'] == G_EARTH
    assert s.model_inputs() == MODEL_INPUTS and s.layer_tops() is None
    assert s.crack_inputs() == [2, 0, -1000, 0, -5000]
    # sigma_I from the aperture, as single_model: a*G/(b*(1-v)) = 8 MPa
    assert s.driving_stresses() == (2*G(30e9, 0.25)/(4000*0.75), 0, 0)
    assert s.driving_stresses()[0] == pytest.approx(8e6)
    # 8 MPa at 0.25 MPa per step: 32 steps, as the tests (plot_gallery.investigation_range)
    assert s.investigation_range() == [i/32 for i in range(1, 33)]
    assert d['output']['figures']['dynamic'] == {'show_every': 4, 'downscaling': 2}


@pytest.mark.parametrize('path', sorted(glob.glob(os.path.join(EXAMPLES, '*.yaml'))))
def test_every_example_is_valid(path):
    s = dies.load_scenario(path)
    assert s.analysis in ('static', 'dynamic')


def test_json_and_yaml_give_the_same_scenario(tmp_path):
    raw = dike_dict()
    path = tmp_path / 'dike.json'
    path.write_text(json.dumps(raw))
    assert dies.load_scenario(str(path)).data == dies.load_scenario(DIKE).data


def test_defaults():
    raw = dike_dict()
    for key in ('remote_stresses', 'lithostatic', 'output', 'name', 'description'):
        del raw[key]
    raw['analysis'] = {'type': 'static'}
    d = dies.Scenario.from_dict(raw).data
    assert d['remote_stresses'] == {'xx': 0, 'yy': 0} and d['lithostatic'] is True
    assert d['output'] == {'results': ['npz'], 'figures': {'static': [], 'dynamic': None},
                           'format': 'png', 'dpi': 200}
    assert d['analysis']['plane_angle'] is None


def test_every_problem_is_reported_together():
    raw = dike_dict()
    raw['layers'][0]['E'] = 30e9                 # no unit
    raw['layers'][0]['frictoin'] = 0.6           # typo
    raw['layers'][0]['T'] = '1.5 MPa'            # positive
    raw['cracks'][0]['sigma_I'] = '8 MPa'        # together with the aperture
    raw['analysis']['steps'] = 32                # together with step
    found = problems(raw)
    text = '\n'.join(found)
    assert len(found) >= 5
    assert 'layers[0].E: 30000000000.0 has no unit' in text
    assert "layers[0].frictoin: unknown key; did you mean 'friction'?" in text
    assert 'layers[0].T: must be negative' in text
    assert 'aperture or sigma_I, not both' in text
    assert 'either step (e.g. 0.25 MPa) or steps' in text


@pytest.mark.parametrize('change, message', [
    (lambda r: r['layers'][0].update(top='-1 km'), 'must start at the surface'),
    (lambda r: r['layers'].append(dict(r['layers'][0], top='1 km')), 'from the top down'),
    (lambda r: r['layers'][0].update(nu=0.5), 'between 0 and 0.5'),
    (lambda r: r['layers'][0].update(nu='0.25'), 'plain number'),
    (lambda r: r['layers'][0].update(friction='0.57 deg'), 'plain number'),
    (lambda r: r['grid'].update(y=['-6 km', '0 m']), 'below the surface'),
    (lambda r: r['grid'].update(x=['5 km', '-5 km']), 'smaller than the second'),
    (lambda r: r['cracks'].append(r['cracks'][0]), 'several cracks are not supported'),
    (lambda r: r['cracks'][0].update(aperture='-2 m'), 'larger than 0'),
    (lambda r: r['cracks'][0].update(lower_tip=['0 m', '-1 km']), 'same point'),
    (lambda r: r['analysis'].update(type='dinamic'), "'static' or 'dynamic'"),
    (lambda r: r['analysis'].update(plane_angle='100 deg'), 'between 0 and 90'),
    (lambda r: r['output'].update(format='jpg'), 'png, pdf, svg'),
    (lambda r: r['output']['figures'].update(static=['s1', 'sigma1']), "unknown figure 'sigma1'"),
    (lambda r: r.pop('gravity'), 'gravity: missing'),
    (lambda r: r.update(lithostatic='yes please'), 'true or false'),
])
def test_invalid_scenarios(change, message):
    raw = dike_dict()
    change(raw)
    found = problems(raw)
    assert any(message in p for p in found), found


def test_static_analysis_rejects_dynamic_settings():
    raw = dike_dict()
    raw['analysis']['type'] = 'static'
    text = '\n'.join(problems(raw))
    assert 'analysis.step: only used in a dynamic analysis' in text
    assert 'output.figures.dynamic: only used in a dynamic analysis' in text


def test_dynamic_analysis_needs_a_driving_stress():
    raw = dike_dict()
    del raw['cracks'][0]['aperture']
    assert any('needs a driving stress' in p for p in problems(raw))


def test_unreadable_files(tmp_path):
    bad = tmp_path / 'bad.yaml'
    bad.write_text('grid: [unclosed')
    with pytest.raises(dies.ScenarioError, match='not valid YAML'):
        dies.load_scenario(str(bad))
    with pytest.raises(dies.ScenarioError, match='unknown extension'):
        dies.load_scenario(str(tmp_path / 'scenario.txt'))
    empty = tmp_path / 'empty.yaml'
    empty.write_text('')
    with pytest.raises(dies.ScenarioError, match='expected a section'):
        dies.load_scenario(str(empty))


# ------------------------------------------- same results as the core functions

def assert_same(a, b):
    """Two DiESResults with the same fields, identical bit for bit (NaN in the same places)."""
    assert sorted(a.names()) == sorted(b.names())
    np.testing.assert_array_equal(a.x, b.x) ; np.testing.assert_array_equal(a.y, b.y)
    for name in a.names():
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_static_run_matches_model_compute():
    """Dike and fault, with and without the lithostatic load and remote stresses."""
    E, nu, d, friction, T = MODEL_INPUTS
    for path, litho, rx, ry in ((DIKE, True, 0, 0), (DIKE, False, -3e6, -1e6),
                                (os.path.join(EXAMPLES, 'fault_example.yaml'), True, 0, 0)):
        raw = small(dies.load_scenario(path).raw, '250 m')
        raw['analysis'] = {'type': 'static'}
        raw['output'] = {}
        raw['lithostatic'] = litho
        raw['remote_stresses'] = {'xx': '%g MPa' % (rx/1e6), 'yy': '%g MPa' % (ry/1e6)}
        s = dies.Scenario.from_dict(raw)
        run = s.run()
        assert run.dynamic is None
        # The same model built by hand, as tests/plot_gallery.py does
        c = s.data['cracks'][0]
        if c['aperture'] is not None:
            px = c['aperture']*G(E, nu)/(4000*(1 - nu))
        else:
            px = c['sigma_I']
        disc = crack(tuple(c['upper_tip']), tuple(c['lower_tip']), px, c['sigma_II'], c['sigma_III'])
        m = model([-5000, 5000], [-6000, -10], 250, 250, 0, E, nu, d, friction, T, G_EARTH)
        assert_same(run.static, m.compute(disc, rx, ry, litho))
        # The coarse grid of the vector figures, as single_model
        mv = model([-5000, 5000], [-6000, -10], 250*2.5, 250*3, 0, E, nu, d, friction, T, G_EARTH)
        assert_same(run.coarse(), mv.compute(disc, rx, ry, litho))
        # The scenario is kept in the metadata of the results
        assert run.static.metadata['scenario']['grid']['cell'] == [250, 250]


def test_dynamic_run_matches_compute_dynamic_model():
    """Every field of every step, bit for bit, for the dike (aperture) and a dilatant
    fault given by its driving stresses, with steps instead of step."""
    s = dies.Scenario.from_dict(small(dike_dict()))
    dyn = s.run().dynamic
    ref = compute_dynamic_model([2, 0, -1000, 0, -5000], MODEL_INPUTS, 0, 0, 0, 0, 0, G_EARTH, True,
                                [i/32 for i in range(1, 33)], [-5000, 5000], [-6000, -10], [500, 500])
    assert len(dyn) == len(ref) == 32
    for a, b in zip(dyn, ref):
        assert_same(a, b)
        assert a.metadata == b.metadata
    assert dyn.metadata['scenario']['name'] == 'Vertical dike, opening only'

    raw = small(dike_dict())
    raw['cracks'][0] = {'upper_tip': ['-500 m', '-1000 m'], 'lower_tip': ['500 m', '-2732.0508 m'],
                        'sigma_I': '2 MPa', 'sigma_II': '10 MPa', 'sigma_III': '5 MPa'}
    raw['remote_stresses'] = {'xx': '-3 MPa', 'yy': '-1 MPa'}
    raw['analysis'] = {'type': 'dynamic', 'steps': 10}
    dyn = dies.Scenario.from_dict(raw).run().dynamic
    ref = compute_dynamic_model([0, -500, -1000, 500, -2732.0508], MODEL_INPUTS, 2e6, 10e6, 5e6, -3e6, -1e6,
                                G_EARTH, True, [i/10 for i in range(1, 11)], [-5000, 5000], [-6000, -10],
                                [500, 500])
    for a, b in zip(dyn, ref):
        assert_same(a, b)


def test_two_layers():
    """Layers from the top down become the lists and layer_tops of the core functions."""
    raw = small(dike_dict())
    raw['layers'].append({'name': 'basement', 'top': '-2 km', 'E': '50 GPa', 'nu': 0.3,
                          'density': '2700 kg/m3', 'friction': 0.6, 'T': '-3 MPa'})
    s = dies.Scenario.from_dict(raw)
    assert s.model_inputs() == [[30e9, 50e9], [0.25, 0.3], [2900, 2700], [0.57, 0.6], [-1.5e6, -3e6]]
    assert s.layer_tops() == [0, -2000]
    run = s.run()
    # The crack crosses the boundary: its aperture uses E and v of the layer at its centre
    # (y = -3000 m, in the basement)
    assert run.crack.x() == pytest.approx(2*G(50e9, 0.3)/(4000*0.7))
    m = model([-5000, 5000], [-6000, -10], 500, 500, *layered_inputs(s.model_inputs(), [0, -2000]), G_EARTH)
    assert_same(run.static, m.compute(run.crack, 0, 0, True))
    ref = compute_dynamic_model([2, 0, -1000, 0, -5000], s.model_inputs(), 0, 0, 0, 0, 0, G_EARTH, True,
                                s.investigation_range(), [-5000, 5000], [-6000, -10], [500, 500], [0, -2000])
    for a, b in zip(run.dynamic, ref):
        assert_same(a, b)


# ---------------------------------------------------------------- output

def test_write(tmp_path):
    raw = small(dike_dict())
    raw['output']['figures']['static'] = 'all'
    raw['output']['dpi'] = 50
    s = dies.Scenario.from_dict(raw, source=DIKE)
    run = s.run()
    written = [os.path.basename(p) for p in run.write(str(tmp_path))]
    steps = ['dynamic_step_%03d.png' % k for k in (4, 8, 12, 16, 20, 24, 28, 32)]
    assert written == (['scenario.yaml', 'static_results.npz', 'static_results.csv', 'dynamic_results.npz',
                        'dynamic_final_step.csv'] + [name + '.png' for name in STATIC_FIGURES] + steps)
    # The files reopen with the classes of DiES and hold the same values
    assert_same(DiESResults.load(str(tmp_path / 'static_results.npz')), run.static)
    dyn = DynamicResults.load(str(tmp_path / 'dynamic_results.npz'))
    assert len(dyn) == 32 and dyn.metadata['scenario']['analysis']['step'] == 0.25e6
    header = (tmp_path / 'dynamic_final_step.csv').read_text().splitlines()[0]
    assert header.startswith('x [m],y [m],failure_sn [MPa]') and header.endswith('failure_step []')
    assert (tmp_path / 'scenario.yaml').read_text() == open(DIKE).read()


def test_write_from_a_dict_and_pdf(tmp_path):
    raw = small(dike_dict())
    raw['analysis'] = {'type': 'static'}
    raw['output'] = {'results': ['csv'], 'figures': {'static': ['s1']}, 'format': 'pdf'}
    written = [os.path.basename(p) for p in dies.Scenario.from_dict(raw).run().write(str(tmp_path))]
    assert written == ['scenario.json', 'static_results.csv', 's1.pdf']
    assert json.loads((tmp_path / 'scenario.json').read_text()) == raw


def test_plane_angle_default_is_the_optimal_angle():
    raw = small(dike_dict())
    raw['analysis'] = {'type': 'static'}
    raw['output'] = {}
    run = dies.Scenario.from_dict(raw).run()
    # 90 - coula, coula = 0.5*arctan(1/0.57)
    assert run.plane_angle() == round(90 - np.degrees(0.5*np.arctan(1/0.57)), 2) == 59.84
