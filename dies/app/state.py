"""
The state of the forms of the DiES app, and its conversion to and from a scenario
(Phase 4, Nov 2026). No Streamlit here, so that it can be tested with pytest and
reused by another interface.

The forms show plain numbers in fixed display units (m, MPa, GPa, kg/m3, m/s2, deg);
the scenario keeps every quantity as a text with its unit ('30 GPa'), as in the YAML
files (docs/scenario.md). The "form" is a plain dict:

    name, description, output        kept from the file opened (the app does not edit them)
    analysis_type                    'static' or 'dynamic', as in the file opened (for saving)
    grid: x_min, x_max, y_base, y_top, cell_x, cell_y        (m)
    gravity                          (m/s2), any value
    remote_xx, remote_yy             (MPa, tension positive)
    lithostatic                      bool
    layers: list of {name, top (m), E (GPa), nu, density (kg/m3), friction, T (MPa)}
    crack: {name, upper_x, upper_y, lower_x, lower_y (m), aperture (m),
            sigma_I, sigma_II, sigma_III (MPa)}; in aperture and sigma_I, 0 means
            "not given" (opening_entries)
    dynamic: {increment: 'step' or 'steps', step (MPa), steps}
    plane_angle                      (deg) or None: only the label of the plane figures

An empty value is None. form_from_raw reads a scenario dict (even one with problems:
what cannot be read is left empty); raw_from_form builds the scenario dict back, which
dies.Scenario.from_dict checks.
"""

import copy
import json
import math
import os

import yaml

from dies.scenario import Scenario, ScenarioError, parse_quantity, UNITS


# ------------------------------------------------------------- display units

# Display unit of each kind of value in the forms: (kind of quantity in the scenario,
# unit written, factor from the display unit to the units of the core)
DISPLAY = {
    'length': ('length', 'm', 1.0),
    'stress': ('stress', 'MPa', 1e6),
    'modulus': ('stress', 'GPa', 1e9),
    'density': ('density', 'kg/m3', 1.0),
    'acceleration': ('acceleration', 'm/s2', 1.0),
    'angle': ('angle', 'deg', 1.0),
}

# Columns of the layer table: (key, display kind or None for plain numbers and texts, header)
LAYER_COLUMNS = [
    ('name', 'text', 'Name'),
    ('top', 'length', 'Top y (m)'),
    ('E', 'modulus', 'E (GPa)'),
    ('nu', None, 'ν'),
    ('density', 'density', 'ρ (kg/m³)'),
    ('friction', None, 'μ′ (friction)'),
    ('T', 'stress', 'T (MPa)'),
]

# Columns of the crack table (one row)
CRACK_COLUMNS = [
    ('name', 'text', 'Name'),
    ('upper_x', 'length', 'Upper tip x (m)'),
    ('upper_y', 'length', 'Upper tip y (m)'),
    ('lower_x', 'length', 'Lower tip x (m)'),
    ('lower_y', 'length', 'Lower tip y (m)'),
    ('aperture', 'length', 'Aperture (m)'),
    ('sigma_I', 'stress', 'σI (MPa)'),
    ('sigma_II', 'stress', 'σII (MPa)'),
    ('sigma_III', 'stress', 'σIII (MPa)'),
]


def is_blank(value):
    """True for an empty cell or field: None, NaN or an empty text."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def to_display(value, kind):
    """
    A scenario value with its unit ('5 km') as a number in the display unit of kind
    (5000.0 for 'length'). Plain numbers (kind None) pass as floats. Returns None if
    the value is missing or cannot be read: the problem is reported when the scenario
    is checked.
    """
    if is_blank(value):
        return None
    if kind is None:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if kind == 'text':
        return str(value)
    quantity_kind, _, factor = DISPLAY[kind]
    try:
        return parse_quantity(value, quantity_kind)/factor
    except ScenarioError:
        return None


def to_quantity(number, kind):
    """
    A number of the forms as a scenario value: '5000 m' for (5000, 'length').
    Plain numbers (kind None) and texts pass unchanged; an empty value gives None
    (the key is then left out of the scenario, which reports it as missing).
    """
    if is_blank(number):
        return None
    if kind is None:
        # Whole numbers stay whole (e.g. a friction of 1, not 1.0), as in a YAML file
        number = float(number)
        return int(number) if number.is_integer() else number
    if kind == 'text':
        return str(number)
    _, unit, _ = DISPLAY[kind]
    # 12 significant digits: enough to give back the number typed, without float noise
    return '%.12g %s' % (float(number), unit)


# --------------------------------------------------------- scenario -> form

def _pair(raw, index, kind):
    # One value of a pair such as grid.x: [-5 km, 5 km]
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return to_display(raw[index], kind)
    return None


def form_from_raw(raw):
    """
    The form of a scenario dict (as read from a YAML or JSON file). Tolerant: a value
    that is missing or cannot be read is left empty (None), so that a file with problems
    can still be opened and fixed in the app.
    """
    raw = raw if isinstance(raw, dict) else {}
    grid = raw.get('grid') if isinstance(raw.get('grid'), dict) else {}
    remote = raw.get('remote_stresses') if isinstance(raw.get('remote_stresses'), dict) else {}
    analysis = raw.get('analysis') if isinstance(raw.get('analysis'), dict) else {}

    form = {
        'name': raw.get('name', 'DiES scenario'),
        'description': raw.get('description', ''),
        'output': copy.deepcopy(raw.get('output')) if isinstance(raw.get('output'), dict) else None,
        'analysis_type': analysis.get('type') if analysis.get('type') in ('static', 'dynamic') else 'static',
        'grid': {
            'x_min': _pair(grid.get('x'), 0, 'length'), 'x_max': _pair(grid.get('x'), 1, 'length'),
            'y_base': _pair(grid.get('y'), 0, 'length'), 'y_top': _pair(grid.get('y'), 1, 'length'),
            'cell_x': _pair(grid.get('cell'), 0, 'length'), 'cell_y': _pair(grid.get('cell'), 1, 'length'),
        },
        'gravity': to_display(raw.get('gravity'), 'acceleration'),
        'remote_xx': to_display(remote.get('xx', '0 MPa'), 'stress'),
        'remote_yy': to_display(remote.get('yy', '0 MPa'), 'stress'),
        'lithostatic': raw.get('lithostatic', True) is True,
        'plane_angle': to_display(analysis.get('plane_angle'), 'angle'),
    }

    # Layers: one row per layer, from the top down
    layers = raw.get('layers') if isinstance(raw.get('layers'), list) else []
    form['layers'] = []
    for i, layer in enumerate(layers):
        layer = layer if isinstance(layer, dict) else {}
        row = {key: to_display(layer.get(key), kind) for key, kind, _ in LAYER_COLUMNS}
        row['name'] = row['name'] or 'layer ' + str(i + 1)
        form['layers'].append(row)

    # The crack (one for now): tips, opening (aperture or sigma_I) and the other two
    # driving stresses (0 by default, as in the scenario)
    cracks = raw.get('cracks') if isinstance(raw.get('cracks'), list) else []
    item = cracks[0] if cracks and isinstance(cracks[0], dict) else {}
    form['crack'] = {
        'name': item.get('name', 'crack 1'),
        'upper_x': _pair(item.get('upper_tip'), 0, 'length'), 'upper_y': _pair(item.get('upper_tip'), 1, 'length'),
        'lower_x': _pair(item.get('lower_tip'), 0, 'length'), 'lower_y': _pair(item.get('lower_tip'), 1, 'length'),
        # The one not given is shown as 0 (in the table, 0 means "not given"; see raw_from_form)
        'aperture': to_display(item.get('aperture', '0 m'), 'length'),
        'sigma_I': to_display(item.get('sigma_I', '0 MPa'), 'stress'),
        'sigma_II': to_display(item.get('sigma_II', '0 MPa'), 'stress'),
        'sigma_III': to_display(item.get('sigma_III', '0 MPa'), 'stress'),
    }

    # Dynamic analysis: the driving stress added at each step, or the number of steps.
    # A static file gets the default of the tests (0.25 MPa per step)
    if 'steps' in analysis and 'step' not in analysis:
        steps = analysis['steps'] if isinstance(analysis['steps'], int) else None
        form['dynamic'] = {'increment': 'steps', 'step': 0.25, 'steps': steps}
    else:
        step = to_display(analysis.get('step', '0.25 MPa'), 'stress')
        form['dynamic'] = {'increment': 'step', 'step': step, 'steps': 32}
    return form


# --------------------------------------------------------- form -> scenario

def _put(target, key, value):
    # Adds key only if the value is not empty, so that a missing value is reported
    # by the scenario check as missing
    if value is not None:
        target[key] = value


def _quantity_pair(a, b, kind):
    # A pair such as grid.x; None if either value is empty
    qa, qb = to_quantity(a, kind), to_quantity(b, kind)
    return None if qa is None or qb is None else [qa, qb]


def opening_entries(aperture, sigma_I):
    """
    Which of the two ways of giving the opening goes to the scenario, as (key, kind)
    pairs. In the table, an empty cell or a 0 means "not given" (27 Nov 2026: Sam typed 0
    in the one he did not want, and the check took it as given):
      - aperture > 0 (or negative, so that the check reports it) and sigma_I 0 or empty:
        the aperture;
      - aperture 0 or empty: sigma_I if it is not 0; if both are 0 or empty, neither
        (the scenario then takes sigma_I = 0, e.g. a fault with shear only);
      - both non-zero: both, and the check reports that only one can be given.
    """
    has_aperture = not is_blank(aperture) and float(aperture) != 0
    has_sigma = not is_blank(sigma_I) and float(sigma_I) != 0
    if has_aperture and has_sigma:
        return [('aperture', 'length'), ('sigma_I', 'stress')]
    if has_aperture:
        return [('aperture', 'length')]
    if has_sigma:
        return [('sigma_I', 'stress')]
    return []


def raw_from_form(form, analysis='static', include_output=False):
    """
    The scenario dict of a form, ready for Scenario.from_dict or to be saved as YAML.

    analysis: 'static' or 'dynamic' (with the step or the number of steps of the form).
    include_output: False for the scenarios the app computes (the app draws its own
    figures and writes on demand); True when saving, to keep the output section of the
    file opened (without the dynamic figures if the analysis saved is static).
    """
    g = form['grid']
    raw = {'name': form.get('name') or 'DiES scenario'}
    if form.get('description'):
        raw['description'] = form['description']

    grid = {}
    _put(grid, 'x', _quantity_pair(g['x_min'], g['x_max'], 'length'))
    _put(grid, 'y', _quantity_pair(g['y_base'], g['y_top'], 'length'))
    _put(grid, 'cell', _quantity_pair(g['cell_x'], g['cell_y'], 'length'))
    raw['grid'] = grid
    _put(raw, 'gravity', to_quantity(form['gravity'], 'acceleration'))

    # Layers, from the top down; empty rows (e.g. one just added) are skipped
    layers = []
    for row in form['layers']:
        if all(is_blank(row.get(key)) for key, _, _ in LAYER_COLUMNS if key != 'name'):
            continue
        layer = {}
        for key, kind, _ in LAYER_COLUMNS:
            _put(layer, key, to_quantity(row.get(key), kind))
        layers.append(layer)
    raw['layers'] = layers

    # The crack: tips and driving stresses
    c = form['crack']
    item = {'name': c.get('name') or 'crack 1'}
    _put(item, 'upper_tip', _quantity_pair(c['upper_x'], c['upper_y'], 'length'))
    _put(item, 'lower_tip', _quantity_pair(c['lower_x'], c['lower_y'], 'length'))
    for key, kind in opening_entries(c.get('aperture'), c.get('sigma_I')):
        item[key] = to_quantity(c[key], kind)
    for key in ('sigma_II', 'sigma_III'):
        _put(item, key, to_quantity(c.get(key), 'stress'))
    raw['cracks'] = [item]

    raw['remote_stresses'] = {}
    _put(raw['remote_stresses'], 'xx', to_quantity(form['remote_xx'], 'stress'))
    _put(raw['remote_stresses'], 'yy', to_quantity(form['remote_yy'], 'stress'))
    raw['lithostatic'] = bool(form['lithostatic'])

    # Analysis: a dynamic one adds the step (MPa) or the number of steps
    section = {'type': analysis}
    if analysis == 'dynamic':
        d = form['dynamic']
        if d['increment'] == 'steps':
            _put(section, 'steps', None if is_blank(d['steps']) else int(d['steps']))
        else:
            _put(section, 'step', to_quantity(d['step'], 'stress'))
    _put(section, 'plane_angle', to_quantity(form.get('plane_angle'), 'angle'))
    raw['analysis'] = section

    # Output section of the file opened, only when saving
    if include_output and form.get('output'):
        output = copy.deepcopy(form['output'])
        if analysis == 'static' and isinstance(output.get('figures'), dict):
            output['figures'].pop('dynamic', None)
        raw['output'] = output
    return raw


# ---------------------------------------------------------------- checking

def check(raw):
    """(Scenario, []) if the scenario is valid, or (None, problems) with every problem found."""
    try:
        return Scenario.from_dict(raw), []
    except ScenarioError as error:
        return None, list(error.problems)


def split_problem(problem):
    """'layers[0].E: must be ...' -> ('layers[0].E', 'must be ...')."""
    where, sep, message = problem.partition(': ')
    return (where, message) if sep else ('', problem)


# Sections of the forms where the problems are shown: the first part of the place
SECTIONS = ('grid', 'gravity', 'remote_stresses', 'lithostatic', 'layers', 'cracks', 'analysis', 'output')


def problem_section(problem):
    """The section of the forms a problem belongs to (one of SECTIONS), or 'general'."""
    where, _ = split_problem(problem)
    head = where.split('.')[0].split('[')[0]
    return head if head in SECTIONS else 'general'


# Readable names of the places in the scenario, for the messages of the forms
_LAYER_HEADERS = {key: header for key, _, header in LAYER_COLUMNS}
_CRACK_FIELDS = {'upper_tip': 'upper tip', 'lower_tip': 'lower tip', 'aperture': 'aperture', 'sigma_I': 'σI',
                 'sigma_II': 'σII', 'sigma_III': 'σIII', 'name': 'name'}
_PLACES = {'grid.x': 'x limits', 'grid.y': 'y limits (base, top)', 'grid.cell': 'cell size',
           'remote_stresses.xx': 'σxx remote', 'remote_stresses.yy': 'σyy remote',
           'analysis.step': 'step', 'analysis.steps': 'number of steps', 'analysis.plane_angle': 'plane angle'}


def describe_problem(problem):
    """
    A problem in the words of the forms: 'layers[1].E: must be ...' becomes
    'Layer 2, E (GPa): must be ...'; 'grid.cell[0]: ...' becomes 'cell size: ...'.
    The units in the messages are those of the scenario (e.g. '-1.5 MPa').
    """
    where, message = split_problem(problem)
    parts = where.replace('[', '.[').split('.')
    if parts and parts[0] == 'layers' and len(parts) > 1 and parts[1].startswith('['):
        index = int(parts[1].strip('[]'))
        field = parts[2] if len(parts) > 2 else None
        return 'Layer ' + str(index + 1) + (', ' + _LAYER_HEADERS.get(field, field) if field else '') + ': ' + message
    if parts and parts[0] == 'cracks':
        field = parts[2] if len(parts) > 2 else None
        if message.startswith('give either aperture or sigma_I'):
            return 'Crack: give the aperture or σI, and 0 in the other one'
        return 'Crack' + (', ' + _CRACK_FIELDS.get(field, field) if field else '') + ': ' + message
    # Pairs (grid.x[0]) are named by the pair
    base = where.split('[')[0]
    if base in _PLACES:
        return _PLACES[base] + ': ' + message
    return problem


# --------------------------------------------------------------- cache key

def scenario_key(scenario):
    """
    The key of a computation: the checked scenario in the units of the core, as sorted
    JSON, without the name, the description, the output section and the plane angle (a
    label), which do not change the results. '5 km' and '5000 m' give the same key.
    """
    data = scenario.to_dict()
    for key in ('name', 'description', 'output'):
        data.pop(key, None)
    # The plane angle is only a label of the plane figures (Phase 4, step 2)
    if isinstance(data.get('analysis'), dict):
        data['analysis'].pop('plane_angle', None)
    return json.dumps(data, sort_keys=True)


# ----------------------------------------------------------- files

def read_scenario_text(text, filename):
    """
    The dict of a scenario file given as text: YAML (.yaml, .yml) or JSON (.json).
    Raises ScenarioError if the file cannot be read (the values are not checked here).
    """
    extension = os.path.splitext(filename)[1].lower()
    try:
        raw = json.loads(text) if extension == '.json' else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as error:
        raise ScenarioError(filename + ': cannot be read (' + str(error).splitlines()[0] + ')')
    if not isinstance(raw, dict):
        raise ScenarioError(filename + ': expected a scenario (a mapping with grid, gravity, layers...)')
    return raw


class _ScenarioDumper(yaml.SafeDumper):
    """YAML writer that puts short lists of values on one line ([-5000 m, 5000 m]), as in the examples."""


def _represent_list(dumper, values):
    # Lists of plain values (pairs, lists of figure names) in flow style; lists of
    # sections (layers, cracks) in block style
    flow = all(not isinstance(v, (dict, list)) for v in values)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', values, flow_style=flow)


_ScenarioDumper.add_representer(list, _represent_list)


def scenario_yaml(raw):
    """A scenario dict as YAML text, in the order of the sections (as the examples)."""
    header = '# DiES scenario, saved from the DiES app. Format: docs/scenario.md\n'
    return header + yaml.dump(raw, Dumper=_ScenarioDumper, sort_keys=False, allow_unicode=True, width=1000)


def cohesion(T):
    """The implicit cohesion C = 2|T| (MPa) of a tensile strength T (MPa), or None."""
    return None if is_blank(T) else 2*abs(float(T))


# Unit kinds of the scenario, re-exported for the tests and the interface
SCENARIO_UNITS = UNITS
