"""
Scenarios: a complete DiES model described in one YAML (or JSON) file (Phase 3, Nov 2026).

A scenario holds the grid, the gravity, the layers, the crack, the loading, the analysis
(static or dynamic) and what to write. DiES reads it, checks it, and runs it with the
functions of the core, which are not modified:

    import dies
    scenario = dies.load_scenario('examples/dike_example.yaml')
    run = scenario.run()          # ScenarioResults: run.static (DiESResults), run.dynamic
    run.write('results/')         # results, figures and a copy of the scenario

or, from a terminal, `dies run examples/dike_example.yaml --out results/` (dies/cli.py).

Format (full description in docs/scenario.md; example in examples/dike_example.yaml):

- Every quantity with dimensions carries its unit ('30 GPa', '-1.5 MPa', '5 km',
  '2900 kg/m3', '9.81 m/s2', '60 deg'). A number without a unit is rejected, so that a
  forgotten factor of 1e6 cannot pass. nu and friction are plain numbers.
- Layers from the top down, each with its `top`; the first one starts at the surface.
- One crack (in a list): a dike gives its `aperture` or `sigma_I` (one of the two); any
  crack may add `sigma_II` and `sigma_III`.
- Analysis `static` or `dynamic`; a dynamic analysis gives the `step` (driving stress
  added at each step) or the number of `steps`.
- All the problems found are reported together, in one ScenarioError.

Internally every value is converted to the units of the core (m, Pa, kg m-3, m s-2,
degrees) and the model is built with layered_inputs and model(), as single_model and
compute_dynamic_model do. The results are identical to those of the core functions
called with the same values (tests/test_scenario.py).
"""

import copy
import difflib
import json
import math
import os
import re
import shutil

import numpy as np
import matplotlib.pyplot as plt

from DiES_core import crack, model, InputError
from Modeling_functions import layered_inputs, crack_layer_properties, compute_dynamic_model, compute_failure_state
from Helper_functions import G, points_distance
from results import DiESResults
import plotting


class ScenarioError(InputError):
    """
    A scenario that cannot be read or is not valid. The message lists every problem
    found, one per line, each with the place in the file (e.g. 'layers[0].E: ...').
    It is an InputError, so code that catches InputError catches it too.
    """

    def __init__(self, problems):
        # A single message, or a list of them (one per problem)
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        if len(self.problems) == 1:
            message = 'Invalid scenario: ' + self.problems[0]
        else:
            message = 'Invalid scenario (' + str(len(self.problems)) + ' problems):\n  ' + '\n  '.join(self.problems)
        super().__init__(message)

    def __str__(self):
        return self.message


# --------------------------------------------------------------------- units

# Accepted units of each kind of quantity, and their factor to the units of the core.
# Unit names are case sensitive (MPa is not mPa).
UNITS = {
    'length': {'m': 1.0, 'km': 1e3, 'cm': 1e-2, 'mm': 1e-3},
    'stress': {'Pa': 1.0, 'kPa': 1e3, 'MPa': 1e6, 'GPa': 1e9},
    'density': {'kg/m3': 1.0, 'kg/m^3': 1.0, 'kg m-3': 1.0, 'g/cm3': 1e3, 'g/cm^3': 1e3},
    'acceleration': {'m/s2': 1.0, 'm/s^2': 1.0, 'm s-2': 1.0},
    'angle': {'deg': 1.0, '°': 1.0, 'rad': 180/math.pi},
}
# Units of the core, in which the scenario keeps every value once read
CORE_UNITS = {'length': 'm', 'stress': 'Pa', 'density': 'kg m-3', 'acceleration': 'm s-2', 'angle': 'deg'}
# Examples for the error messages
_EXAMPLES = {'length': "'5 km'", 'stress': "'-1.5 MPa'", 'density': "'2900 kg/m3'",
             'acceleration': "'9.81 m/s2'", 'angle': "'60 deg'"}
# A number (with optional sign, decimals and exponent) followed by its unit
_QUANTITY = re.compile(r'^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$')


def parse_quantity(value, kind, where='value'):
    """
    Converts a quantity written with its unit ('30 GPa', '-1.5 MPa', '5 km') into a
    float in the units of the core (CORE_UNITS: m, Pa, kg m-3, m s-2, degrees).

    kind: 'length', 'stress', 'density', 'acceleration' or 'angle'.
    where: the place of the value in the scenario, for the error message.
    Raises ScenarioError if the value has no unit, an unknown unit or one of another kind.
    """
    accepted = UNITS[kind]
    example = _EXAMPLES[kind]
    # A bare number: the unit is missing (the error that the units are there to catch)
    if isinstance(value, bool) or not isinstance(value, str):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raise ScenarioError(where + ': ' + repr(value) + ' has no unit; write it with its unit, e.g. ' + example)
        raise ScenarioError(where + ': expected a ' + kind + ' with its unit, e.g. ' + example + ', got ' + repr(value))
    match = _QUANTITY.match(value)
    if match is not None and not match.group(2):
        # A number written as text (YAML reads '30e9' as text): the unit is missing too
        raise ScenarioError(where + ": '" + value + "' has no unit; write it with its unit, e.g. " + example)
    if match is None:
        raise ScenarioError(where + ": '" + value + "' is not a number followed by its unit, e.g. " + example)
    number, unit = float(match.group(1)), match.group(2)
    if unit not in accepted:
        # A unit of another kind (e.g. '30 m' for a stress) gets a more specific message
        other = [k for k, units in UNITS.items() if unit in units]
        hint = (' (a unit of ' + other[0] + ', but a ' + kind + ' is expected)') if other else ''
        raise ScenarioError(where + ": unknown unit '" + unit + "'" + hint + '; use one of: ' + ', '.join(accepted))
    return number*accepted[unit]


# ------------------------------------------------------- reading and checking

class _Checker(object):
    """
    Collects the problems found while reading a scenario, so that all of them are
    reported together, and provides the small checks used by _parse.
    Each check returns the value read, or None if it is not valid.
    """

    def __init__(self):
        self.problems = []

    def problem(self, where, message):
        self.problems.append(where + ': ' + message)

    def quantity(self, value, kind, where):
        # A value with its unit, converted to the units of the core
        try:
            return parse_quantity(value, kind, where)
        except ScenarioError as error:
            self.problems.extend(error.problems)
            return None

    def number(self, value, where):
        # A plain number (dimensionless): a unit here is a mistake
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.problem(where, 'expected a plain number (it is dimensionless), got ' + repr(value))
            return None
        return float(value)

    def integer(self, value, where, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            self.problem(where, 'expected a whole number >= ' + str(minimum) + ', got ' + repr(value))
            return None
        return value

    def boolean(self, value, where):
        if not isinstance(value, bool):
            self.problem(where, 'expected true or false, got ' + repr(value))
            return None
        return value

    def text(self, value, where):
        if not isinstance(value, str):
            self.problem(where, 'expected a text, got ' + repr(value))
            return None
        return value

    def pair(self, value, kind, where):
        # Two quantities of the same kind, e.g. [-5 km, 5 km]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            self.problem(where, 'expected a list of two values, e.g. [' + _EXAMPLES[kind].strip("'") + ', '
                         + _EXAMPLES[kind].strip("'") + '], got ' + repr(value))
            return None
        values = [self.quantity(v, kind, where + '[' + str(i) + ']') for i, v in enumerate(value)]
        return None if None in values else values

    def mapping(self, value, where, required=(), optional=()):
        """
        A section of the scenario: a mapping with known keys. Unknown keys are reported
        (with the closest known one, to catch typos) and so are missing required keys.
        Returns the mapping, or an empty dict if it is not one. A section that is not in
        the file (_ABSENT) gives an empty dict silently: its absence is reported, if it is
        required, by the section that contains it.
        """
        if value is _ABSENT:
            return {}
        if not isinstance(value, dict):
            self.problem(where or 'scenario', 'expected a section with keys (' + ', '.join(tuple(required) + tuple(optional))
                         + '), got ' + repr(value))
            return {}
        known = tuple(required) + tuple(optional)
        for key in value:
            if key not in known:
                close = difflib.get_close_matches(str(key), known, n=1)
                hint = ("; did you mean '" + close[0] + "'?") if close else ''
                self.problem(where + '.' + str(key) if where else str(key),
                             'unknown key' + hint + ' Valid keys: ' + ', '.join(known))
        for key in required:
            if key not in value:
                self.problem(where + '.' + key if where else key, 'missing')
        return value


# Marks a section that is not in the file (different from one that is empty)
_ABSENT = object()


def _join(where, key):
    # Place of a key in the scenario, e.g. 'layers[0].E'
    return where + '.' + key if where else key


def _parse(raw):
    """
    Reads and checks a scenario (the dict loaded from the file) and returns it with
    every value in the units of the core and the defaults filled in. Raises one
    ScenarioError with every problem found.
    """
    check = _Checker()
    top = check.mapping(raw, '', required=('grid', 'gravity', 'layers', 'cracks', 'analysis'),
                        optional=('name', 'description', 'remote_stresses', 'lithostatic', 'output'))
    data = {}

    # Name and description, only kept in the metadata of the results
    data['name'] = check.text(top.get('name', 'DiES scenario'), 'name')
    data['description'] = check.text(top.get('description', ''), 'description')

    # Grid: limits along x and y and the cell size, in m. y is negative downwards
    # and the top of the model must be below the surface (as model() expects)
    grid = check.mapping(top.get('grid', _ABSENT), 'grid', required=('x', 'y', 'cell'))
    x = check.pair(grid.get('x'), 'length', 'grid.x') if 'x' in grid else None
    y = check.pair(grid.get('y'), 'length', 'grid.y') if 'y' in grid else None
    cell = check.pair(grid.get('cell'), 'length', 'grid.cell') if 'cell' in grid else None
    if x is not None and x[0] >= x[1]:
        check.problem('grid.x', 'the first limit must be smaller than the second')
    if y is not None:
        if y[0] >= y[1]:
            check.problem('grid.y', 'the first limit (the base) must be smaller than the second (the top)')
        if y[1] >= 0:
            check.problem('grid.y', 'the top of the model must be below the surface (y < 0), e.g. -10 m')
    if cell is not None and min(cell) <= 0:
        check.problem('grid.cell', 'the cell sizes must be larger than 0')
    data['grid'] = {'x': x, 'y': y, 'cell': cell}

    # Gravity (9.81 m/s2 on Earth, 3.71 m/s2 on Mars); no default, to choose it knowingly
    data['gravity'] = check.quantity(top['gravity'], 'acceleration', 'gravity') if 'gravity' in top else None
    if data['gravity'] is not None and data['gravity'] < 0:
        check.problem('gravity', 'must be 0 or larger')

    # Layers, from the top down; each one goes from its top to the top of the next one,
    # and the last one down to the base of the model and below
    layers = []
    raw_layers = top.get('layers', [])
    if 'layers' in top and (not isinstance(raw_layers, list) or not raw_layers):
        check.problem('layers', 'expected a list with at least one layer')
        raw_layers = []
    for i, raw_layer in enumerate(raw_layers):
        where = 'layers[' + str(i) + ']'
        layer = check.mapping(raw_layer, where, required=('top', 'E', 'nu', 'density', 'friction', 'T'),
                              optional=('name',))
        parsed = {'name': check.text(layer.get('name', 'layer ' + str(i + 1)), _join(where, 'name'))}
        kinds = (('top', 'length'), ('E', 'stress'), ('nu', None), ('density', 'density'),
                 ('friction', None), ('T', 'stress'))
        for key, kind in kinds:
            if key not in layer:
                parsed[key] = None
            elif kind is None:
                parsed[key] = check.number(layer[key], _join(where, key))
            else:
                parsed[key] = check.quantity(layer[key], kind, _join(where, key))
        # Physical limits of each property (T is always negative: Sam's convention)
        limits = (('E', lambda v: v > 0, 'must be larger than 0'),
                  ('nu', lambda v: 0 <= v < 0.5, 'must be between 0 and 0.5 (0.5 excluded)'),
                  ('density', lambda v: v > 0, 'must be larger than 0'),
                  ('friction', lambda v: v > 0, 'must be larger than 0'),
                  ('T', lambda v: v < 0, 'must be negative (tension is negative on the Mohr diagram), e.g. -1.5 MPa'))
        for key, valid, message in limits:
            if parsed[key] is not None and not valid(parsed[key]):
                check.problem(_join(where, key), message)
        layers.append(parsed)
    tops = [layer['top'] for layer in layers]
    if layers and None not in tops:
        # The first layer starts at the surface: the lithostatic load of the core
        # takes the uppermost layer up to y = 0, so any other top would be misleading
        if tops[0] != 0:
            check.problem('layers[0].top', 'the first layer must start at the surface (top: 0 m)')
        if any(b >= a for a, b in zip(tops, tops[1:])):
            check.problem('layers', 'the tops must go from the top down (each one deeper than the previous one)')
        if y is not None and any(t <= y[0] for t in tops[1:]):
            check.problem('layers', 'a layer starts at or below the base of the model (' + str(y[0]) + ' m)')
    data['layers'] = layers

    # Cracks: one for now, in a list (superposing several is in the long-term roadmap)
    cracks = []
    raw_cracks = top.get('cracks', [])
    if 'cracks' in top and (not isinstance(raw_cracks, list) or len(raw_cracks) != 1):
        check.problem('cracks', 'expected a list with one crack (several cracks are not supported yet)')
        raw_cracks = raw_cracks[:1] if isinstance(raw_cracks, list) else []
    for i, raw_crack in enumerate(raw_cracks):
        where = 'cracks[' + str(i) + ']'
        item = check.mapping(raw_crack, where, required=('upper_tip', 'lower_tip'),
                             optional=('name', 'aperture', 'sigma_I', 'sigma_II', 'sigma_III'))
        parsed = {'name': check.text(item.get('name', 'crack ' + str(i + 1)), _join(where, 'name'))}
        for key in ('upper_tip', 'lower_tip'):
            parsed[key] = check.pair(item[key], 'length', _join(where, key)) if key in item else None
            if parsed[key] is not None and parsed[key][1] > 0:
                check.problem(_join(where, key), 'the tip must be at or below the surface (y <= 0)')
        if parsed['upper_tip'] is not None and parsed['upper_tip'] == parsed['lower_tip']:
            check.problem(where, 'the two tips are the same point')
        # Opening: the total aperture (dikes) or the opening driving stress, not both,
        # since one follows from the other (eq. 2 of Rivas-Dorado et al., 2023)
        if 'aperture' in item and 'sigma_I' in item:
            check.problem(where, 'give either aperture or sigma_I, not both (one follows from the other)')
        parsed['aperture'] = check.quantity(item['aperture'], 'length', _join(where, 'aperture')) \
            if 'aperture' in item else None
        if parsed['aperture'] is not None and parsed['aperture'] <= 0:
            check.problem(_join(where, 'aperture'), 'must be larger than 0 (for a closing crack, give a negative sigma_I)')
        # Driving stresses (Pa): sigma_I opening (> 0) or closing (< 0); sigma_II slip,
        # > 0 normal sense (hanging wall down-dip); sigma_III out of the X-Y plane
        for key in ('sigma_I', 'sigma_II', 'sigma_III'):
            parsed[key] = check.quantity(item[key], 'stress', _join(where, key)) if key in item else 0.0
        cracks.append(parsed)
    data['cracks'] = cracks

    # Loading: constant remote stresses (tension positive) and the lithostatic load
    remote = check.mapping(top.get('remote_stresses', _ABSENT), 'remote_stresses', optional=('xx', 'yy'))
    data['remote_stresses'] = {key: check.quantity(remote[key], 'stress', 'remote_stresses.' + key)
                               if key in remote else 0.0 for key in ('xx', 'yy')}
    data['lithostatic'] = check.boolean(top.get('lithostatic', True), 'lithostatic')

    # Analysis: static (one evaluation) or dynamic (driving stresses growing step by step)
    analysis = check.mapping(top.get('analysis', _ABSENT), 'analysis', required=('type',),
                             optional=('step', 'steps', 'plane_angle'))
    kind = analysis.get('type')
    if 'type' in analysis and kind not in ('static', 'dynamic'):
        check.problem('analysis.type', "must be 'static' or 'dynamic', got " + repr(kind))
    step = check.quantity(analysis['step'], 'stress', 'analysis.step') if 'step' in analysis else None
    steps = check.integer(analysis['steps'], 'analysis.steps') if 'steps' in analysis else None
    if kind == 'dynamic':
        if ('step' in analysis) == ('steps' in analysis):
            check.problem('analysis', 'a dynamic analysis needs either step (e.g. 0.25 MPa) or steps (e.g. 32), '
                                      'not both')
        if step is not None and step <= 0:
            check.problem('analysis.step', 'must be larger than 0')
    elif kind == 'static':
        for key in ('step', 'steps'):
            if key in analysis:
                check.problem('analysis.' + key, 'only used in a dynamic analysis')
    # The angle written in the label of the figures of the stresses on planes; those
    # stresses are on the optimal planes of the friction (see plotting.plot_plane_stresses)
    plane_angle = check.quantity(analysis['plane_angle'], 'angle', 'analysis.plane_angle') \
        if 'plane_angle' in analysis else None
    if plane_angle is not None and not 0 <= plane_angle <= 90:
        check.problem('analysis.plane_angle', 'must be between 0 and 90 deg')
    data['analysis'] = {'type': kind, 'step': step, 'steps': steps, 'plane_angle': plane_angle}

    # A dynamic analysis needs a driving stress to grow
    if kind == 'dynamic' and cracks and cracks[0]['aperture'] is None and \
            all(cracks[0][key] == 0 for key in ('sigma_I', 'sigma_II', 'sigma_III')):
        check.problem('cracks[0]', 'a dynamic analysis needs a driving stress: give aperture, sigma_I, '
                                   'sigma_II or sigma_III')

    data['output'] = _parse_output(check, top.get('output', _ABSENT), kind)

    if check.problems:
        raise ScenarioError(check.problems)
    return data


def _parse_output(check, raw, kind):
    """The output section: results files, figures, their format and resolution (with defaults)."""
    output = check.mapping(raw, 'output', optional=('results', 'figures', 'format', 'dpi'))
    # Results files: npz (DiESResults.save / DynamicResults.save) and csv (to_csv)
    results = output.get('results', ['npz'])
    if not isinstance(results, list) or any(r not in RESULT_FORMATS for r in results):
        check.problem('output.results', 'expected a list with npz and/or csv, got ' + repr(results))
        results = []
    # Figures of the static model (by name, or 'all') and of the steps of the dynamic model
    figures = check.mapping(output.get('figures', _ABSENT), 'output.figures', optional=('static', 'dynamic'))
    static = figures.get('static', [])
    if static == 'all':
        static = list(STATIC_FIGURES)
    if not isinstance(static, list):
        check.problem('output.figures.static', "expected a list of figure names, or 'all'")
        static = []
    for name in static:
        if name not in STATIC_FIGURES:
            close = difflib.get_close_matches(str(name), list(STATIC_FIGURES), n=1)
            hint = ("; did you mean '" + close[0] + "'?") if close else ''
            check.problem('output.figures.static', 'unknown figure ' + repr(name) + hint
                          + ' Valid figures: ' + ', '.join(STATIC_FIGURES))
    dynamic = None
    if 'dynamic' in figures:
        raw_dynamic = figures['dynamic'] if figures['dynamic'] is not None else {}
        section = check.mapping(raw_dynamic, 'output.figures.dynamic', optional=('show_every', 'downscaling'))
        dynamic = {'show_every': check.integer(section.get('show_every', 4), 'output.figures.dynamic.show_every'),
                   'downscaling': check.integer(section.get('downscaling', 1), 'output.figures.dynamic.downscaling')}
        if kind == 'static':
            check.problem('output.figures.dynamic', 'only used in a dynamic analysis (analysis.type: dynamic)')
    fmt = output.get('format', 'png')
    if fmt not in FIGURE_FORMATS:
        check.problem('output.format', 'must be one of ' + ', '.join(FIGURE_FORMATS) + ', got ' + repr(fmt))
    dpi = check.integer(output.get('dpi', 200), 'output.dpi')
    return {'results': list(results), 'figures': {'static': [s for s in static if s in STATIC_FIGURES],
                                                  'dynamic': dynamic},
            'format': fmt, 'dpi': dpi}


# -------------------------------------------------------------------- figures

def _crack_material(run):
    # E and v of the layer at the centre of the crack (as the core uses them for the aperture)
    E, v = crack_layer_properties(run.model, run.crack.up_tip(), run.crack.low_tip())
    return v, E

# Figures of the static model, by name: (results used, function drawing it). 'fine'
# is the grid of the scenario; 'coarse' is a grid 2.5 times coarser along x and 3
# times along y, used by single_model for the vectors and the trajectories, so that
# the arrows can be read.
STATIC_FIGURES = {
    'ux': ('coarse', lambda res, run: plotting.plot_displacement(res, 'ux', 'vectors')),
    'uy': ('coarse', lambda res, run: plotting.plot_displacement(res, 'uy', 'vectors')),
    'uxy': ('coarse', lambda res, run: plotting.plot_displacement(res, 'uxy', 'vectors')),
    'ux_contours': ('fine', lambda res, run: plotting.plot_displacement(res, 'ux', 'contours')),
    'uy_contours': ('fine', lambda res, run: plotting.plot_displacement(res, 'uy', 'contours')),
    'uxy_contours': ('fine', lambda res, run: plotting.plot_displacement(res, 'uxy', 'contours')),
    'sxx': ('fine', lambda res, run: plotting.plot_stresses(res, 'sxx')),
    'syy': ('fine', lambda res, run: plotting.plot_stresses(res, 'syy')),
    'sxy': ('fine', lambda res, run: plotting.plot_stresses(res, 'sxy')),
    'sxz': ('fine', lambda res, run: plotting.plot_stresses(res, 'sxz')),
    'szz': ('fine', lambda res, run: plotting.plot_stresses(res, 'szz')),
    'sav': ('fine', lambda res, run: plotting.plot_stresses(res, 'sav')),
    'tmax': ('fine', lambda res, run: plotting.plot_stresses(res, 'tmax')),
    's1': ('fine', lambda res, run: plotting.plot_principal_stresses(res, 'principal')),
    's3': ('fine', lambda res, run: plotting.plot_principal_stresses(res, 'secondary')),
    'angle': ('fine', lambda res, run: plotting.plot_principal_stresses(res, 'angle')),
    'trajectories': ('coarse', lambda res, run: plotting.plot_stress_trajectories(res, False)),
    'trajectories_planes': ('coarse', lambda res, run: plotting.plot_stress_trajectories(res, True)),
    'plane_normal': ('fine', lambda res, run: plotting.plot_plane_stresses(res, 'normal', run.plane_angle())),
    'plane_shear': ('fine', lambda res, run: plotting.plot_plane_stresses(res, 'shear', run.plane_angle())),
    'crack_displacements': (None, lambda res, run: plotting.plot_crack_displacements(run.crack, *_crack_material(run))),
    # Failure state of the single model, with the crack loaded at once (Phase 4, Nov 2026)
    'failure_mode': ('failure', lambda res, run: plotting.plot_failure_modes(res)),
    'failure_ratio': ('failure', lambda res, run: plotting.plot_failure_ratio(res)),
}
RESULT_FORMATS = ('npz', 'csv')
FIGURE_FORMATS = ('png', 'pdf', 'svg')


# ------------------------------------------------------------------- scenario

class Scenario(object):
    """
    A checked scenario, with every value in the units of the core (m, Pa, kg m-3,
    m s-2, degrees). Create it with load_scenario(path) or Scenario.from_dict(data).

    The parts of the model, as the core functions take them: model_inputs(),
    layer_tops(), crack_inputs(), driving_stresses(), investigation_range(),
    build_model(). run() computes the model and returns a ScenarioResults.
    """

    def __init__(self, data, raw=None, source=None):
        # data: the checked scenario (from _parse); raw: the dict as read; source: its file
        self.data = data
        self.raw = raw
        self.source = source

    @classmethod
    def from_dict(cls, raw, source=None):
        """Checks a scenario given as a dict (as read from YAML or JSON)."""
        raw = copy.deepcopy(raw)
        return cls(_parse(raw), raw, source)

    def __repr__(self):
        a = self.data['analysis']
        return ('Scenario(' + repr(self.data['name']) + ', ' + str(a['type']) + ', '
                + str(len(self.data['layers'])) + ' layer(s))')

    # Short access to the parts that are used most
    @property
    def name(self):
        return self.data['name']

    @property
    def analysis(self):
        return self.data['analysis']['type']

    @property
    def output(self):
        return self.data['output']

    def to_dict(self):
        """The checked scenario in the units of the core, as a JSON-serialisable dict
        (stored in the metadata of the results)."""
        data = copy.deepcopy(self.data)
        data['units'] = dict(CORE_UNITS)
        return data

    # -------------------------------------------- inputs of the core functions

    def model_inputs(self):
        """[E, v, density, friction, T]: numbers for one layer, or one list per property
        with a value per layer, from the top down (as single_model takes them)."""
        layers = self.data['layers']
        keys = ('E', 'nu', 'density', 'friction', 'T')
        if len(layers) == 1:
            return [layers[0][key] for key in keys]
        return [[layer[key] for layer in layers] for key in keys]

    def layer_tops(self):
        """None for one layer (as single_model takes it), or the top of each layer, in m."""
        layers = self.data['layers']
        return None if len(layers) == 1 else [layer['top'] for layer in layers]

    def crack_inputs(self):
        """[total aperture (0 if not given), x and y of the upper tip, x and y of the lower tip]."""
        c = self.data['cracks'][0]
        return [c['aperture'] or 0.0] + list(c['upper_tip']) + list(c['lower_tip'])

    def build_model(self, coarse=False):
        """
        The model() object of the scenario. coarse=True gives the grid used for the vector
        and trajectory figures (cells 2.5 times larger along x and 3 times along y, as in
        single_model).
        """
        g = self.data['grid']
        cx, cy = g['cell']
        if coarse:
            cx, cy = cx*2.5, cy*3
        return model(g['x'], g['y'], cx, cy, *layered_inputs(self.model_inputs(), self.layer_tops()),
                     self.data['gravity'])

    def driving_stresses(self, current_model=None):
        """
        The final driving stresses of the crack, (sigma_I, sigma_II, sigma_III) in Pa.
        With an aperture, sigma_I follows from it as in single_model and
        compute_dynamic_model: a*G/(b*(1-v)), with E and v of the layer at the centre of
        the crack (eq. 2 of Rivas-Dorado et al., 2023).
        """
        c = self.data['cracks'][0]
        sigma_I = c['sigma_I']
        if c['aperture'] is not None:
            current_model = current_model or self.build_model()
            E, v = crack_layer_properties(current_model, c['upper_tip'], c['lower_tip'])
            b = points_distance(c['upper_tip'], c['lower_tip'])
            sigma_I = c['aperture']*G(E, v)/(b*(1 - v))
        return sigma_I, c['sigma_II'], c['sigma_III']

    def build_crack(self, current_model=None):
        """The crack() object with the final driving stresses."""
        c = self.data['cracks'][0]
        return crack(tuple(c['upper_tip']), tuple(c['lower_tip']), *self.driving_stresses(current_model))

    def investigation_range(self, current_model=None):
        """
        Fractions of the final driving stresses at each step of a dynamic analysis:
        i/n for i = 1..n. With steps, n = steps; with step, n is the number of steps
        that brings the largest driving stress to its final value adding about step at
        each one (rounded, so that the last step reaches the final value exactly; e.g.
        8 MPa with 0.25 MPa gives 32 steps).
        """
        a = self.data['analysis']
        if a['type'] != 'dynamic':
            raise ScenarioError('analysis.type: the investigation range is only defined for a dynamic analysis')
        if a['steps'] is not None:
            n = a['steps']
        else:
            largest = max(abs(s) for s in self.driving_stresses(current_model))
            n = max(1, int(round(largest/a['step'])))
        return [i/n for i in range(1, n + 1)]

    # ------------------------------------------------------------------ run

    def run(self):
        """
        Computes the scenario and returns a ScenarioResults:
        - static: the model with the crack at its final driving stresses
          (model.compute: displacements, stresses, principal stresses...), for any analysis;
        - dynamic: for a dynamic analysis, the DynamicResults of compute_dynamic_model.
        The metadata of both keep the scenario (to_dict()).
        """
        rem = self.data['remote_stresses']
        litho = self.data['lithostatic']
        current_model = self.build_model()
        disc = self.build_crack(current_model)

        # Static model, with the crack at its final load
        static = current_model.compute(disc, rem['xx'], rem['yy'], litho)
        static.metadata['scenario'] = self.to_dict()

        dynamic = None
        if self.analysis == 'dynamic':
            # compute_dynamic_model takes the opening as the aperture (crack_inputs[0]) or,
            # when it is 0, as the driving stress dx
            g = self.data['grid']
            sigma_I = 0.0 if self.data['cracks'][0]['aperture'] is not None else self.data['cracks'][0]['sigma_I']
            sigma_II, sigma_III = self.data['cracks'][0]['sigma_II'], self.data['cracks'][0]['sigma_III']
            dynamic = compute_dynamic_model(self.crack_inputs(), self.model_inputs(), sigma_I, sigma_II, sigma_III,
                                            rem['xx'], rem['yy'], self.data['gravity'], litho,
                                            self.investigation_range(current_model), g['x'], g['y'], g['cell'],
                                            self.layer_tops())
            dynamic.metadata['scenario'] = self.to_dict()
        return ScenarioResults(self, current_model, disc, static, dynamic)


class ScenarioResults(object):
    """
    The results of Scenario.run():
    - scenario, model, crack: the Scenario, the model() and the crack() at its final load;
    - static: DiESResults of the model with the crack at its final load;
    - dynamic: DynamicResults of a dynamic analysis (None for a static one).

    figure(name) draws one static figure (names in STATIC_FIGURES); figures() yields
    those listed in the output section of the scenario; write(folder) saves results,
    figures and a copy of the scenario.
    """

    def __init__(self, scenario, current_model, disc, static, dynamic=None):
        self.scenario = scenario
        self.model = current_model
        self.crack = disc
        self.static = static
        self.dynamic = dynamic
        self._coarse = None
        self._failure = None

    def __repr__(self):
        return ('ScenarioResults(' + repr(self.scenario.name) + ', static ' + str(self.static.shape)
                + (', ' + str(len(self.dynamic)) + ' dynamic steps' if self.dynamic is not None else '') + ')')

    def coarse(self):
        """Results on the coarser grid of the vector and trajectory figures (computed once, when needed)."""
        if self._coarse is None:
            rem = self.scenario.data['remote_stresses']
            coarse_model = self.scenario.build_model(coarse=True)
            self._coarse = coarse_model.compute(self.crack, rem['xx'], rem['yy'], self.scenario.data['lithostatic'])
            self._coarse.metadata['scenario'] = self.scenario.to_dict()
        return self._coarse

    def failure(self):
        """
        Failure state of the static results (compute_failure_state: r/r_c and the failure
        mode of each node with the crack at its final load), computed once, when needed.
        """
        if self._failure is None:
            self._failure = compute_failure_state(self.static)
        return self._failure

    def plane_angle(self):
        """
        The angle written in the label of plane_normal and plane_shear: analysis.plane_angle,
        or, if not given, the optimal angle of the layer at the centre of the crack
        (90 - coula, in degrees), which is the angle of the planes those figures show.
        """
        angle = self.scenario.data['analysis']['plane_angle']
        if angle is not None:
            return angle
        coula = np.atleast_1d(np.asarray(self.model.layers_coula, dtype=float))
        index = int(self.model.layer_index(self.crack.c()[1]))
        return round(90 - math.degrees(coula[index]), 2)

    def figure(self, name):
        """Draws the static figure called name (see STATIC_FIGURES) and returns it."""
        if name not in STATIC_FIGURES:
            raise ScenarioError('unknown figure ' + repr(name) + '. Valid figures: ' + ', '.join(STATIC_FIGURES))
        grid, draw = STATIC_FIGURES[name]
        res = {'coarse': self.coarse, 'failure': self.failure}.get(grid, lambda: self.static)()
        return draw(res, self)

    def dynamic_steps_to_draw(self):
        """Indices (from 0) of the dynamic steps drawn: one in show_every, and the last one."""
        settings = self.scenario.output['figures']['dynamic']
        if self.dynamic is None or settings is None:
            return []
        n = len(self.dynamic)
        every = settings['show_every']
        return [k for k in range(n) if (k + 1) % every == 0 or k == n - 1]

    def figures(self):
        """
        Yields (file name without extension, figure) for every figure listed in the output
        section: the static ones by name, and the dynamic steps as 'dynamic_step_<step>'.
        Every dynamic step uses the colour range of the final step (as dynamic_model).
        """
        for name in self.scenario.output['figures']['static']:
            yield name, self.figure(name)
        steps = self.dynamic_steps_to_draw()
        if steps:
            downscaling = self.scenario.output['figures']['dynamic']['downscaling']
            color_range = plotting.dynamic_color_range(self.dynamic)
            for k in steps:
                step = self.dynamic[k]
                yield 'dynamic_step_%03d' % step.metadata['step'], \
                    plotting.plot_dynamic_step(step, downscaling, color_range)

    def write(self, folder):
        """
        Writes into folder (created if needed) what the output section asks for, and a
        copy of the scenario. Files:
            scenario.yaml (or .json)       the scenario as given
            static_results.npz / .csv      DiESResults.save / to_csv of the static model
            dynamic_results.npz            DynamicResults.save, every step
            dynamic_final_step.csv         the last step, with failure_step (the step at
                                           which each node first failed, 0 if never)
            <figure>.<format>              each static figure, and dynamic_step_<step>
        Returns the list of paths written.
        """
        out = self.scenario.output
        os.makedirs(folder, exist_ok=True)
        written = []

        # The scenario as given, for the record (its file, or the dict as YAML)
        source = self.scenario.source
        if source is not None and os.path.isfile(source):
            path = os.path.join(folder, 'scenario' + os.path.splitext(source)[1].lower())
            if os.path.abspath(path) != os.path.abspath(source):
                shutil.copyfile(source, path)
        else:
            path = os.path.join(folder, 'scenario.json')
            with open(path, 'w') as f:
                json.dump(self.scenario.raw, f, indent=2)
        written.append(path)

        # Results files
        if 'npz' in out['results']:
            path = os.path.join(folder, 'static_results.npz')
            self.static.save(path) ; written.append(path)
        if 'csv' in out['results']:
            path = os.path.join(folder, 'static_results.csv')
            self.static.to_csv(path) ; written.append(path)
        if self.dynamic is not None:
            if 'npz' in out['results']:
                path = os.path.join(folder, 'dynamic_results.npz')
                self.dynamic.save(path) ; written.append(path)
            if 'csv' in out['results']:
                # The last step, plus the step at which each node first failed
                final = self.dynamic[-1]
                fields = dict(final.fields)
                fields['failure_step'] = self.dynamic.failure_step()
                table = DiESResults(final.x, final.y, fields, final.metadata,
                                    dict(final.units, failure_step=''))
                path = os.path.join(folder, 'dynamic_final_step.csv')
                table.to_csv(path) ; written.append(path)

        # Figures, closed once saved
        for name, fig in self.figures():
            path = os.path.join(folder, name + '.' + out['format'])
            fig.savefig(path, dpi=out['dpi'], format=out['format'])
            plt.close(fig)
            written.append(path)
        return written


# ------------------------------------------------------------------- loading

def load_scenario(path):
    """
    Reads and checks a scenario file: YAML (.yaml, .yml) or JSON (.json).
    Returns a Scenario; raises ScenarioError with every problem found.
    """
    extension = os.path.splitext(path)[1].lower()
    if extension not in ('.json', '.yaml', '.yml'):
        raise ScenarioError(str(path) + ': unknown extension; use .yaml, .yml or .json')
    try:
        with open(path, encoding='utf-8') as f:
            text = f.read()
    except OSError as error:
        raise ScenarioError(str(path) + ': cannot be read (' + str(error) + ')')
    if extension == '.json':
        try:
            raw = json.loads(text)
        except ValueError as error:
            raise ScenarioError(str(path) + ': not valid JSON (' + str(error) + ')')
    else:
        try:
            import yaml
        except ImportError:
            raise ImportError('Reading YAML scenarios needs PyYAML: pip install pyyaml')
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ScenarioError(str(path) + ': not valid YAML (' + str(error) + ')')
    return Scenario.from_dict(raw, source=os.path.abspath(path))
