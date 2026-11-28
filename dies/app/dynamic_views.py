"""
The outputs of the Dynamic model in the app (Phase 4, step 3, Nov 2026): what the player
draws at each step, the drawing options of each output, and what the probes and profiles
give at that step. No Streamlit here.

Sam's choices (28 Nov 2026): move through the steps one by one or play them as a movie,
for every output, with the failure maps both as the app draws them (the colour of the
failure state of each node) and as his scripts do (sigma_n with the traces of the
failure planes).

Every output is drawn by plotting.py from the DynamicResults of a run (ScenarioResults.dynamic);
step k counts from 1 (the k-th step, k/n of the final load).
"""

from collections import OrderedDict

import numpy as np

import plotting
from results import DiESResults
from Helper_functions import true_min_max

# Drawing options of each kind of output (names of plotting.STYLE_DEFAULTS, plus options
# of the app: 'arrow_spacing' (the downscaling of the plane traces), 'same_range' (the
# colours of the final step in every step), 'unit' and 'share')
PLANES = ('cmap', 'limits', 'contours', 'info_box', 'arrow_scale', 'arrow_spacing', 'above', 'traces', 'same_range')
SHEAR = ('cmap', 'limits', 'center_zero', 'contours', 'contour_labels', 'step', 'same_range')
FIRST = ('cmap', 'limits', 'info_box', 'unit')
APP_OPTIONS = ('arrow_spacing', 'same_range', 'unit', 'share', 'step')

# Units of the map of the first failure
FIRST_FAILURE_UNITS = OrderedDict([('MPa', 'driving stress (MPa)'), ('percent', '% of the final load'),
                                   ('step', 'step number')])


def first_failure_results(dyn, upto):
    """
    The first failure of each node up to step upto, as a DiESResults (for the probes and
    profiles): first_failure_load (MPa, NaN where it has not failed yet) and
    first_failure_step (0 where it has not failed yet).
    """
    last = dyn[-1]
    first = dyn.failure_step()
    first = np.where(first <= upto, first, 0)
    loads = dyn.failure_load(upto)/1e6
    return DiESResults(last.x, last.y, {'first_failure_load': loads, 'first_failure_step': first.astype(float)},
                       last.metadata)


class Output(object):
    """
    One output of the player. key, label; options: the drawing options it takes; unit:
    of its colour scale; is_map: drawn on the grid (probes and profiles are marked on it);
    profile: the fields of its profiles, as (field, source, factor, label), source being
    'step' (the results of the step) or 'first' (first_failure_results); draw(dyn, k,
    style, options): the figure of step k.
    """

    def __init__(self, key, label, options, unit, is_map, profile, draw):
        self.key, self.label, self.options, self.unit = key, label, options, unit
        self.is_map, self.profile, self.draw = is_map, profile, draw


def _planes(dyn, k, style, options):
    # sigma_n with the traces of the failure planes (plotting.plot_dynamic_step, Sam's figure),
    # by default with the colour range of the final step in every step
    color_range = plotting.dynamic_color_range(dyn) if options.get('same_range', True) else None
    return plotting.plot_dynamic_step(dyn[k - 1], int(options.get('arrow_spacing') or 1), color_range, **style)


def _shear(dyn, k, style, options):
    # tau on the failure planes, by default with the range of the final step in every step
    if options.get('same_range', True) and 'limits' not in style and not style.get('center_zero'):
        style = dict(style, limits=tuple(true_min_max(dyn[-1]['failure_ss'])))
    return plotting.plot_field(dyn[k - 1], 'failure_ss', r'$\tau$ on the failure plane (MPa)', 1.0,
                               options.get('step'), **style)


OUTPUTS = [
    Output('planes', 'σn on the failure planes, with their traces', PLANES, 'MPa', True,
           [('failure_sn', 'step', 1.0, 'σn (MPa)')], _planes),
    Output('shear', 'τ on the failure planes', SHEAR, 'MPa', True,
           [('failure_ss', 'step', 1.0, 'τ (MPa)')], _shear),
    Output('failure_mode', 'Failure state of each node', (), '', True,
           [('first_failure_load', 'first', 1.0, 'first failure (MPa)')],
           lambda dyn, k, style, options: plotting.plot_failure_modes(dyn[k - 1])),
    Output('first_failure', 'First failure of each node (up to this step)', FIRST, '', True,
           [('first_failure_load', 'first', 1.0, 'first failure (MPa)')],
           lambda dyn, k, style, options: plotting.plot_first_failure(dyn, k, options.get('unit') or 'MPa',
                                                                       **style)),
    Output('evolution', 'Evolution of the failure states', ('share',), '', False, [],
           lambda dyn, k, style, options: plotting.plot_failure_evolution(dyn, k, bool(options.get('share')))),
]


def output(key):
    """The Output by its key."""
    for item in OUTPUTS:
        if item.key == key:
            return item
    raise KeyError(key)


def style_for(item, options):
    """
    The plotting options of an output: those it takes, without the options of the app and
    without those left at their default (so that it is drawn exactly as by the scripts).
    """
    style = {}
    for name in item.options:
        if name in APP_OPTIONS:
            continue
        value = options.get(name)
        if value is not None and value != plotting.STYLE_DEFAULTS[name]:
            style[name] = value
    return style


def _source(dyn, k, source):
    return first_failure_results(dyn, k) if source == 'first' else dyn[k - 1]


def probe_table(run, k, points):
    """
    The probes at step k, one row per probe (as dies.app.views.probe_table): its label,
    the node used, and at that step the failure state, sigma_n, tau and theta on the
    failure plane, and the step and load of its first failure (up to step k).
    """
    dyn = run.dynamic
    step = dyn[k - 1]
    first = first_failure_results(dyn, k)
    rows = []
    for x, y, label in points:
        values = step.probe(x, y, names=['failure_mode', 'failure_sn', 'failure_ss', 'failure_theta'])
        earliest = first.probe(x, y, names=['first_failure_step', 'first_failure_load'])
        rows.append(OrderedDict([
            ('Probe', label), ('node x (m)', values['x']), ('node y (m)', values['y']),
            ('Failure state', str(values['failure_mode'])),
            ('σn (MPa)', values['failure_sn']), ('τ (MPa)', values['failure_ss']), ('θ (°)', values['failure_theta']),
            ('First failure: step', earliest['first_failure_step']),
            ('First failure: driving stress (MPa)', earliest['first_failure_load']),
        ]))
    return rows


def profile(run, k, item, A, B, n=200):
    """
    The profile of the fields of an output at step k along A -> B, as dies.app.views.profile:
    'distance (m)', 'x (m)', 'y (m)' and the fields of the output.
    """
    dyn = run.dynamic
    out = dyn[k - 1].profile(A, B, n, names=[])
    data = OrderedDict([('distance (m)', out['distance']), ('x (m)', out['x']), ('y (m)', out['y'])])
    for field, source, factor, label in item.profile:
        data[label] = _source(dyn, k, source).profile(A, B, n, names=[field])[field]*factor
    return data


def step_label(dyn, k):
    """Text of step k, e.g. 'Step 12 of 32 · 37.5 % · driving stress 3 MPa'."""
    meta = dyn[k - 1].metadata
    return 'Step %d of %d · %g %% of the final load · driving stress %g MPa' % (
        k, len(dyn), round(100*meta['percentage'], 2), round(dyn.step_loads()[k - 1]/1e6, 3))
