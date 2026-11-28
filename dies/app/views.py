"""
The figures of the views of the Single model (Phase 4, step 2, Nov 2026), and what the
interface needs to know about each one: how to draw it, which drawing options it takes,
and which fields go to the profile. No Streamlit here.

Every figure is drawn by the functions of plotting.py (the same ones the scripts use),
from a ScenarioResults (dies.Scenario.run): its static results, its results on the
coarser grid of the arrows and trajectories (coarse()), or its failure state (failure()).

Also here: the probe (values at the node nearest to a point), the default profile (along
the top of the grid, i.e. the surface profile, Sam's choice of 27 Nov 2026) and the
optimal plane angle used as the default label of the plane figures.
"""

import math
from collections import OrderedDict

import numpy as np

import plotting

# Drawing options of the app, by kind of figure: names of plotting.STYLE_DEFAULTS, plus two
# of the app: 'step', the contour interval, and 'arrow_spacing', the density of the arrows
# (one every n nodes of the grid; None: the coarser grid of the scenario figures)
SCALAR = ('cmap', 'limits', 'center_zero', 'contours', 'contour_labels', 'step')
ARROWS = ('cmap', 'limits', 'center_zero', 'arrow_scale', 'arrow_spacing', 'info_box')
TRAJECTORIES = ('arrow_scale', 'arrow_spacing')
# Options of the app that are not passed to plotting.py
APP_OPTIONS = ('step', 'arrow_spacing')

# Arrow spacings offered: one arrow every n nodes of the grid (0: the default one). The
# defaults of these two lists are not None: a select box whose value is None shows no
# choice after the page runs again
ARROW_SPACINGS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20]

# Colour maps offered ('': the figure's own one)
COLOUR_MAPS = ['', 'viridis', 'cividis', 'plasma', 'magma', 'inferno', 'RdBu_r', 'coolwarm', 'seismic',
               'Spectral_r', 'twilight']


class Figure(object):
    """
    One figure of a view.

    key: its name (as in the scenario figures when it is one of them); label: as shown;
    grid: the results it is drawn from ('fine', 'coarse' or 'failure'); options: the
    drawing options it takes; unit: the unit of its colour scale (for the limits);
    profile: the fields drawn in the profile, as (field, source, factor, label), where
    the field is multiplied by factor (e.g. Pa -> MPa); draw(res, run, style, step,
    extra): returns the matplotlib figure.
    """

    def __init__(self, key, label, grid, options, unit, profile, draw):
        self.key, self.label, self.grid, self.options = key, label, grid, options
        self.unit, self.profile, self.draw = unit, profile, draw

    def results(self, run, options=None):
        """
        The results this figure is drawn from. For the arrows and trajectories, with an
        arrow spacing n in options, the static results at one node every n (subsample).
        """
        spacing = (options or {}).get('arrow_spacing')
        if self.grid == 'coarse' and spacing:
            return subsample(run.static, int(spacing))
        return {'coarse': run.coarse, 'failure': run.failure}.get(self.grid, lambda: run.static)()


def subsample(res, n):
    """
    The results at one node every n along x and y (Phase 4: the arrow density of the app),
    centred on the grid. The values are those of the nodes kept: nothing is computed again.
    """
    from results import DiESResults
    if n <= 1:
        return res
    rows, columns = res.shape
    # Start so that the nodes left out are shared between both sides
    i0 = ((rows - 1) % n)//2 ; j0 = ((columns - 1) % n)//2
    cut = (slice(i0, None, n), slice(j0, None, n))
    fields = {name: values[cut] for name, values in res.fields.items()}
    return DiESResults(res.x[cut], res.y[cut], fields, res.metadata, res.units)


def _displacement(component, option):
    # Displacement figure: arrows (on the coarser grid) or colour map with contours
    return lambda res, run, style, step, extra: plotting.plot_displacement(res, component, option, step, **style)


def _stress(field):
    return lambda res, run, style, step, extra: plotting.plot_stresses(res, field, step, **style)


def _principal(which):
    return lambda res, run, style, step, extra: plotting.plot_principal_stresses(res, which, step, **style)


def _plane(which):
    # The angle of the label: the one typed, or the optimal one
    return lambda res, run, style, step, extra: plotting.plot_plane_stresses(
        res, which, extra.get('plane_angle') or optimal_plane_angle(run), step, **style)


# Fields of the profile, as (field, source, factor, label)
_UX = ('ux', 'fine', 1.0, 'ux (m)')
_UY = ('uy', 'fine', 1.0, 'uy (m)')
_S1 = ('s1', 'fine', 1e-6, 'σ1 (MPa, compression +)')
_S3 = ('s3', 'fine', 1e-6, 'σ3 (MPa, compression +)')


def _mpa(field, label):
    return (field, 'fine', 1e-6, label + ' (MPa)')


# The views and their figures, in the order of the selectors (the arrows and trajectories are
# drawn on the coarser grid of the scenario figures, or with the arrow spacing chosen)
VIEWS = OrderedDict([
    ('Displacements', [
        Figure('uxy', '|u| in the plane: arrows', 'coarse', ARROWS, 'm', [_UX, _UY], _displacement('uxy', 'vectors')),
        Figure('ux', 'ux: arrows', 'coarse', ARROWS, 'm', [_UX], _displacement('ux', 'vectors')),
        Figure('uy', 'uy: arrows', 'coarse', ARROWS, 'm', [_UY], _displacement('uy', 'vectors')),
        Figure('uxy_contours', '|u| in the plane: colour map', 'fine', SCALAR + ('info_box',), 'm', [_UX, _UY],
               _displacement('uxy', 'contours')),
        Figure('ux_contours', 'ux: colour map', 'fine', SCALAR + ('info_box',), 'm', [_UX],
               _displacement('ux', 'contours')),
        Figure('uy_contours', 'uy: colour map', 'fine', SCALAR + ('info_box',), 'm', [_UY],
               _displacement('uy', 'contours')),
    ]),
    ('Stresses', [
        Figure('sxx', 'σxx', 'fine', SCALAR, 'MPa', [_mpa('sxx', 'σxx')], _stress('sxx')),
        Figure('syy', 'σyy', 'fine', SCALAR, 'MPa', [_mpa('syy', 'σyy')], _stress('syy')),
        Figure('sxy', 'σxy', 'fine', SCALAR, 'MPa', [_mpa('sxy', 'σxy')], _stress('sxy')),
        Figure('sxz', 'σxz (mode III)', 'fine', SCALAR, 'MPa', [_mpa('sxz', 'σxz')], _stress('sxz')),
        Figure('szz', 'σzz (plane strain)', 'fine', SCALAR, 'MPa', [_mpa('szz', 'σzz')], _stress('szz')),
        Figure('sav', 'Mean stress (σxx + σyy)/2', 'fine', SCALAR, 'MPa', [_mpa('sav', 'σav')], _stress('sav')),
        Figure('tmax', 'Maximum shear stress', 'fine', SCALAR, 'MPa', [_mpa('tmax', 'τmax')], _stress('tmax')),
    ]),
    ('Principal & trajectories', [
        Figure('s1', 'σ1 (maximum compressive)', 'fine', SCALAR, 'MPa', [_S1], _principal('principal')),
        Figure('s3', 'σ3 (minimum compressive)', 'fine', SCALAR, 'MPa', [_S3], _principal('secondary')),
        Figure('angle', 'Orientation of σ1', 'fine', ('cmap',), 'deg', [('s1a', 'fine', 1.0, 'σ1 angle from +x (°)')],
               _principal('angle')),
        Figure('trajectories', 'Trajectories of σ1 and σ3', 'coarse', TRAJECTORIES, '', [_S1, _S3],
               lambda res, run, style, step, extra: plotting.plot_stress_trajectories(res, False, **style)),
        Figure('trajectories_planes', 'Trajectories and optimal shear planes', 'coarse', TRAJECTORIES, '',
               [_S1, _S3],
               lambda res, run, style, step, extra: plotting.plot_stress_trajectories(res, True, **style)),
    ]),
    ('Planes & failure', [
        Figure('plane_normal', 'σn on the optimal shear planes', 'fine', SCALAR + ('info_box',), 'MPa',
               [('sn', 'fine', 1e-6, 'σn (MPa, compression +)')], _plane('normal')),
        Figure('plane_shear', 'τ on the optimal shear planes', 'fine', SCALAR + ('info_box',), 'MPa',
               [('ss', 'fine', 1e-6, 'τ (MPa)')], _plane('shear')),
        Figure('failure_mode', 'Failure state of each node', 'failure', (), '',
               [('failure_ratio', 'failure', 1.0, 'r / r_c')],
               lambda res, run, style, step, extra: plotting.plot_failure_modes(res)),
        Figure('failure_ratio', 'How close to failure: r / r_c', 'failure', ('cmap', 'limits', 'contours'), '',
               [('failure_ratio', 'failure', 1.0, 'r / r_c')],
               lambda res, run, style, step, extra: plotting.plot_failure_ratio(res, **style)),
    ]),
])


def figure(view, key):
    """The Figure of a view by its key."""
    for item in VIEWS[view]:
        if item.key == key:
            return item
    raise KeyError(key)


def style_for(item, options):
    """
    The drawing options of the app for a figure, as (style, step): style holds the
    plotting options the figure takes, without those left at their default (so that the
    figure is drawn exactly as by the scripts); step is the contour interval (None: automatic).
    """
    style = {}
    for name in item.options:
        if name in APP_OPTIONS:
            continue
        value = options.get(name)
        if value is not None and value != plotting.STYLE_DEFAULTS[name]:
            style[name] = value
    return style, (options.get('step') if 'step' in item.options else None)


def optimal_plane_angle(run):
    """
    The angle of the optimal shear planes in the layer at the centre of the crack,
    90 - coula in degrees (59.84 with a friction of 0.57): the planes the plane figures
    show, and the default of their label (as ScenarioResults.plane_angle without an angle given).
    """
    coula = np.atleast_1d(np.asarray(run.model.layers_coula, dtype=float))
    index = int(run.model.layer_index(run.crack.c()[1]))
    return round(90 - math.degrees(coula[index]), 2)


# ------------------------------------------------------------------- probe and profile

# Fields of the probe table: (field, source, factor, label)
PROBE_FIELDS = [
    ('ux', 'fine', 1.0, 'ux (m)'), ('uy', 'fine', 1.0, 'uy (m)'),
    ('sxx', 'fine', 1e-6, 'σxx (MPa)'), ('syy', 'fine', 1e-6, 'σyy (MPa)'), ('sxy', 'fine', 1e-6, 'σxy (MPa)'),
    ('s1', 'fine', 1e-6, 'σ1 (MPa, compression +)'), ('s3', 'fine', 1e-6, 'σ3 (MPa, compression +)'),
    ('s1a', 'fine', 1.0, 'σ1 angle from +x (°)'),
    ('sn', 'fine', 1e-6, 'σn, optimal planes (MPa)'), ('ss', 'fine', 1e-6, 'τ, optimal planes (MPa)'),
    ('failure_ratio', 'failure', 1.0, 'r / r_c'), ('failure_mode', 'failure', None, 'Failure state'),
]


def _source(run, source):
    return run.failure() if source == 'failure' else run.static


def probe(run, x, y):
    """
    Values at the node nearest to (x, y), as (node x, node y, rows), where rows are
    (label, value) pairs in plotting units (MPa for stresses).
    """
    rows = []
    node = run.static.probe(x, y, names=[])
    for field, source, factor, label in PROBE_FIELDS:
        value = _source(run, source).probe(x, y, names=[field])[field]
        rows.append((label, value if factor is None else value*factor))
    return node['x'], node['y'], rows


def default_profile(res):
    """
    The default profile: along the top row of nodes, from the first to the last column
    (the surface profile, e.g. the subsidence and extension above a dike). Returns (A, B).
    """
    x, y = res.x_nodes(), res.y_nodes()
    top = float(np.max(y))
    return (float(np.min(x)), top), (float(np.max(x)), top)


def probe_table(run, points):
    """
    The probes as one table (Sam, 27 Nov 2026: a band under the figure): one row per probe,
    as a list of dicts with 'Probe' (its label), the node used ('node x (m)', 'node y (m)')
    and the value of each field of PROBE_FIELDS. points: (x, y, label) of each probe.
    """
    rows = []
    for x, y, label in points:
        node_x, node_y, values = probe(run, x, y)
        row = OrderedDict([('Probe', label), ('node x (m)', node_x), ('node y (m)', node_y)])
        row.update(values)
        rows.append(row)
    return rows


# Columns of the probe table that are edited (Sam, 27 Nov 2026: the probes are defined in
# their own table), and the field of each probe they set
PROBE_INPUTS = OrderedDict([('x (m)', 'x'), ('y (m)', 'y'), ('On the map', 'show')])


def apply_table_edits(entries, edits, new_entry):
    """
    The probes after the edits of their table (st.data_editor: a dict with 'edited_rows'
    {row: {column: value}}, 'added_rows' [{column: value}] and 'deleted_rows' [row]).
    entries: the probes before, as dicts with 'x', 'y', 'show' and 'id', in the order of
    the rows; new_entry(values): a new probe from the values given in an added row (the
    missing ones filled by the caller). An emptied cell keeps the value it had. Returns
    the new list (edits first, then deletions, then the added rows at the end).
    """
    result = [dict(entry) for entry in entries]
    for row, change in (edits.get('edited_rows') or {}).items():
        row = int(row)
        if not 0 <= row < len(result):
            continue
        for column, value in change.items():
            if column in PROBE_INPUTS and value is not None:
                result[row][PROBE_INPUTS[column]] = bool(value) if column == 'On the map' else float(value)
    deleted = {int(row) for row in (edits.get('deleted_rows') or [])}
    result = [entry for row, entry in enumerate(result) if row not in deleted]
    for added in edits.get('added_rows') or []:
        values = {PROBE_INPUTS[column]: value for column, value in added.items()
                  if column in PROBE_INPUTS and value is not None}
        result.append(new_entry(values))
    return result


def profile(run, item, A, B, n=200):
    """
    The profile of the fields of a figure along A -> B: a dict with 'distance' (m), 'x',
    'y' and, for each field, its label -> 1D array in plotting units.
    """
    out = run.static.profile(A, B, n, names=[])
    data = OrderedDict([('distance (m)', out['distance']), ('x (m)', out['x']), ('y (m)', out['y'])])
    for field, source, factor, label in item.profile:
        values = _source(run, source).profile(A, B, n, names=[field])[field]
        data[label] = values*factor
    return data
