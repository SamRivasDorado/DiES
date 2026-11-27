"""
One view of the results of the Single model (Phase 4, step 2, Nov 2026; layout of Sam's
sketch of 27 Nov 2026): a selector of its figures and the drawing options, the figure,
under it the table of the probes, and under that the profiles, two per row. Probes and
profiles are added with buttons (none at first) and removed one by one. The figures and
what they take are listed in dies/app/views.py.

State:
- Every tab of Streamlit is drawn at every run of the script, so each view has its own
  widgets (keys start with the name of the view). The option widgets are always drawn,
  and disabled when the figure does not take them, so that their values are kept when
  switching between figures.
- The probes and profiles are shared by all the views: their positions live in
  st.session_state['probes'] and ['profiles'] (lists of dicts, each with an id). The
  widgets of each view are set from those lists before they are drawn, and write back to
  them when changed (on_change), so that a probe moved in one view is moved in all. The
  probes are edited in their table (st.data_editor): its edits are applied to the list
  (views.apply_table_edits) and a version number in the keys of the tables makes every
  view draw its table again from the list.
"""

import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import streamlit as st

from dies.app import compute, figures, session, views
from dies.app.ui import common


def _key(view, name):
    # Key of a widget of a view, e.g. 'Stresses.cmap'
    return view + '.' + name


# ------------------------------------------------------------------- drawing options

def _spacing_label(spacing, cell):
    # Text of an arrow spacing: 'default' or 'every 3 nodes (300 m)'
    if spacing is None:
        return 'default (as dies run)'
    return 'every node (%g m)' % cell if spacing == 1 else 'every %d nodes (%g m)' % (spacing, spacing*cell)


def drawing_options(view, item, cell):
    """
    The drawing options of a view, in an expander. Returns a dict with the options of
    plotting.STYLE_DEFAULTS, 'step' and 'arrow_spacing' (None where automatic or not set).
    cell: the smallest spacing of the grid nodes, in m (for the labels of the arrow spacing).
    """
    k = lambda name: _key(view, name)
    takes = lambda name: name in item.options
    unit = (' (' + item.unit + ')') if item.unit else ''
    options = {}
    with st.expander('Drawing options', expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            options['cmap'] = st.selectbox('Colour map', views.COLOUR_MAPS, key=k('cmap'), disabled=not takes('cmap'),
                                           format_func=lambda name: 'figure default' if name is None else name)
            options['center_zero'] = st.checkbox('Colours centred on 0', key=k('center_zero'),
                                                 disabled=not takes('center_zero'),
                                                 help='Symmetric colours around 0: with a diverging map (RdBu_r, '
                                                      'coolwarm, seismic), tension and compression get opposite '
                                                      'colours and 0 is white')
        with c2:
            manual = st.checkbox('Manual colour limits', key=k('manual_limits'), disabled=not takes('limits'))
            low, high = st.columns(2)
            vmin = low.number_input('min' + unit, key=k('vmin'), value=None, format='%g', disabled=not manual)
            vmax = high.number_input('max' + unit, key=k('vmax'), value=None, format='%g', disabled=not manual)
            options['limits'] = None
            if takes('limits') and manual:
                if vmin is None or vmax is None or not vmin < vmax:
                    st.caption('Give min < max to use them.')
                else:
                    options['limits'] = (vmin, vmax)
        with c3:
            options['contours'] = st.checkbox('Contours', key=k('contours'), value=True,
                                              disabled=not takes('contours'))
            options['contour_labels'] = st.checkbox('Contour labels', key=k('contour_labels'), value=True,
                                                    disabled=not (takes('contour_labels') and options['contours']))
            manual_step = st.checkbox('Manual contour step', key=k('manual_step'), disabled=not takes('step'))
            step = st.number_input('step' + unit, key=k('step'), value=None, format='%g',
                                   disabled=not (takes('step') and manual_step))
            options['step'] = step if (takes('step') and manual_step and step and step > 0) else None
        with c4:
            options['info_box'] = st.checkbox('Text box', key=k('info_box'), value=True,
                                              disabled=not takes('info_box'),
                                              help='The box with the driving stresses (and the material)')
            options['arrow_scale'] = st.slider('Arrow length', 0.25, 4.0, 1.0, 0.25, key=k('arrow_scale'),
                                               disabled=not takes('arrow_scale'),
                                               help='Length of the arrows and trajectory segments, relative to '
                                                    'the default')
            spacing = st.selectbox('Arrow density', views.ARROW_SPACINGS, key=k('arrow_spacing'),
                                   disabled=not takes('arrow_spacing'),
                                   format_func=lambda n: _spacing_label(n, cell),
                                   help='One arrow every n nodes of the grid. Default: the coarser grid of the '
                                        'scenario figures (2.5 times the cell along x, 3 times along y)')
            options['arrow_spacing'] = spacing if takes('arrow_spacing') else None
    return options


# ------------------------------------------------------------- probes and profiles: state

def _items(kind):
    # The shared list of probes ('probes') or profiles ('profiles')
    return st.session_state.setdefault(kind, [])


def _add(kind, values):
    # Callback of the Add buttons: a new probe or profile, with a new id
    st.session_state[kind + '_next'] = st.session_state.get(kind + '_next', 0) + 1
    entry = dict(values, id=st.session_state[kind + '_next'], show=True)
    _items(kind).append(entry)


def _remove(kind, ident):
    # Callback of the Remove buttons
    st.session_state[kind] = [entry for entry in _items(kind) if entry['id'] != ident]


def _store(kind, ident, field, key):
    # Callback of a widget: its new value goes to the shared list
    for entry in _items(kind):
        if entry['id'] == ident:
            entry[field] = st.session_state[key]


def _bound(widget, label, kind, entry, field, view, **kwargs):
    """
    A widget bound to a field of a probe or profile: set from the shared list before it is
    drawn, and writing back to it when changed. Returns its value.
    """
    key = _key(view, '%s%d.%s' % (kind, entry['id'], field))
    st.session_state[key] = entry[field]
    return widget(label, key=key, on_change=_store, args=(kind, entry['id'], field, key), **kwargs)


def _probe_default(run, count):
    """
    Starting point of a new probe: above the upper tip of the crack, halfway to the top of
    the grid (where a graben forms above a dike), moved along x for each probe already there.
    """
    static = run.static
    (ux, uy) = run.crack.up_tip()
    top = float(np.max(static.y_nodes()))
    x_nodes = static.x_nodes()
    shift = (float(x_nodes.max()) - float(x_nodes.min()))/10*count
    return {'x': round(float(ux) + shift, 1), 'y': round((float(uy) + top)/2, 1)}


def _profile_default(run, count):
    """
    Starting line of a new profile: the first one along the top row of nodes (the surface
    profile, Sam's choice), the second one horizontal through the centre of the crack, then
    vertical lines across the grid beside the crack.
    """
    static = run.static
    (xa, ya), (xb, yb) = views.default_profile(static)
    cx, cy = run.crack.c()
    if count == 1:
        ya = yb = float(cy)
    elif count >= 2:
        y_nodes = static.y_nodes()
        x = float(cx) + (xb - xa)/8*(count - 1)
        xa = xb = min(max(x, xa), xb)
        ya, yb = float(y_nodes.max()), float(y_nodes.min())
    return {'ax': round(xa, 1), 'ay': round(ya, 1), 'bx': round(xb, 1), 'by': round(yb, 1)}


def _swatch(text, colour):
    # A label in the colour of its marks on the map (with a dark outline, readable on any theme)
    return ('<span style="color:%s; font-weight:700; text-shadow: 0 0 2px #000, 0 0 2px #000">%s</span>'
            % (colour, text))


# -------------------------------------------------------------------- probes and profiles

def _apply_probe_edits(key, default):
    """
    Callback of the probe table: its edits go to the shared list of probes, and the
    version of the tables goes up, so that the table of every view is drawn again from
    the list (a table keeps its own edits until its key changes). default: the starting
    point of a probe added as an empty row.
    """
    def new_entry(values):
        st.session_state['probes_next'] = st.session_state.get('probes_next', 0) + 1
        return dict(default, show=True, **values, id=st.session_state['probes_next'])

    st.session_state['probes'] = views.apply_table_edits(_items('probes'), st.session_state[key], new_entry)
    st.session_state['probes_version'] = st.session_state.get('probes_version', 0) + 1


def _add_probe(values):
    # Callback of Add probe: a new probe, and the tables drawn again
    _add('probes', values)
    st.session_state['probes_version'] = st.session_state.get('probes_version', 0) + 1


def _probes(view, run):
    """
    The probes, in one table (Sam, 27 Nov 2026: their definition and their values together):
    x, y and 'On the map' are edited in the table; the node used and the values at it are
    computed. Rows are added with Add probe (or the + of the table) and removed by selecting
    them and deleting them in the table. Returns the marks for the map.
    """
    probes = _items('probes')
    default = _probe_default(run, len(probes))
    head, button = st.columns([5, 1])
    head.markdown('**Probes**: the values at the node nearest to each point')
    button.button('Add probe', key=_key(view, 'add_probe'), on_click=_add_probe, args=(default,), width='stretch')
    if not probes:
        st.caption('No probe yet: add one to read the values of every field at a point.')
        return []
    labels = ['P%d' % (i + 1) for i in range(len(probes))]
    rows = views.probe_table(run, [(e['x'], e['y'], label) for e, label in zip(probes, labels)])
    # The table: label, the inputs, then the node used and the values (computed)
    table = pd.DataFrame([OrderedDict([('Probe', row['Probe']), ('x (m)', e['x']), ('y (m)', e['y']),
                                       ('On the map', bool(e['show']))]
                                      + [(name, value) for name, value in row.items() if name != 'Probe'])
                          for e, row in zip(probes, rows)])
    computed = [name for name in table.columns if name not in views.PROBE_INPUTS]
    config = {name: st.column_config.NumberColumn(name, format='%.4g', disabled=True)
              for name in computed if name not in ('Probe', 'Failure state')}
    config['Probe'] = st.column_config.TextColumn('Probe', disabled=True, width='small')
    config['Failure state'] = st.column_config.TextColumn('Failure state', disabled=True)
    config['x (m)'] = st.column_config.NumberColumn('x (m)', format='%g', step=1.0, required=False)
    config['y (m)'] = st.column_config.NumberColumn('y (m)', format='%g', step=1.0, required=False)
    config['On the map'] = st.column_config.CheckboxColumn('On the map')
    key = _key(view, 'probe_table@%d' % st.session_state.get('probes_version', 0))
    # The hint goes above the table: the toolbar of the table, shown above it on hover, then
    # covers the hint instead of the Add probe button
    st.caption('Edit x, y and "On the map" in the table; the node used and its values follow. To remove probes, '
               'select their rows (left edge of the table) and delete them.')
    st.data_editor(table, key=key, num_rows='dynamic', hide_index=True, column_config=config, width='stretch',
                   on_change=_apply_probe_edits, args=(key, default))
    return [(row['node x (m)'], row['node y (m)'], row['Probe']) for e, row in zip(probes, rows) if e['show']]


def _profile_chart_options(view):
    """Options of the profile plots (grids, axis labels, line), in an expander."""
    k = lambda name: _key(view, 'profile_' + name)
    defaults = figures.PROFILE_OPTIONS
    with st.expander('Profile plot options', expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        options = {
            'major_grid': c1.checkbox('Major grid', key=k('major_grid'), value=defaults['major_grid']),
            'minor_grid': c2.checkbox('Minor grid', key=k('minor_grid'), value=defaults['minor_grid']),
            'axis_labels': c3.checkbox('Axis labels', key=k('axis_labels'), value=defaults['axis_labels'],
                                       help='Without them, the name of each field is written inside its plot'),
            'markers': c4.checkbox('Sampled points', key=k('markers'), value=defaults['markers']),
            'line_width': c5.slider('Line width', 0.5, 4.0, defaults['line_width'], 0.1, key=k('line_width')),
        }
    return options


def _profiles(view, run, item, run_key):
    """The profiles: Add button, plot options and one card per profile, two per row. Returns the marks."""
    profiles = _items('profiles')
    head, button = st.columns([5, 1])
    head.markdown('**Profiles**: the fields of the figure along a line A → B')
    button.button('Add profile', key=_key(view, 'add_profile'), on_click=_add,
                  args=('profiles', _profile_default(run, len(profiles))), width='stretch')
    if not profiles:
        st.caption('No profile yet: add one (the first one runs along the surface, the top row of nodes).')
        return []
    plot_options = _profile_chart_options(view)
    surface = _profile_default(run, 0)
    marks = []
    for start in range(0, len(profiles), 2):
        columns = st.columns(2)
        for column, (i, entry) in zip(columns, list(enumerate(profiles))[start:start + 2]):
            label = str(i + 1)
            with column.container(border=True):
                st.markdown(_swatch('Profile %s: A%s → B%s' % (label, label, label), figures.mark_colour(i)),
                            unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                xa = _bound(c1.number_input, 'A x', 'profiles', entry, 'ax', view, format='%g', step=100.0)
                ya = _bound(c2.number_input, 'A y', 'profiles', entry, 'ay', view, format='%g', step=100.0)
                xb = _bound(c3.number_input, 'B x', 'profiles', entry, 'bx', view, format='%g', step=100.0)
                yb = _bound(c4.number_input, 'B y', 'profiles', entry, 'by', view, format='%g', step=100.0)
                c1, c2, c3 = st.columns(3)
                show = _bound(c1.checkbox, 'On the map', 'profiles', entry, 'show', view)

                def to_surface(ident=entry['id']):
                    # Back to the surface profile (callback: before the widgets are drawn again)
                    for e in _items('profiles'):
                        if e['id'] == ident:
                            e.update(surface)

                c2.button('Surface', key=_key(view, 'surface%d' % entry['id']), on_click=to_surface,
                          help='Along the top row of nodes, from one side of the grid to the other', width='stretch')
                c3.button('Remove', key=_key(view, 'remove_profile%d' % entry['id']), on_click=_remove,
                          args=('profiles', entry['id']), width='stretch')
                if (xa, ya) == (xb, yb):
                    st.caption('A and B are the same point.')
                    continue
                data = views.profile(run, item, (xa, ya), (xb, yb))
                colour = figures.line_colour(i)
                cache = json.dumps([item.key, xa, ya, xb, yb, label, plot_options], sort_keys=True)
                st.image(compute.figure_bytes('profile:' + view, run_key, cache, 'png', common.SCREEN_DPI,
                                              lambda: figures.plot_profile(data, colour, **plot_options)),
                         width='stretch')
                st.download_button('CSV', pd.DataFrame(data).to_csv(index=False),
                                   file_name='profile_%s_%s.csv' % (label, item.key), mime='text/csv',
                                   key=_key(view, 'profile_csv%d' % entry['id']), on_click='ignore')
                if show:
                    marks.append(((xa, ya), (xb, yb), label))
    return marks


# ------------------------------------------------------------------------------- view

def render(view, run_state, form):
    """
    Draws a view. run_state: the last run of the single model (dies.app.compute.run), or
    None; form: the form as it is now (for the label of the plane figures).
    """
    if run_state is None:
        st.info('Press **Run single model** to compute the model and see this view.')
        return
    run = run_state['results']
    items = views.VIEWS[view]
    labels = {item.key: item.label for item in items}
    keys = [item.key for item in items]

    # The figure, and the label of the plane figures (Planes & failure)
    if view == 'Planes & failure':
        left, right = st.columns([3, 2])
        chosen = left.selectbox('Figure', keys, format_func=labels.get, key=_key(view, 'figure'))
        right.number_input('Plane angle α in the label (°)', key=session.key('plane_angle'), value=None, format='%g',
                           placeholder='optimal: %g' % views.optimal_plane_angle(run),
                           help='Only the label of the σn and τ figures: the stresses are always those on the '
                                'optimal shear planes. Empty: the optimal angle.')
    else:
        left, _ = st.columns([3, 2])
        chosen = left.selectbox('Figure', keys, format_func=labels.get, key=_key(view, 'figure'))
    item = views.figure(view, chosen)
    cell = min(float(np.min(np.abs(np.diff(v)))) for v in (run.static.x_nodes(), run.static.y_nodes()))
    options = drawing_options(view, item, cell)
    style, step = views.style_for(item, options)
    extra = {'plane_angle': form.get('plane_angle')}

    # The figure goes first on the page but is drawn last, once the probes and profiles
    # (drawn under it) are known: an empty slot keeps its place
    _, middle, _ = st.columns([1, 3, 1])
    slot = middle.empty()
    probes = _probes(view, run)
    profiles = _profiles(view, run, item, run_state['key'])
    with slot.container():
        res = item.results(run, options)
        # Everything that changes the figure goes into the label of its cache entry
        label = json.dumps({'style': style, 'step': step, 'spacing': options.get('arrow_spacing'), 'extra': extra,
                            'probes': probes, 'profiles': profiles}, sort_keys=True, default=list)

        def draw():
            return figures.add_overlays(item.draw(res, run, style, step, extra), probes, profiles)

        common.show_figure(view + ':' + item.key, run_state['key'], label, draw, file_stem=item.key)
