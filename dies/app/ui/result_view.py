"""
One view of the results of the Single model (Phase 4, step 2, Nov 2026): a selector of
its figures, the drawing options, the figure, and beside it the probe and the profile.
The figures and what they take are listed in dies/app/views.py.

Every tab of Streamlit is drawn at every run of the script, so each view keeps its own
widgets (keys start with the name of the view). The option widgets are always drawn,
and disabled when the figure shown does not take them, so that their values are kept
when switching between figures.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st

from dies.app import compute, figures, session, views
from dies.app.ui import common


def _key(view, name):
    # Key of a widget of a view, e.g. 'Stresses.cmap'
    return view + '.' + name


def _init(key, value):
    # Starting value of a widget, set once (its value then lives in st.session_state)
    if key not in st.session_state:
        st.session_state[key] = value


def drawing_options(view, item):
    """
    The drawing options of a view, in an expander. Returns a dict with the options of
    plotting.STYLE_DEFAULTS and 'step' (None where automatic or not set).
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
    return options


def _probe_and_profile(view, run, item, run_key):
    """
    The probe and the profile, beside the figure. run_key: the scenario key of the run
    (it identifies the cached profile figures). Returns (probe node or None, profile
    (A, B) or None), the marks to draw on the map.
    """
    k = lambda name: _key(view, name)
    static = run.static
    # Starting points: the probe above the upper tip of the crack, halfway to the top of
    # the grid (where a graben forms above a dike); the profile along the top of the grid
    (ux, uy) = run.crack.up_tip()
    top = float(np.max(static.y_nodes()))
    _init(k('probe_x'), round(float(ux), 1))
    _init(k('probe_y'), round((float(uy) + top)/2, 1))
    A, B = views.default_profile(static)
    for name, value in zip(('ax', 'ay', 'bx', 'by'), (A[0], A[1], B[0], B[1])):
        _init(k(name), round(value, 1))

    marks = {'probe': None, 'profile': None}

    # Probe: the values at the node nearest to (x, y)
    st.markdown('**Probe**')
    left, right = st.columns(2)
    x = left.number_input('x (m)', key=k('probe_x'), format='%g', step=100.0)
    y = right.number_input('y (m)', key=k('probe_y'), format='%g', step=100.0)
    node_x, node_y, rows = views.probe(run, x, y)
    show_probe = st.checkbox('Show the probe on the map', key=k('show_probe'), value=True)
    st.caption('Node used: x = %g m, y = %g m' % (node_x, node_y))
    table = pd.DataFrame({'Field': [label for label, _ in rows],
                          'Value': [v if isinstance(v, str) else '%.4g' % v for _, v in rows]})
    st.dataframe(table, hide_index=True, width='stretch', height=35*len(rows) + 38)
    if show_probe:
        marks['probe'] = (node_x, node_y)

    # Profile: the fields of the figure along A -> B, by default the surface profile
    st.markdown('**Profile A → B**')
    c1, c2, c3, c4 = st.columns(4)
    xa = c1.number_input('A x', key=k('ax'), format='%g', step=100.0)
    ya = c2.number_input('A y', key=k('ay'), format='%g', step=100.0)
    xb = c3.number_input('B x', key=k('bx'), format='%g', step=100.0)
    yb = c4.number_input('B y', key=k('by'), format='%g', step=100.0)

    def surface():
        # Back to the surface profile (callback: runs before the widgets are drawn again)
        for name, value in zip(('ax', 'ay', 'bx', 'by'), (A[0], A[1], B[0], B[1])):
            st.session_state[k(name)] = round(value, 1)

    left, right = st.columns(2)
    show_profile = left.checkbox('Show the line on the map', key=k('show_profile'), value=True)
    right.button('Surface profile', key=k('surface'), on_click=surface,
                 help='Along the top row of nodes, from one side of the grid to the other')
    if (xa, ya) == (xb, yb):
        st.caption('A and B are the same point.')
        return marks['probe'], None
    data = views.profile(run, item, (xa, ya), (xb, yb))
    label = json.dumps([item.key, xa, ya, xb, yb])
    st.image(compute.figure_bytes('profile:' + view, run_key, label, 'png', common.SCREEN_DPI,
                                  lambda: figures.plot_profile(data)), width='stretch')
    csv = pd.DataFrame(data).to_csv(index=False)
    st.download_button('Profile as CSV', csv, file_name='profile_' + item.key + '.csv', mime='text/csv',
                       key=k('profile_csv'), on_click='ignore')
    if show_profile:
        marks['profile'] = ((xa, ya), (xb, yb))
    return marks['probe'], marks['profile']


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
        chosen = st.selectbox('Figure', keys, format_func=labels.get, key=_key(view, 'figure'))
    item = views.figure(view, chosen)
    options = drawing_options(view, item)
    style, step = views.style_for(item, options)
    extra = {'plane_angle': form.get('plane_angle')}

    left, right = st.columns([3, 2])
    with right:
        probe, profile = _probe_and_profile(view, run, item, run_state['key'])
    with left:
        res = item.results(run)
        # Everything that changes the figure goes into the label of its cache entry
        label = json.dumps({'style': style, 'step': step, 'extra': extra, 'probe': probe, 'profile': profile},
                           sort_keys=True, default=list)

        def draw():
            return figures.add_overlays(item.draw(res, run, style, step, extra), probe, profile)

        common.show_figure(view + ':' + item.key, run_state['key'], label, draw, file_stem=item.key)
