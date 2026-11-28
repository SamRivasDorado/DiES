"""
The Dynamic model tab: the driving stresses of the crack grow step by step and the
failure of each node is followed (Scenario.run with a dynamic analysis, i.e.
compute_dynamic_model).

Phase 4, step 1: the increment, the Run button and a summary of the final step.
Phase 4, step 3 (28 Nov 2026, Sam: "movernos paso a paso ... y darle al play y ver la
pelicula"): a player of the steps, for every output of dies/app/dynamic_views.py. A
slider and buttons move through the steps; Play shows them as a movie. The probes and
profiles (shared with the Single model) give their values at the step shown.

The player is a fragment (st.fragment): while it plays, only the fragment runs again,
every 1/fps seconds, and moves on one step. Before playing, the figures of all the steps
are drawn once (with a progress bar) and cached, so that the movie runs smoothly.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st

import plotting

from dies.app import compute, dynamic_views, export, figures, session, views
from dies.app.ui import common, result_view
from dies.app.ui.single_model import show_run_status

# Name of the player in the keys of its widgets (and of the probe and profile widgets)
VIEW = 'Dynamic'

# Speeds of the movie, in steps (frames) per second
SPEEDS = [0.5, 1, 2, 3, 4, 6]


def render_increment():
    """
    The increment of the driving stresses: the stress added at each step, or the number
    of steps. Both fields are always drawn (the one not chosen is disabled), so that the
    value of each one is kept when switching between them.
    """
    choice = st.radio('Increment', ['step', 'steps'], horizontal=True, key=session.key('dynamic.increment'),
                      format_func=lambda c: 'Driving stress per step' if c == 'step' else 'Number of steps',
                      help='With a stress per step, the number of steps is rounded so that the last one reaches '
                           'the final load exactly (8 MPa with 0.25 MPa: 32 steps).')
    left, right, _ = st.columns([1, 1, 2])
    with left:
        st.number_input('Step (MPa)', key=session.key('dynamic.step'), value=None, step=0.05, format='%g',
                        disabled=choice != 'step')
    with right:
        st.number_input('Steps', key=session.key('dynamic.steps'), value=None, step=1, min_value=1, format='%d',
                        disabled=choice != 'steps')


def render(scenario, problems, key, form=None):
    """
    Draws the tab. scenario: the checked dynamic scenario (None if it has problems);
    problems: the problems of the dynamic analysis itself (those shared with the single
    model are shown next to their fields); key: its cache key; form: the form as it is now
    (for the trace spacing of the scenario file, output.figures.dynamic.downscaling).
    """
    render_increment()
    common.show_problems(st, problems)

    left, right = st.columns([1, 3])
    with left:
        pressed = st.button('Run dynamic model', key='run_dynamic', type='primary', disabled=scenario is None,
                            width='stretch')
    if pressed:
        with st.spinner('Computing the dynamic model, step by step...'):
            st.session_state['dynamic_run'] = compute.run(scenario.raw, key)
    run = st.session_state.get('dynamic_run')
    with right:
        if scenario is None:
            st.error('Problems to fix before running: see the messages above and next to the fields.')
        else:
            show_run_status(run, key, 'Dynamic model')

    if run is None:
        return
    # Summary of the run: steps, final driving stresses and the failure mode of the nodes
    # at the last step
    dyn = run['results'].dynamic
    last = dyn[-1].metadata
    st.markdown('**%d steps.** Final driving stresses: σI = %g MPa, σII = %g MPa, σIII = %g MPa.'
                % (len(dyn), round(last['dx']/1e6, 3), round(last['dy']/1e6, 3), round(last['dz']/1e6, 3)))
    # Nodes of each failure state at the last step, with their share of the nodes that fail
    # (as in the legend of the failure-mode map; Sam, 27 Nov 2026): 'Below' is left out
    shares = plotting.failure_mode_shares(dyn.final_mode())
    rows = [(mode, n, '%.1f %%' % percent if percent is not None else '') for mode, n, percent in shares
            if mode != 'Below']
    below = sum(n for mode, n, _ in shares if mode == 'Below')
    if rows:
        table = pd.DataFrame(rows, columns=['Failure state at the last step', 'Nodes', 'Share of the nodes that fail'])
        st.dataframe(table, hide_index=True, width='content')
    first = dyn.failure_step()
    failed = first > 0
    st.caption('Nodes below failure at the last step: %d of %d.' % (below, first.size)
               + (' First failures at step %d.' % int(first[failed].min()) if failed.any() else ''))
    st.divider()
    render_player(run, trace_spacing(form))
    st.divider()
    render_export(run, form or {}, trace_spacing(form))


def trace_spacing(form):
    """
    The spacing of the plane traces in the scenario file (output.figures.dynamic.downscaling;
    the scenarios computed by the app have no output section), 1 if it does not give one.
    """
    output = (form or {}).get('output') or {}
    try:
        return max(int(((output.get('figures') or {}).get('dynamic') or {}).get('downscaling', 1)), 1)
    except (TypeError, ValueError):
        return 1


# ---------------------------------------------------------------------- drawing options

def _k(name):
    # Key of a widget of the player
    return VIEW + '.' + name


def _spacing_label(spacing, cell, default):
    # Text of a trace spacing: 'as the scenario (every 2 nodes)' or 'every 3 nodes (300 m)'
    if not spacing:
        return 'as the scenario (every %d node%s)' % (default, '' if default == 1 else 's')
    return 'every node (%g m)' % cell if spacing == 1 else 'every %d nodes (%g m)' % (spacing, spacing*cell)


def drawing_options(item, cell, downscaling):
    """
    The drawing options of an output of the player, in an expander (as those of the Single
    model; the widgets that the output does not take are disabled). Returns a dict with the
    options of plotting.STYLE_DEFAULTS and those of the app (dynamic_views.APP_OPTIONS).
    cell: the smallest spacing of the nodes (m); downscaling: the trace spacing of the scenario.
    """
    takes = lambda name: name in item.options
    unit = (' (' + item.unit + ')') if item.unit else ''
    options = {}
    with st.expander('Drawing options', expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            options['cmap'] = st.selectbox('Colour map', views.COLOUR_MAPS, key=_k('cmap'), disabled=not takes('cmap'),
                                           format_func=lambda name: name or 'figure default') or None
            options['same_range'] = st.checkbox('Same colours in every step', key=_k('same_range'), value=True,
                                                disabled=not takes('same_range'),
                                                help='The colour range of the final step in every step, so that '
                                                     'the colours of the movie can be compared')
            options['center_zero'] = st.checkbox('Colours centred on 0', key=_k('center_zero'),
                                                 disabled=not takes('center_zero'))
        with c2:
            manual = st.checkbox('Manual colour limits', key=_k('manual_limits'), disabled=not takes('limits'))
            low, high = st.columns(2)
            vmin = low.number_input('min' + unit, key=_k('vmin'), value=None, format='%g', disabled=not manual)
            vmax = high.number_input('max' + unit, key=_k('vmax'), value=None, format='%g', disabled=not manual)
            options['limits'] = None
            if takes('limits') and manual:
                if vmin is None or vmax is None or not vmin < vmax:
                    st.caption('Give min < max to use them.')
                else:
                    options['limits'] = (vmin, vmax)
            options['unit'] = st.selectbox('First failure as', list(dynamic_views.FIRST_FAILURE_UNITS),
                                           key=_k('unit'), disabled=not takes('unit'),
                                           format_func=dynamic_views.FIRST_FAILURE_UNITS.get)
        with c3:
            options['contours'] = st.checkbox('Contours', key=_k('contours'), value=True,
                                              disabled=not takes('contours'),
                                              help='In the σn map: the line σn = 0')
            options['contour_labels'] = st.checkbox('Contour labels', key=_k('contour_labels'), value=True,
                                                    disabled=not (takes('contour_labels') and options['contours']))
            options['above'] = st.checkbox('Mark nodes above failure', key=_k('above'), value=True,
                                           disabled=not takes('above'), help='Red mesh over the nodes that fail')
            options['share'] = st.checkbox('As % of all the nodes', key=_k('share'), disabled=not takes('share'))
        with c4:
            options['info_box'] = st.checkbox('Text box', key=_k('info_box'), value=True,
                                              disabled=not takes('info_box'))
            options['traces'] = st.checkbox('Traces of the failure planes', key=_k('traces'), value=True,
                                            disabled=not takes('traces'))
            options['arrow_scale'] = st.slider('Trace length', 0.25, 4.0, 1.0, 0.25, key=_k('arrow_scale'),
                                               disabled=not (takes('arrow_scale') and options['traces']))
            spacing = st.selectbox('Trace density', views.ARROW_SPACINGS, key=_k('arrow_spacing'),
                                   disabled=not (takes('arrow_spacing') and options['traces']),
                                   format_func=lambda n: _spacing_label(n, cell, downscaling),
                                   help='One trace every n nodes of the grid')
            options['arrow_spacing'] = spacing or downscaling
    return options


# -------------------------------------------------------------------------------- player

def _move(where, n):
    # Callback of the step buttons ('first', 'previous', 'next', 'last'): the movie pauses
    step = st.session_state[_k('step')]
    st.session_state[_k('step')] = {'first': 1, 'previous': max(step - 1, 1), 'next': min(step + 1, n),
                                    'last': n}[where]
    st.session_state[_k('playing')] = False


def _frame_label(item, style, options, probe_marks, profile_marks, k):
    # Everything that changes a frame, for its cache entry
    return json.dumps({'output': item.key, 'style': style, 'options': {name: options.get(name)
                                                                         for name in dynamic_views.APP_OPTIONS},
                       'probes': probe_marks, 'profiles': profile_marks, 'step': k}, sort_keys=True, default=list)


def _frame_bytes(run_state, item, style, options, probe_marks, profile_marks, k):
    # The PNG of step k on screen (cached)
    dyn = run_state['results'].dynamic
    overlays = (probe_marks, profile_marks) if item.is_map else ([], [])

    def draw():
        return figures.add_overlays(item.draw(dyn, k, dict(style), options), *overlays)

    label = _frame_label(item, style, options, probe_marks, profile_marks, k)
    return label, draw


def render_player(run_state, downscaling=1):
    """
    The player of the steps of a dynamic run (run_state: dies.app.compute.run): output,
    drawing options, the step controls, the figure of the step, its probes and profiles.
    downscaling: the default spacing of the plane traces (trace_spacing).
    """
    run = run_state['results']
    dyn = run.dynamic
    n = len(dyn)
    st.session_state.setdefault(_k('step'), n)
    st.session_state.setdefault(_k('playing'), False)
    if st.session_state[_k('step')] > n:
        st.session_state[_k('step')] = n

    st.markdown('**Steps of the run**')
    outputs = [item.key for item in dynamic_views.OUTPUTS]
    labels = {item.key: item.label for item in dynamic_views.OUTPUTS}
    left, _ = st.columns([3, 2])
    chosen = left.selectbox('Output', outputs, format_func=labels.get, key=_k('output'))
    item = dynamic_views.output(chosen)
    static = dyn[0]
    cell = min(float(np.min(np.abs(np.diff(v)))) for v in (static.x_nodes(), static.y_nodes()))
    options = drawing_options(item, cell, downscaling)
    style = dynamic_views.style_for(item, options)

    # Play / Pause: the whole page runs again, so that the fragment gets its new timer
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1.4, 1, 1, 1.2, 1.2])
    playing = st.session_state[_k('playing')]
    step = st.session_state[_k('step')]
    c1.button('⏮', key=_k('first'), on_click=_move, args=('first', n), help='First step', width='stretch')
    c2.button('◀', key=_k('previous'), on_click=_move, args=('previous', n), help='Previous step',
              width='stretch')
    if c3.button('⏸ Pause' if playing else '▶ Play', key=_k('play'), type='primary', width='stretch'):
        st.session_state[_k('playing')] = not playing
        if not playing and step >= n:
            st.session_state[_k('step')] = 1   # Play at the last step: from the start
        st.rerun()
    c4.button('▶', key=_k('next'), on_click=_move, args=('next', n), help='Next step', width='stretch')
    c5.button('⏭', key=_k('last'), on_click=_move, args=('last', n), help='Last step', width='stretch')
    fps = c6.selectbox('Speed', SPEEDS, index=2, key=_k('fps'), format_func=lambda v: '%g steps/s' % v,
                       label_visibility='collapsed')
    loop = c7.checkbox('Loop', key=_k('loop'), help='Start again after the last step')

    @st.fragment(run_every=1.0/fps if playing else None)
    def player():
        playing = st.session_state[_k('playing')]
        if playing and st.session_state.get(_k('ticked')):
            # One step on (the first run after Play shows the current step)
            step = st.session_state[_k('step')] + 1
            if step > n:
                if loop:
                    step = 1
                else:
                    st.session_state[_k('playing')] = False
                    st.session_state[_k('ticked')] = False
                    st.rerun()
            st.session_state[_k('step')] = step
        st.session_state[_k('ticked')] = playing
        k = st.slider('Step', 1, n, key=_k('step'), disabled=playing)
        st.caption(dynamic_views.step_label(dyn, k))

        # The figure goes first but is drawn last, once the probes and profiles are known
        _, middle, _ = st.columns([1, 3, 1])
        slot = middle.empty()
        probe_marks = result_view.probes(VIEW, run, table=lambda points: dynamic_views.probe_table(run, k, points))
        profile_marks = []
        if item.profile:
            profile_marks = result_view.profiles(VIEW, run, run_state['key'], '%s_step%d' % (item.key, k),
                                                 lambda A, B: dynamic_views.profile(run, k, item, A, B))
        with slot.container():
            if playing:
                _prepare(run_state, item, style, options, probe_marks, profile_marks, n)
            label, draw = _frame_bytes(run_state, item, style, options, probe_marks, profile_marks, k)
            png = compute.figure_bytes('dynamic', run_state['key'], label, 'png', common.SCREEN_DPI, draw)
            st.image(png, width='stretch')
            # The PNG on screen can be downloaded at once, and the movie of the output as it is
            # drawn (with its options, probes and profiles) as an animated GIF
            if not playing:
                _downloads(run_state, item, style, options, probe_marks, profile_marks, n, k, png)

    player()


def _prepare(run_state, item, style, options, probe_marks, profile_marks, n):
    """
    Before a movie: the figures of all its steps, drawn once and cached (with a progress
    bar), so that it then plays at its speed. Skipped if they are already cached.
    """
    done = st.session_state.setdefault(_k('prepared'), set())
    ident = (run_state['key'], _frame_label(item, style, options, probe_marks, profile_marks, 0))
    if ident in done:
        return
    bar = st.progress(0.0, text='Drawing the steps of the movie...')
    for k in range(1, n + 1):
        label, draw = _frame_bytes(run_state, item, style, options, probe_marks, profile_marks, k)
        compute.figure_bytes('dynamic', run_state['key'], label, 'png', common.SCREEN_DPI, draw)
        bar.progress(k/n, text='Drawing the steps of the movie: %d of %d' % (k, n))
    bar.empty()
    done.add(ident)


def _downloads(run_state, item, style, options, probe_marks, profile_marks, n, k, png):
    """
    Under the figure of the player: the PNG of the step, and the movie of all the steps as
    an animated GIF (made when asked, at the speed of the player, and kept for the same
    figures and speed).
    """
    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.download_button('PNG of this step', png, file_name='%s_step_%d.png' % (item.key, k), mime='image/png',
                       key=_k('download'), on_click='ignore', width='stretch')
    fps = st.session_state.get(_k('fps'), 2)
    request = (run_state['key'], _frame_label(item, style, options, probe_marks, profile_marks, 0), fps)
    if c2.button('Movie (GIF)', key=_k('movie'), width='stretch',
                 help='All the steps of this output as an animated GIF, at the speed of the player'):
        _prepare(run_state, item, style, options, probe_marks, profile_marks, n)
        frames = []
        for step in range(1, n + 1):
            label, draw = _frame_bytes(run_state, item, style, options, probe_marks, profile_marks, step)
            frames.append(compute.figure_bytes('dynamic', run_state['key'], label, 'png', common.SCREEN_DPI, draw))
        with st.spinner('Making the GIF...'):
            st.session_state[_k('gif')] = (request, export.movie_gif(frames, fps))
    movie = st.session_state.get(_k('gif'))
    if movie and movie[0] == request:
        c3.download_button('Download the GIF (%.1f MB)' % (len(movie[1])/1e6), movie[1],
                           file_name='%s_movie.gif' % item.key, mime='image/gif', type='primary',
                           key=_k('gif_download'), on_click='ignore', width='stretch')


def render_export(run_state, form, downscaling):
    """
    Export of the dynamic run: a ZIP with what `dies run` writes for a dynamic scenario (the
    scenario, the static and dynamic results, the static figures chosen and the figure of
    one step in n), drawn with their default options (dies/app/export.py).
    """
    from dies.scenario import STATIC_FIGURES
    run = run_state['results']
    st.markdown('**Export**: a ZIP file with what `dies run` writes for this dynamic scenario: the scenario, the '
                'results of the final load (static) and of every step (dynamic), and the figures chosen, drawn '
                'with their default options.')
    c1, c2, c3, c4 = st.columns([2, 4, 2, 2])
    results = c1.multiselect('Results files', ['npz', 'csv'], default=['npz', 'csv'], key='dynexport.results',
                             help='npz: every field, of the final load and of every step; csv: one row per node, '
                                  'of the final load and of the last step (with the step of first failure)')
    chosen = c2.multiselect('Figures of the final load', list(STATIC_FIGURES), default=['failure_mode'],
                            key='dynexport.figures', help='Names as in the output section of a scenario')
    every = c3.number_input('Figure of one step in', 0, len(run.dynamic), min(4, len(run.dynamic)), 1,
                            key='dynexport.every', help='dynamic_step_<step> figures (σn and failure planes): one '
                                                        'step in n, and the last one; 0: none')
    spacing = c3.number_input('Trace spacing (nodes)', 1, 50, downscaling, 1, key='dynexport.spacing')
    fmt = c4.selectbox('Figure format', ['png', 'pdf', 'svg'], key='dynexport.format')
    dpi = c4.number_input('Resolution (dpi)', 50, 600, 200, 50, key='dynexport.dpi', disabled=fmt != 'png')
    request = json.dumps([run_state['key'], form.get('plane_angle'), results, chosen, every, spacing, fmt, dpi])
    if st.button('Prepare the ZIP', key='dynexport.prepare', disabled=not (results or chosen or every)):
        with st.spinner('Writing the results and drawing the figures...'):
            data, files = export.export_zip(run, form.get('plane_angle'), results, chosen, fmt, dpi,
                                            dynamic={'show_every': every, 'downscaling': spacing})
        st.session_state['dynexport.zip'] = (request, data, files)
    prepared = st.session_state.get('dynexport.zip')
    if prepared and prepared[0] == request:
        _, data, files = prepared
        st.download_button('Download the ZIP (%.1f MB)' % (len(data)/1e6), data,
                           file_name=export.folder_name(run.scenario.name) + '_dynamic.zip', mime='application/zip',
                           type='primary', key='dynexport.download', on_click='ignore')
        st.caption('%d files: %s' % (len(files), ', '.join(files)))
    elif prepared:
        st.caption('The choices or the results have changed: prepare the ZIP again.')
