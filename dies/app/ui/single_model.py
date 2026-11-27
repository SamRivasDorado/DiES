"""
The Single model tab: one evaluation of the model (Scenario.run with a static analysis),
with its views: Model (step 1), the views of the results with their drawing options,
probe and profile (step 2, ui/result_view.py and dies/app/views.py) and Export.
"""

import json

import streamlit as st

from dies.app import compute, export, figures, views
from dies.app.ui import common, result_view

# Views of the tab: the model, the views of the results (dies.app.views.VIEWS) and the export
VIEWS = ['Model'] + list(views.VIEWS) + ['Export']


def show_run_status(run, current_key, what):
    """
    One line on the last run of a tab: when it was computed (or that it came from the
    cache) and whether the inputs have changed since then; and the warnings of the core.
    """
    if run is None:
        st.caption('Not computed yet: press Run.')
        return
    if run['key'] != current_key:
        st.warning('The inputs have changed since the last run: the results shown are those of the previous '
                   'inputs. Press Run to update them.')
    results = run['results']
    ny, nx = results.static.shape
    origin = 'from the cache' if run['cached'] else 'computed in %.2g s' % run['elapsed']
    st.caption('%s: %d × %d nodes, %s.' % (what, nx, ny, origin))
    for message in run['warnings']:
        st.warning(message)


def render(scenario, problems, key, form):
    """
    Draws the tab. scenario: the checked static scenario (None if it has problems);
    problems: its problems (shown next to their fields, here only counted); key: its
    cache key; form: the form as it is now.
    """
    # Run: disabled while the scenario has problems
    left, right = st.columns([1, 3])
    with left:
        pressed = st.button('Run single model', key='run_static', type='primary', disabled=scenario is None,
                            width='stretch')
    if pressed:
        with st.spinner('Computing the model...'):
            st.session_state['static_run'] = compute.run(scenario.raw, key)
    with right:
        if scenario is None:
            st.error('%d problem(s) to fix before running: see the messages next to the fields.' % len(problems))
        else:
            show_run_status(st.session_state.get('static_run'), key, 'Single model')

    tabs = st.tabs(VIEWS)
    # Model: the sketch of the model and the profile of the crack, without computing it
    with tabs[0]:
        if scenario is None:
            st.info('The sketch of the model appears when the inputs have no problems.')
        else:
            left, right = st.columns([3, 2])
            with left:
                common.show_figure('model', key, scenario.name, lambda: figures.plot_model(scenario),
                                   file_stem='model')
            with right:
                common.show_figure('crack_displacements', key, '', lambda: figures.plot_crack(scenario),
                                   file_stem='crack_displacements')
                st.caption('Walls of the crack under its final driving stresses, with E and ν of the layer '
                           'at its centre.')
    # The views of the results, from the last run (also when the inputs have changed since)
    for view, tab in zip(views.VIEWS, tabs[1:-1]):
        with tab:
            result_view.render(view, st.session_state.get('static_run'), form)
    with tabs[-1]:
        render_export(st.session_state.get('static_run'), form)


def render_export(run_state, form):
    """
    The Export view: a ZIP with the scenario, the results files and the static figures
    chosen, as `dies run` writes them (dies/app/export.py). The ZIP is built when asked,
    and kept for the same run and choices.
    """
    if run_state is None:
        st.info('Press **Run single model** to compute the model and export it.')
        return
    from dies.scenario import STATIC_FIGURES
    run = run_state['results']
    st.markdown('A ZIP file with what `dies run` writes: the scenario (`scenario.yaml`), the results and the '
                'figures chosen. The figures are drawn with their default options; to keep a figure as you see '
                'it in a view, use the PNG, PDF and SVG buttons under it.')
    left, middle, right = st.columns([2, 4, 2])
    results = left.multiselect('Results files', ['npz', 'csv'], default=['npz', 'csv'], key='export.results',
                               help='npz: every field, to reopen with DiESResults.load; csv: one row per node')
    names = list(STATIC_FIGURES)
    chosen = middle.multiselect('Figures', names, default=export.DEFAULT_FIGURES, key='export.figures',
                                help='Names as in the output section of a scenario (docs/scenario.md)')
    fmt = right.selectbox('Figure format', ['png', 'pdf', 'svg'], key='export.format')
    dpi = right.number_input('Resolution (dpi)', 50, 600, 200, 50, key='export.dpi', disabled=fmt != 'png')
    # The ZIP is built only when asked (drawing many figures takes a few seconds)
    request = json.dumps([run_state['key'], form.get('plane_angle'), results, chosen, fmt, dpi])
    if st.button('Prepare the ZIP', key='export.prepare', disabled=not (results or chosen)):
        with st.spinner('Writing the results and drawing the figures...'):
            data, files = export.export_zip(run, form.get('plane_angle'), results, chosen, fmt, dpi)
        st.session_state['export.zip'] = (request, data, files)
    prepared = st.session_state.get('export.zip')
    if prepared and prepared[0] == request:
        _, data, files = prepared
        st.download_button('Download the ZIP (%.1f MB)' % (len(data)/1e6), data,
                           file_name=export.folder_name(run.scenario.name) + '.zip', mime='application/zip',
                           type='primary', key='export.download', on_click='ignore')
        st.caption('%d files: %s' % (len(files), ', '.join(files)))
    elif prepared:
        st.caption('The choices or the results have changed: prepare the ZIP again.')
