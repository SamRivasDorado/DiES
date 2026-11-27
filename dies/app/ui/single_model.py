"""
The Single model tab: one evaluation of the model (Scenario.run with a static analysis),
with its views. Phase 4, step 1: the Run button and the Model view; the views of the
results (displacements, stresses, principal stresses and trajectories, planes and
failure) and the export come in step 2.
"""

import streamlit as st

from dies.app import compute, figures
from dies.app.ui import common

# Views of the tab
VIEWS = ['Model', 'Displacements', 'Stresses', 'Principal & trajectories', 'Planes & failure', 'Export']


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


def render(scenario, problems, key):
    """
    Draws the tab. scenario: the checked static scenario (None if it has problems);
    problems: its problems (shown next to their fields, here only counted); key: its
    cache key.
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
    for tab in tabs[1:]:
        with tab:
            st.info('This view comes in the next step of the app (Phase 4, step 2).')
