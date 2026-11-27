"""
The Dynamic model tab: the driving stresses of the crack grow step by step and the
failure of each node is followed (Scenario.run with a dynamic analysis, i.e.
compute_dynamic_model). Phase 4, step 1: the increment, the Run button and a summary
of the final step; the figures of the steps come in step 3.
"""

import pandas as pd
import streamlit as st

import plotting

from dies.app import compute, session
from dies.app.ui import common
from dies.app.ui.single_model import show_run_status


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


def render(scenario, problems, key):
    """
    Draws the tab. scenario: the checked dynamic scenario (None if it has problems);
    problems: the problems of the dynamic analysis itself (those shared with the single
    model are shown next to their fields); key: its cache key.
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
    st.info('The figures of the steps (σn and failure planes, failure-mode map, step of first failure) come in '
            'step 3 of the app.')
