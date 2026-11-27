"""
The sidebar: the definitions common to a project (Sam's layout, 27 Nov 2026), i.e. the
scenario file (open, save, back to the example), the grid, the gravity and the loading
(remote stresses and lithostatic stresses). The layers and the crack are in the model
setup, above the tabs, because their tables need the width of the page.
"""

import os

import streamlit as st

from dies.app import session, state
from dies.scenario import ScenarioError


def _on_upload():
    # Callback of the file uploader: reads the file and puts it in the widgets. It runs
    # before the script, so the widgets are drawn with the values of the file
    uploaded = st.session_state.get(session.key('upload'))
    if uploaded is None:
        return
    try:
        raw = state.read_scenario_text(uploaded.getvalue().decode('utf-8'), uploaded.name)
    except (ScenarioError, UnicodeDecodeError) as error:
        st.session_state['upload_error'] = getattr(error, 'problems', [str(error)])
        return
    st.session_state.pop('upload_error', None)
    # The problems of the file are kept and shown: some of them (unknown keys, the output
    # section) have no field in the forms
    _, problems = state.check(raw)
    session.load(state.form_from_raw(raw), uploaded.name, problems)


def _number(label, name, step, help=None, fmt='%g'):
    # A number field bound to its widget key (its value lives in st.session_state).
    # value=None lets the field be emptied: the scenario check then reports it as
    # missing, instead of the field going back to 0 silently
    return st.number_input(label, key=session.key(name), value=None, step=step, format=fmt, help=help)


def render():
    """
    Draws the sidebar. Returns the containers where the app writes, once the whole form
    is known, the problems of each section ('grid', 'gravity', 'loading') and the save
    button ('save').
    """
    slots = {}
    with st.sidebar:
        st.markdown('## DiES')
        st.caption('Discontinuities in Elastic Space')

        # Scenario file: open (YAML or JSON), save (filled in at the end), back to the example
        st.markdown('#### Scenario')
        st.file_uploader('Open a scenario (.yaml, .yml, .json)', type=['yaml', 'yml', 'json'],
                         key=session.key('upload'), on_change=_on_upload)
        if st.session_state.get('upload_error'):
            st.error('\n'.join('- ' + p for p in st.session_state['upload_error']))
        st.caption('Open: ' + st.session_state['source'])
        if st.session_state['load_problems']:
            with st.expander('Problems in the file opened (%d)' % len(st.session_state['load_problems'])):
                st.markdown('\n'.join('- ' + p for p in st.session_state['load_problems']))
        slots['save'] = st.container()
        st.button('Back to the example', on_click=session.load_example)

        # Grid limits and cell size, in m (y negative downwards; the top below the surface)
        st.markdown('#### Grid (m)')
        left, right = st.columns(2)
        with left:
            _number('x min', 'grid.x_min', 500.0)
            _number('y base', 'grid.y_base', 500.0)
            _number('cell x', 'grid.cell_x', 10.0)
        with right:
            _number('x max', 'grid.x_max', 500.0)
            _number('y top', 'grid.y_top', 10.0, help='Top of the grid, below the surface (y < 0), e.g. -10 m')
            _number('cell y', 'grid.cell_y', 10.0)
        slots['grid'] = st.container()

        # Gravity: any value, so that any planet or exotic case can be modelled
        st.markdown('#### Gravity')
        _number('g (m/s²)', 'gravity', 0.1,
                help='Any value: e.g. 9.81 (Earth), 3.71 (Mars), 1.62 (Moon), 0 (no gravity)')
        slots['gravity'] = st.container()

        # Loading: constant remote stresses (tension positive) and the lithostatic stresses
        st.markdown('#### Loading')
        left, right = st.columns(2)
        with left:
            _number('σxx remote (MPa)', 'remote_xx', 1.0, help='Tension positive')
        with right:
            _number('σyy remote (MPa)', 'remote_yy', 1.0, help='Tension positive')
        st.checkbox('Lithostatic stresses', key=session.key('lithostatic'),
                    help='Add the lithostatic stresses of the layers')
        slots['loading'] = st.container()
    return slots


def render_save(slot, form):
    """
    The save button, in its place of the sidebar: the scenario of the form as YAML, with
    the analysis chosen (static or dynamic) and the output section of the file opened.
    Disabled while that scenario has problems.
    """
    with slot:
        analysis = st.radio('Analysis in the saved file', ['static', 'dynamic'], horizontal=True,
                            key=session.key('save_analysis'))
        raw = state.raw_from_form(form, analysis, include_output=True)
        _, problems = state.check(raw)
        source = st.session_state['source']
        stem = os.path.splitext(source)[0] if source.lower().endswith(('.yaml', '.yml', '.json')) else 'scenario'
        st.download_button('Save .yaml', state.scenario_yaml(raw), file_name=stem + '.yaml',
                           mime='application/x-yaml', disabled=bool(problems), on_click='ignore',
                           help='Fix the problems shown to save' if problems else None)
