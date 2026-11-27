"""
Session state of the DiES app (Phase 4, Nov 2026): the values of the widgets, the
scenario opened and the last runs, in st.session_state.

How the state is kept:
- Every widget has a key, and its value lives in st.session_state under that key.
  Opening a scenario writes the values of its form into those keys (load()), and
  current_form() reads them back into a form (dies.app.state).
- The keys carry a version number (e.g. 'grid.x_min@3'). Opening a file, or going back
  to the example, increases the version: the widgets of the new version start with the
  values of the file, and nothing typed for the previous one is mixed in.
- The two tables (layers and crack) are st.data_editor widgets: their starting rows are
  kept in st.session_state (layers_rows, crack_row) and the rows as edited are what the
  widget returns when it is drawn.
- The results of the last run of each tab are kept with the key of the scenario they
  were computed with (static_run, dynamic_run), so that the app can say when the inputs
  have changed since then.
"""

import copy

import streamlit as st

from dies.app import state
from dies.app.defaults import DEFAULT_SCENARIO, DEFAULT_SOURCE


# Widgets with a single value, and their place in the form ('grid.x_min' is
# form['grid']['x_min']). The two tables are handled apart.
SCALAR_FIELDS = [
    'grid.x_min', 'grid.x_max', 'grid.y_base', 'grid.y_top', 'grid.cell_x', 'grid.cell_y',
    'gravity', 'remote_xx', 'remote_yy', 'lithostatic',
    'dynamic.increment', 'dynamic.step', 'dynamic.steps',
]


def key(name):
    """The key of a widget in the current version, e.g. 'gravity@2'."""
    return name + '@' + str(st.session_state['version'])


def _get(form, path):
    # Value at a place of the form: 'grid.x_min' -> form['grid']['x_min']
    value = form
    for part in path.split('.'):
        value = value[part]
    return value


def _set(form, path, value):
    # Sets the value at a place of the form
    parts = path.split('.')
    target = form
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def load(form, source, problems=()):
    """
    Puts a form in the widgets (a new version of their keys): the scenario opened, or
    the example. source: the name shown in the sidebar; problems: those found in the
    file, shown until the next one is opened. The results of previous runs are dropped.
    """
    st.session_state['version'] = st.session_state.get('version', 0) + 1
    st.session_state['form'] = copy.deepcopy(form)
    st.session_state['source'] = source
    st.session_state['load_problems'] = list(problems)
    for name in SCALAR_FIELDS:
        st.session_state[key(name)] = _get(form, name)
    # Analysis saved in the YAML: that of the file opened
    st.session_state[key('save_analysis')] = form['analysis_type']
    # Starting rows of the two tables
    st.session_state['layers_rows'] = copy.deepcopy(form['layers'])
    st.session_state['crack_row'] = copy.deepcopy(form['crack'])
    for name in ('static_run', 'dynamic_run'):
        st.session_state.pop(name, None)


def init():
    """On the first run of a session, opens the example scenario."""
    if 'version' not in st.session_state:
        load(state.form_from_raw(DEFAULT_SCENARIO), DEFAULT_SOURCE)


def load_example():
    """Goes back to the example scenario (button of the sidebar)."""
    load(state.form_from_raw(DEFAULT_SCENARIO), DEFAULT_SOURCE)


def current_form(layers_rows, crack_row):
    """
    The form as it is now: the widget values (from st.session_state), the rows of the
    two tables as edited, and what the app does not edit (name, description, output,
    plane angle) from the scenario opened.
    """
    form = copy.deepcopy(st.session_state['form'])
    for name in SCALAR_FIELDS:
        _set(form, name, st.session_state.get(key(name), _get(form, name)))
    form['layers'] = [dict(row) for row in layers_rows]
    form['crack'] = dict(crack_row)
    return form
