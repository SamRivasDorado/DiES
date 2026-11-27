"""
The DiES app (Phase 4, Nov 2026): Streamlit interface on top of the dies package.

    streamlit run dies/app/streamlit_app.py      (from the repository folder)
    dies-app                                     (once installed with pip install -e .[app])

Layout (Sam's choice, 27 Nov 2026): on the left, the definitions common to a project
(scenario file, grid, gravity, loading); on the right, the model setup (layers and
crack) and two tabs, Single model and Dynamic model, each with its own Run button.

At every interaction Streamlit runs this script again from the top. The script reads
the widgets into a form (dies.app.session), builds the two scenarios of the form
(static and dynamic) and checks them, shows each problem next to its field, and draws
the tabs. Nothing is computed until a Run button is pressed, and a scenario already
computed comes back from the cache (dies.app.compute).
"""

import os
import sys

# Run from the repository without installing: the repository folder (two levels up)
# holds the dies package and the modules of the core
try:
    import dies  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import dies  # noqa: F401

import streamlit as st

from dies.app import session, state
from dies.app.ui import common, sidebar, setup, single_model, dynamic_model

st.set_page_config(page_title='DiES', layout='wide')

# First run of the session: the example scenario
session.init()

# Left: definitions of the project. Right: model setup (layers and crack)
side_slots = sidebar.render()
st.markdown('### DiES · Discontinuities in Elastic Space')
st.caption('Units in the forms: m · MPa · GPa · kg/m³ · m/s². Stresses: tension positive.')
layer_rows, crack_row, setup_slots = setup.render()

# The form as it is now, and its two scenarios, checked
form = session.current_form(layer_rows, crack_row)
static_raw = state.raw_from_form(form, 'static')
dynamic_raw = state.raw_from_form(form, 'dynamic')
static_scenario, static_problems = state.check(static_raw)
dynamic_scenario, dynamic_problems = state.check(dynamic_raw)

# Problems of the static scenario next to their fields (they are shared by both tabs)
grouped = common.problems_by_section(static_problems)
common.show_problems(side_slots['grid'], grouped.get('grid'))
common.show_problems(side_slots['gravity'], grouped.get('gravity'))
common.show_problems(side_slots['loading'], grouped.get('remote_stresses', []) + grouped.get('lithostatic', []))
common.show_problems(setup_slots['layers'], grouped.get('layers'))
common.show_problems(setup_slots['cracks'], grouped.get('cracks'))
others = grouped.get('analysis', []) + grouped.get('general', [])
if others:
    common.show_problems(st, others)
# Problems of the dynamic analysis itself (e.g. its step), shown in its tab
dynamic_only = [p for p in dynamic_problems if p not in static_problems]

# The two tabs
single_tab, dynamic_tab = st.tabs(['Single model', 'Dynamic model'])
with single_tab:
    single_model.render(static_scenario, static_problems,
                        state.scenario_key(static_scenario) if static_scenario else None)
with dynamic_tab:
    dynamic_model.render(dynamic_scenario, dynamic_only,
                         state.scenario_key(dynamic_scenario) if dynamic_scenario else None)

# Save button, in its place of the sidebar, now that the whole form is known
sidebar.render_save(side_slots['save'], form)
