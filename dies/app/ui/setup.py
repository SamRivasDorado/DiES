"""
The model setup, above the two tabs: the layers and the crack, as editable tables
(st.data_editor). Shared by the single and the dynamic model.
"""

import pandas as pd
import streamlit as st

from dies.app import session, state


def _column_config(columns):
    # Header and type of each column (text or number)
    config = {}
    for key, kind, header in columns:
        if kind == 'text':
            config[key] = st.column_config.TextColumn(header)
        else:
            config[key] = st.column_config.NumberColumn(header, format='%g')
    return config


def _frame(rows, columns):
    # Table of rows with the number columns as floats (an empty cell is NaN), so that
    # the editor treats them as numbers even when a whole column is empty
    frame = pd.DataFrame(rows, columns=[key for key, _, _ in columns])
    for key, kind, _ in columns:
        if kind != 'text':
            frame[key] = pd.to_numeric(frame[key], errors='coerce').astype(float)
    return frame


def _rows(frame, columns):
    # Rows of an edited table as dicts, with empty cells as None
    rows = []
    for record in frame.to_dict('records'):
        rows.append({key: (None if state.is_blank(record.get(key)) else record.get(key)) for key, _, _ in columns})
    return rows


def render():
    """
    Draws the model setup. Returns (layer rows, crack row, slots), where slots holds the
    containers for the problems of the layers ('layers') and of the crack ('cracks').
    """
    slots = {}
    with st.expander('Model setup: layers and crack', expanded=True):
        # Layers, from the top down; rows can be added and removed
        st.markdown('**Layers** (from the top down; the first one starts at the surface, top y = 0)')
        layers = _frame(st.session_state['layers_rows'], state.LAYER_COLUMNS)
        edited = st.data_editor(layers, key=session.key('layers_table'), num_rows='dynamic', hide_index=True,
                                column_config=_column_config(state.LAYER_COLUMNS), width='stretch')
        layer_rows = _rows(edited, state.LAYER_COLUMNS)
        # The cohesion is implicit, C = 2|T| (not an input)
        cohesions = ['%s: %g MPa' % (row['name'] or 'layer ' + str(i + 1), state.cohesion(row['T']))
                     for i, row in enumerate(layer_rows) if state.cohesion(row['T']) is not None]
        if cohesions:
            st.caption('Cohesion (implicit, C = 2|T|): ' + ' · '.join(cohesions))
        slots['layers'] = st.container()

        # The crack (one for now): aperture or σI, not both
        st.markdown('**Crack** (give the aperture or σI and leave the other one empty; σII > 0: normal sense)')
        crack = _frame([st.session_state['crack_row']], state.CRACK_COLUMNS)
        edited = st.data_editor(crack, key=session.key('crack_table'), num_rows='fixed', hide_index=True,
                                column_config=_column_config(state.CRACK_COLUMNS), width='stretch')
        crack_rows = _rows(edited, state.CRACK_COLUMNS)
        slots['cracks'] = st.container()
    return layer_rows, crack_rows[0], slots
