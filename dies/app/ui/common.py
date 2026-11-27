"""Helpers shared by the parts of the interface: problems and figures with their downloads."""

import re

import streamlit as st

from dies.app import compute, state

# Formats offered for download, and their MIME types
FIGURE_FORMATS = {'png': 'image/png', 'pdf': 'application/pdf', 'svg': 'image/svg+xml'}

# Resolution of the figures on screen and in the PNG download
SCREEN_DPI = 110
DOWNLOAD_DPI = 200


def show_problems(container, problems):
    """Writes each problem, in the words of the forms, in container (nothing if there are none)."""
    if problems:
        container.error('\n'.join('- ' + state.describe_problem(p) for p in problems))


def problems_by_section(problems):
    """{section: [problems]} (sections of dies.app.state.SECTIONS, or 'general')."""
    grouped = {}
    for problem in problems:
        grouped.setdefault(state.problem_section(problem), []).append(problem)
    return grouped


def _file_name(text):
    # A text as a file name: letters, digits, '-' and '_'
    return re.sub(r'[^A-Za-z0-9_-]+', '_', text).strip('_') or 'figure'


def show_figure(name, key, label, draw, file_stem=None):
    """
    Shows a figure (drawn by draw(), cached by name, scenario key and label) and the
    buttons to download it as PNG, PDF or SVG. Each format is drawn once and cached.
    """
    st.image(compute.figure_bytes(name, key, label, 'png', SCREEN_DPI, draw), width='stretch')
    stem = _file_name(file_stem or name)
    columns = st.columns(len(FIGURE_FORMATS) + 3)
    for column, (fmt, mime) in zip(columns, FIGURE_FORMATS.items()):
        dpi = DOWNLOAD_DPI if fmt == 'png' else SCREEN_DPI
        column.download_button(fmt.upper(), compute.figure_bytes(name, key, label, fmt, dpi, draw),
                               file_name=stem + '.' + fmt, mime=mime, key='download_' + name + '_' + fmt,
                               on_click='ignore')
