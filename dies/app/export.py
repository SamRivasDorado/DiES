"""
Export of the Single model from the app (Phase 4, step 2, Nov 2026): a ZIP file with what
`dies run` writes, i.e. the scenario, the results files and the static figures chosen,
drawn with their default options. No Streamlit here.

The results are not computed again: the files are written from the ScenarioResults of
the last run, through ScenarioResults.write() (dies/scenario.py), with an output section
built from the choices of the Export view.
"""

import copy
import io
import os
import re
import tempfile
import zipfile

from dies.scenario import Scenario, ScenarioResults, STATIC_FIGURES
from dies.app import state
from dies.app.figures import DRAW_LOCK

# Figures of the export by default: those of the dike example, plus the failure state
DEFAULT_FIGURES = ['uxy', 'sxy', 's1', 'angle', 'trajectories', 'plane_normal', 'failure_mode', 'crack_displacements']


def folder_name(name):
    """Name of the folder inside the ZIP: the scenario name as a file name, plus '_results'."""
    return (re.sub(r'[^A-Za-z0-9_-]+', '_', name or 'scenario').strip('_') or 'scenario') + '_results'


def export_zip(run, plane_angle=None, results=('npz', 'csv'), figures=DEFAULT_FIGURES, fmt='png', dpi=200):
    """
    The ZIP file (bytes) of the single model of run (a ScenarioResults), and the names of
    the files in it. plane_angle: the label of the plane figures (None: as in the run, by
    default the optimal angle); results: 'npz' and/or 'csv'; figures: names of
    STATIC_FIGURES; fmt: 'png', 'pdf' or 'svg'; dpi: resolution of the PNG figures.
    """
    unknown = [name for name in figures if name not in STATIC_FIGURES]
    if unknown:
        raise ValueError('unknown figure(s): ' + ', '.join(unknown))
    # The scenario of the run, as a static analysis, with the output section of the export
    raw = copy.deepcopy(run.scenario.raw)
    raw['analysis'] = {'type': 'static'}
    if plane_angle is not None:
        raw['analysis']['plane_angle'] = state.to_quantity(plane_angle, 'angle')
    raw['output'] = {'results': list(results), 'figures': {'static': list(figures)}, 'format': fmt, 'dpi': int(dpi)}
    with tempfile.TemporaryDirectory() as tmp:
        # The scenario as a YAML file, so that the ZIP holds scenario.yaml (write() copies it)
        source = os.path.join(tmp, 'scenario.yaml')
        with open(source, 'w', encoding='utf-8') as f:
            f.write(state.scenario_yaml(raw))
        scenario = Scenario.from_dict(raw, source=source)
        # The results of the run, without computing them again
        exported = ScenarioResults(scenario, run.model, run.crack, run.static, None)
        out = os.path.join(tmp, 'out')
        # write() draws the figures with pyplot: one figure at a time (see dies.app.figures)
        with DRAW_LOCK:
            paths = exported.write(out)
        # Everything into a ZIP, inside a folder named after the scenario
        folder = folder_name(scenario.name)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, folder + '/' + os.path.basename(path))
    return buffer.getvalue(), [os.path.basename(path) for path in paths]
