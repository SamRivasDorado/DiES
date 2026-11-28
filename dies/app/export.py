"""
Export from the app (Phase 4, Nov 2026): a ZIP file with what `dies run` writes, i.e. the
scenario, the results files and the figures chosen, drawn with their default options;
for the Single model (step 2) and the Dynamic model (step 3, with its steps). Also the
movie of the steps of a dynamic run, as an animated GIF (movie_gif). No Streamlit here.

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


def export_zip(run, plane_angle=None, results=('npz', 'csv'), figures=DEFAULT_FIGURES, fmt='png', dpi=200,
               dynamic=None):
    """
    The ZIP file (bytes) of run (a ScenarioResults), and the names of the files in it.
    plane_angle: the label of the plane figures (None: as in the run, by default the
    optimal angle); results: 'npz' and/or 'csv'; figures: names of STATIC_FIGURES; fmt:
    'png', 'pdf' or 'svg'; dpi: resolution of the PNG figures.
    dynamic: None for the single model (a static analysis); for a dynamic run, the
    output.figures.dynamic section, {'show_every': n, 'downscaling': m} (the figure of
    one step in n, and the last one: dynamic_step_<step>; show_every 0: no step figures),
    and the dynamic results are written too (dynamic_results.npz, dynamic_final_step.csv).
    """
    unknown = [name for name in figures if name not in STATIC_FIGURES]
    if unknown:
        raise ValueError('unknown figure(s): ' + ', '.join(unknown))
    if dynamic is not None and run.dynamic is None:
        raise ValueError('a dynamic export needs the results of a dynamic run')
    # The scenario of the run (as a static analysis for the single model), with the output
    # section of the export
    raw = copy.deepcopy(run.scenario.raw)
    raw['analysis'] = dict(raw.get('analysis') or {}) if dynamic is not None else {'type': 'static'}
    raw['analysis'].pop('plane_angle', None)
    if plane_angle is not None:
        raw['analysis']['plane_angle'] = state.to_quantity(plane_angle, 'angle')
    sections = {'static': list(figures)}
    if dynamic is not None and dynamic.get('show_every'):
        sections['dynamic'] = {'show_every': int(dynamic['show_every']),
                               'downscaling': int(dynamic.get('downscaling') or 1)}
    raw['output'] = {'results': list(results), 'figures': sections, 'format': fmt, 'dpi': int(dpi)}
    with tempfile.TemporaryDirectory() as tmp:
        # The scenario as a YAML file, so that the ZIP holds scenario.yaml (write() copies it)
        source = os.path.join(tmp, 'scenario.yaml')
        with open(source, 'w', encoding='utf-8') as f:
            f.write(state.scenario_yaml(raw))
        scenario = Scenario.from_dict(raw, source=source)
        # The results of the run, without computing them again
        exported = ScenarioResults(scenario, run.model, run.crack, run.static,
                                   run.dynamic if dynamic is not None else None)
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


def movie_gif(frames, fps=2.0, width=900):
    """
    An animated GIF (bytes) of the frames (PNG bytes, e.g. the figures of the steps of the
    player), fps frames per second, looping. The frames are scaled to width pixels (keeping
    their shape) and share the palette of the first one, so that the colours do not flicker.
    """
    from PIL import Image
    if not frames:
        raise ValueError('a movie needs at least one frame')
    images = []
    for data in frames:
        image = Image.open(io.BytesIO(data)).convert('RGB')
        if image.width != width:
            image = image.resize((width, max(int(round(image.height*width/image.width)), 1)), Image.LANCZOS)
        images.append(image)
    size = images[0].size
    images = [image if image.size == size else image.resize(size, Image.LANCZOS) for image in images]
    palette = images[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    buffer = io.BytesIO()
    frames[0].save(buffer, format='GIF', save_all=True, append_images=frames[1:], loop=0,
                   duration=int(round(1000.0/fps)), disposal=1)
    return buffer.getvalue()
