"""
Renders a fixed set of DiES figures as PNG files, with the DiES modules of a given
folder. tests/compare_figures.py runs it twice (reference commit and current code)
and compares the images pixel by pixel.

Figures (one PNG each, named after the call that produced it):
    single_model__<case>__litho<0|1>__<parameter>   the 13 parameters of single_model,
                                                    for the dike and the 60-degree fault
    gallery__<case>__NN_<label>                     every static figure of
                                                    tests/plot_gallery.py
    dynamic__<case>__step<k>                        plot_dynamic_step for one step in 4
                                                    and the last (0.25 MPa per step)
    dynamic_model__<file name>                      the legacy dynamic_model wrapper
                                                    (plotting=True), 3 steps
    mohr__<k>                                       evaluate_stress_state(..., 'plot')

Usage:

    python tests/figure_cases.py --code-root <folder with DiES_core.py> --out <folder>
"""

import argparse
import os
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))

# single_model parameters (see its docstring)
PARAMETERS = ('ux', 'uy', 'uxy', 'sx', 'sy', 'sxy', 'sav', 't', 's1', 's3', 'st', 'sn', 'ss')
# Dynamic figures on a 250 m grid: the failure criterion is still evaluated node by node
RESOLUTION_DYNAMIC = [250, 250]
# Mohr states (p, s) in Pa, one per failure mode
MOHR_STATES = ((20e6, 10e6), (5e6, -1.5e6), (6e6, -1.4e6), (12e6, -1e6), (40e6, 8e6), (10e6, -3e6))


def save(fig, folder, name):
    """Saves a figure at a fixed dpi and closes it."""
    fig.savefig(os.path.join(folder, name + '.png'), dpi=80)
    plt.close(fig)


def render_all(code_root, out):
    warnings.filterwarnings('ignore')
    # The DiES modules of code_root first; tests/ for visual_check and plot_gallery
    sys.path[:0] = [os.path.abspath(code_root), TESTS]
    from Modeling_functions import single_model, compute_dynamic_model, dynamic_model
    from Failure_Determination import evaluate_stress_state
    import plotting
    import plot_gallery
    from visual_check import CASES, MODEL_INPUTS, G_EARTH, XLIMS, YLIMS, RESOLUTION
    os.makedirs(out, exist_ok=True)

    for case, (title, ci, dx, dy, dz) in CASES.items():
        # 1. single_model, every parameter, with and without lithostatic load
        for litho in (False, True):
            for parameter in PARAMETERS:
                fig = single_model(ci, MODEL_INPUTS, dx, dy, dz, 0, 0, G_EARTH, litho, 60,
                                   XLIMS, YLIMS, RESOLUTION, parameter)
                save(fig, out, 'single_model__' + case + '__litho' + str(int(litho)) + '__' + parameter)
        # 2. Every static figure of the gallery (model.compute + plotting.py)
        for i, (label, fig) in enumerate(plot_gallery.figures(case, True, 0, 0, 60), start=1):
            safe = label.replace(' ', '_').replace(',', '').replace('=', '-').replace('°', 'deg').replace('.', '')
            save(fig, out, 'gallery__' + case + '__%02d_%s' % (i, safe))
        # 3. Dynamic model: one step in 4 and the last
        steps = plot_gallery.investigation_range(case, 0.25)
        dyn = compute_dynamic_model(ci, MODEL_INPUTS, dx, dy, dz, 0, 0, G_EARTH, True, steps,
                                    XLIMS, YLIMS, RESOLUTION_DYNAMIC)
        for k, step in enumerate(dyn, start=1):
            if k % 4 == 0 or k == len(dyn):
                save(plotting.plot_dynamic_step(step, 2), out, 'dynamic__' + case + '__step%02d' % k)
        # 4. The legacy dynamic_model wrapper, which saves its own figures
        with tempfile.TemporaryDirectory() as folder:
            dynamic_model(ci, MODEL_INPUTS, dx, dy, dz, 0, 0, G_EARTH, True, [1/3, 2/3, 1],
                          XLIMS, YLIMS, RESOLUTION_DYNAMIC, 2, True, folder + os.sep)
            for name in sorted(os.listdir(folder)):
                image = plt.imread(os.path.join(folder, name))
                plt.imsave(os.path.join(out, 'dynamic_model__' + case + '__' + name), image)
        plt.close('all')

    # 5. Mohr circles
    cfi, T = MODEL_INPUTS[3], MODEL_INPUTS[4]
    for k, (p, s) in enumerate(MOHR_STATES):
        fig = evaluate_stress_state(p, s, cfi, T, 'plot')
        save(fig if fig is not None else plt.gcf(), out, 'mohr__%d' % k)
    print('Rendered', len(os.listdir(out)), 'figures to', out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--code-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    render_all(args.code_root, args.out)


if __name__ == '__main__':
    main()
