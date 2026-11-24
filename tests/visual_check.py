"""
Visual check of DiES after every change to the core.

Runs two fixed reference cases through single_model and saves, for each one,
a stress field and a displacement vector field (quiver), plus a 2x2 panel
with the four figures:

    1. Vertical dike: opening-mode crack only (aperture > 0, dy = dz = 0)
    2. Non-vertical fault, pure shear (aperture 0, dx = 0, dy != 0), dip 60 deg

Usage, from the repository root:

    python tests/visual_check.py                 # saves to visual_check/
    python tests/visual_check.py --out my_folder --stress s1

The stress parameter can be any of those accepted by single_model
(sx, sy, sxy, sav, t, s1, s3); the displacement is always 'uxy' (vectors).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Modeling_functions import single_model

# Material: E (Pa), v, density (kg m-3), cfi, T (Pa). Cohesion is implicit, C = 2|T|
# Values of the example model in Rivas-Dorado et al. (2023, JSG 167, 104778). With v = 0.25
# and cfi = 0.57 the lithostatic state alone does not reach failure (with cfi >= 0.6 it does).
# T is negative: tension is negative on the Mohr diagram.
MODEL_INPUTS = [30e9, 0.25, 2900, 0.57, -1.5e6]
G_MARS = 3.71
XLIMS = [-5000, 5000]
YLIMS = [-6000, -10]
RESOLUTION = [100, 100]

# name: (title, crack_inputs, dx, dy, dz)
# crack_inputs: [total aperture, x upper tip, y upper tip, x lower tip, y lower tip]
CASES = {
    'dike': ('Vertical dike, opening only (a = 2 m)',
             [2, 0, -1000, 0, -5000], 0, 0, 0),
    # sigma_II raised from 5 to 15 MPa on 24 Nov 2026: with 5 MPa the fault barely
    # generated discontinuities in dynamic_model
    'fault': ('Fault dipping 60°, pure shear (σII = 15 MPa)',
              [0, -500, -1000, 500, -1000 - 1000*np.tan(np.radians(60))], 0, 15e6, 0),
}


def run_case(name, stress, litho, rx, ry):
    title, crack_inputs, dx, dy, dz = CASES[name]
    figures = {}
    for parameter in (stress, 'uxy'):
        fig = single_model(crack_inputs, MODEL_INPUTS, dx, dy, dz, rx, ry, G_MARS, litho,
                           60, XLIMS, YLIMS, RESOLUTION, parameter)
        fig.suptitle(title + ' — ' + parameter, fontsize=11)
        figures[parameter] = fig
    return figures


def figure_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='visual_check')
    parser.add_argument('--stress', default='sxy')
    parser.add_argument('--no-litho', action='store_true', help='run with litho=False')
    parser.add_argument('--rx', type=float, default=0.0, help='remote stress along x, Pa')
    parser.add_argument('--ry', type=float, default=0.0, help='remote stress along y, Pa')
    parser.add_argument('--label', default='', help='text added to the panel title, e.g. a commit')
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    litho = not args.no_litho

    images = []
    for name in CASES:
        figures = run_case(name, args.stress, litho, args.rx, args.ry)
        for parameter, fig in figures.items():
            path = os.path.join(args.out, name + '_' + parameter + '.png')
            fig.savefig(path, dpi=110)
            images.append((name + ' · ' + parameter, figure_to_array(fig)))
            plt.close(fig)

    panel, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (label, image) in zip(axes.ravel(), images):
        ax.imshow(image)
        ax.set_axis_off()
    title = 'DiES visual check — litho=' + str(litho) + ', rx=' + str(args.rx/1e6) + ' MPa, ry=' + str(args.ry/1e6) + ' MPa'
    if args.label:
        title = title + ' — ' + args.label
    panel.suptitle(title, fontsize=13)
    panel.tight_layout()
    panel_path = os.path.join(args.out, 'visual_check.png')
    panel.savefig(panel_path, dpi=100)
    plt.close(panel)
    print('Saved', panel_path)
    return panel_path


if __name__ == '__main__':
    main()
