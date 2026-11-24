"""
Gallery of every figure DiES can produce, for the two reference cases of
tests/visual_check.py (vertical dike, opening only; 60-degree pure-shear fault).

For each case it saves one PNG per figure in <out>/<case>/ and puts all of them,
in order, in <out>/DiES_gallery.pdf:

    plot_displacement        ux, uy, uxy  x  vectors, contours
    plot_stresses            sxa, sya, sav, sxya, sxza, sza, ta
    plot_principal_stresses  principal (s1), secondary (s3), angle
    plot_stress_trajectories without and with the optimal shear planes
    plot_plane_stresses      normal and shear stress on planes at --angle
    crack.plot_mode_I_crack_displacements
    dynamic_model            one figure per step (plotting=True)

Usage, from the repository root:

    python tests/plot_gallery.py --out gallery
"""

import argparse
import glob
import os
import shutil
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from DiES_core import model, crack
from Helper_functions import G, points_distance
from Modeling_functions import dynamic_model
from visual_check import CASES, MODEL_INPUTS, G_MARS, XLIMS, YLIMS, RESOLUTION

warnings.filterwarnings('ignore')


def build(name):
    title, ci, dx, dy, dz = CASES[name]
    E, nu, d, cfi, T = MODEL_INPUTS
    a = ci[0] ; up = (ci[1], ci[2]) ; lo = (ci[3], ci[4])
    # Same as single_model: an opening crack gets dx from its aperture
    px = dx if a == 0 else a*G(E, nu)/(points_distance(up, lo)*(1 - nu))
    disc = crack(up, lo, px, dy, dz)
    stress_model = model(XLIMS, YLIMS, RESOLUTION[0], RESOLUTION[1], 500, E, nu, d, cfi, T, G_MARS)
    # single_model uses coarser grids for the vector plots and the trajectories
    vector_model = model(XLIMS, YLIMS, RESOLUTION[0]*2.5, RESOLUTION[1]*3, 500, E, nu, d, cfi, T, G_MARS)
    return title, disc, stress_model, vector_model


def figures(name, litho, rx, ry, angle):
    """Yields (label, figure) for every static figure of one case."""
    title, disc, m, mv = build(name)
    for comp in ('ux', 'uy', 'uxy'):
        yield 'plot_displacement ' + comp + ', vectors', mv.plot_displacement(disc, comp, 'vectors', 0.25)
        yield 'plot_displacement ' + comp + ', contours', m.plot_displacement(disc, comp, 'contours', 0.1)
    for stress, step in (('sxa', 2.5), ('sya', 5), ('sav', 5), ('sxya', 1), ('sxza', 1), ('sza', 5), ('ta', 1)):
        yield 'plot_stresses ' + stress, m.plot_stresses(disc, stress, rx, ry, litho, step)
    for stress, step in (('principal', 5), ('secondary', 2.5), ('angle', 10)):
        yield 'plot_principal_stresses ' + stress, m.plot_principal_stresses(disc, stress, rx, ry, litho, step)
    for planes in (False, True):
        yield 'plot_stress_trajectories, planes=' + str(planes), mv.plot_stress_trajectories(disc, planes, rx, ry, litho)
    for stress in ('normal', 'shear'):
        yield 'plot_plane_stresses ' + stress + ', planes at ' + str(angle) + '°', \
              m.plot_plane_stresses(disc, stress, rx, ry, angle, litho, 0.5)
    yield 'crack.plot_mode_I_crack_displacements', disc.plot_mode_I_crack_displacements(MODEL_INPUTS[1], MODEL_INPUTS[0])


def dynamic_figures(name, litho, rx, ry, folder, steps):
    title, ci, dx, dy, dz = CASES[name]
    os.makedirs(folder, exist_ok=True)
    for old in glob.glob(os.path.join(folder, '*.png')):
        os.remove(old)
    dynamic_model(ci, MODEL_INPUTS, dx, dy, dz, rx, ry, G_MARS, litho, steps,
                  XLIMS, YLIMS, RESOLUTION, 2, True, folder)
    plt.close('all')
    # Files are named ..._<step>_<dx>_MPa.png; sort by step
    paths = glob.glob(os.path.join(folder, '*.png'))
    return sorted(paths, key=lambda p: float(os.path.basename(p).split('_')[-3]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='gallery')
    parser.add_argument('--angle', type=float, default=60.0, help='plane angle for plot_plane_stresses, degrees')
    parser.add_argument('--no-litho', action='store_true')
    parser.add_argument('--rx', type=float, default=0.0)
    parser.add_argument('--ry', type=float, default=0.0)
    parser.add_argument('--steps', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0],
                        help='investigation_range of dynamic_model')
    args = parser.parse_args(argv)
    litho = not args.no_litho
    os.makedirs(args.out, exist_ok=True)
    pdf_path = os.path.join(args.out, 'DiES_gallery.pdf')

    with PdfPages(pdf_path) as pdf:
        for name in CASES:
            title = CASES[name][0]
            case_dir = os.path.join(args.out, name)
            if os.path.isdir(case_dir):
                shutil.rmtree(case_dir)
            os.makedirs(case_dir)

            cover = plt.figure(figsize=(11, 3))
            cover.text(0.5, 0.6, title, ha='center', fontsize=20, weight='bold')
            cover.text(0.5, 0.3, 'litho=' + str(litho) + ', rx=' + str(args.rx/1e6) + ' MPa, ry='
                       + str(args.ry/1e6) + ' MPa, g = ' + str(G_MARS) + ' m/s², E = '
                       + str(MODEL_INPUTS[0]/1e9) + ' GPa, ν = ' + str(MODEL_INPUTS[1])
                       + ', cfi = ' + str(MODEL_INPUTS[3]) + ', T = ' + str(MODEL_INPUTS[4]/1e6) + ' MPa',
                       ha='center', fontsize=11)
            pdf.savefig(cover) ; plt.close(cover)

            for i, (label, fig) in enumerate(figures(name, litho, args.rx, args.ry, args.angle), start=1):
                fig.suptitle(title + '\n' + label, fontsize=10)
                safe = label.replace(' ', '_').replace(',', '').replace('=', '-').replace('°', 'deg').replace('.', '')
                fig.savefig(os.path.join(case_dir, '%02d_%s.png' % (i, safe)), dpi=110, bbox_inches='tight')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                print(name, '|', label)

            for j, path in enumerate(dynamic_figures(name, litho, args.rx, args.ry,
                                                     os.path.join(case_dir, 'dynamic'), args.steps), start=1):
                image = plt.imread(path)
                page = plt.figure(figsize=(9, 9*image.shape[0]/image.shape[1] + 0.6))
                ax = page.add_axes([0, 0, 1, 1 - 0.6/page.get_figheight()])
                ax.imshow(image) ; ax.set_axis_off()
                page.suptitle(title + '\ndynamic_model, step ' + str(j) + ' of ' + str(len(args.steps))
                              + ' (' + os.path.basename(path) + ')', fontsize=10)
                pdf.savefig(page) ; plt.close(page)
                print(name, '| dynamic_model step', j)

    print('Saved', pdf_path)
    return pdf_path


if __name__ == '__main__':
    main()
