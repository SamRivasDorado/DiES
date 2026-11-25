"""
Check of the failure criterion on the reference cases of tests/visual_check.py
(vertical dike, opening only; 60-degree pure-shear fault, sigma_II = 15 MPa), with
the dynamic model at 0.25 MPa per step (Sam's convention), Earth gravity, litho on.

For every node, at the step where it first fails, it recomputes the stresses of
that step and checks that:
  1. r/r_c is within the failure band (or, for the nodes marked failure_jumped, that
     the node was below the band at the previous step and beyond it at this one);
  2. the mode is the one of the contact of its Mohr circle with the envelope;
  3. the stresses resolved on the recorded failure plane (from the stress tensor)
     point in the recorded direction 2theta of the Mohr diagram and, for nodes inside
     the band, are exactly the recorded sigma_n and tau;
  4. the sign of sigma_n agrees with the mode (Mode I at sigma_n ~ T; mixed mode
     sigma_n < 0; Mode II sigma_n >= 0) and 2theta is 90 + phi on the Coulomb line;
  5. the shear planes dip towards the crack.

Figures, in <out>:
  <case>_maps.png   final mode, first-failure step, jumped nodes and r/r_c at the last step
  <case>_mohr.png   Mohr circles of one node per failure mode (and one jumped node), at
                    the step before failure and at the failure step, with the band

Usage, from the repository root:

    python tests/failure_check.py --out failure_check
"""

import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
for folder in (ROOT, TESTS):
    if folder not in sys.path:
        sys.path.insert(0, folder)
warnings.filterwarnings('ignore')

from DiES_core import model, crack
from Failure_Determination import FAILURE_TOLERANCE, failure_contact
from Modeling_functions import compute_dynamic_model
from plot_gallery import investigation_range
from visual_check import CASES, MODEL_INPUTS, G_EARTH, XLIMS, YLIMS, RESOLUTION

MODES = ['Below', 'Above', 'Mode I', 'Mixed-mode I-II', 'Mode II - tension-compression',
         'Mode II - compression', 'Singular']
COLORS = ['#f2f2f2', '#d62728', 'blue', 'dodgerblue', 'purple', '#ff7f0e', 'black']
FAILED = MODES[2:6]
PLANES = ('plane_mode_I', 'plane_mixed_left', 'plane_tc_left', 'plane_comp_left',
          'plane_mixed_right', 'plane_tc_right', 'plane_comp_right')
STEP_MPA = 0.25


def step_stresses(m, ci, step, litho=True):
    """Stresses of a step of the dynamic model, recomputed from its driving stresses."""
    meta = step.metadata
    disc = crack((ci[1], ci[2]), (ci[3], ci[4]), meta['dx'], meta['dy'], meta['dz'])
    return m.compute(disc, 0, 0, litho), disc


def run_case(name):
    """Runs the dynamic model and checks every failed node. Returns the results and the checks."""
    title, ci, dx, dy, dz = CASES[name]
    steps = investigation_range(name, STEP_MPA)
    dyn = compute_dynamic_model(ci, MODEL_INPUTS, dx, dy, dz, 0, 0, G_EARTH, True, steps, XLIMS, YLIMS, RESOLUTION)
    E, nu, rho, cfi, T = MODEL_INPUTS
    m = model(XLIMS, YLIMS, RESOLUTION[0], RESOLUTION[1], 500, E, nu, rho, cfi, T, G_EARTH)
    first = dyn.failure_step()
    last = dyn[-1]
    phi = np.degrees(np.arctan(cfi))
    checks = {key: [0, 0] for key in ('band', 'mode', 'direction', 'values', 'sign_and_angle', 'dip')}
    worst = {'values': 0.0, 'direction': 0.0}
    examples = {}

    def record(key, ok):
        checks[key][0] += 1
        checks[key][1] += int(not ok)

    previous = None
    for k, step in enumerate(dyn, start=1):
        res, disc = step_stresses(m, ci, step)
        nodes = np.argwhere((first == k) & np.isin(last['failure_mode'], FAILED))
        if len(nodes):
            ratio, csn, css, c2t, cmode = failure_contact(res.s1, res.s3, cfi, T)
            prev_ratio = None
            if previous is not None:
                prev_ratio = failure_contact(previous.s1, previous.s3, cfi, T)[0]
        for i, j in nodes:
            mode = last['failure_mode'][i, j]
            jumped = last['failure_jumped'][i, j] == 1
            s1, s3 = res.s1[i, j], res.s3[i, j]
            q = ratio[i, j]
            # 1. Band (or the jump over it)
            if jumped:
                below_before = prev_ratio is None or prev_ratio[i, j] < 1 - FAILURE_TOLERANCE
                record('band', (q > 1 + FAILURE_TOLERANCE or s3 < T) and below_before)
            else:
                record('band', abs(q - 1) <= FAILURE_TOLERANCE and s3 >= T)
            # 2. Mode of the contact
            record('mode', mode == cmode[i, j])
            # 3. Stresses resolved on the recorded plane, from the tensor (compression positive)
            b = [last[p][i, j] for p in PLANES if np.isfinite(last[p][i, j])][0]
            tensor = -np.array([[res.sxx[i, j], res.sxy[i, j]], [res.sxy[i, j], res.syy[i, j]]])/1e6
            normal = np.array([-np.sin(b), np.cos(b)])
            traction = tensor @ normal
            sn_plane = normal @ traction
            ss_plane = np.linalg.norm(traction - sn_plane*normal)
            c = (s1 + s3)/2e6
            direction = np.degrees(np.arctan2(ss_plane, sn_plane - c))
            two_theta = 2*last['failure_theta'][i, j]
            record('direction', abs(direction - two_theta) < 1e-6)
            worst['direction'] = max(worst['direction'], abs(direction - two_theta))
            if not jumped:
                diff = max(abs(sn_plane - last['failure_sn'][i, j]), abs(ss_plane - last['failure_ss'][i, j]))
                record('values', diff < 1e-6)
                worst['values'] = max(worst['values'], diff)
            # 4. Sign of sigma_n and 2theta by mode
            sn = last['failure_sn'][i, j]
            if mode == 'Mode I':
                ok = abs(sn - T/1e6) <= FAILURE_TOLERANCE*(c - T/1e6) + 1e-9 and np.isclose(two_theta, 180)
            elif mode == 'Mixed-mode I-II':
                ok = sn < 0 and 90 + phi < two_theta < 180
            else:
                ok = sn >= -1e-9 and np.isclose(two_theta, 90 + phi)
            record('sign_and_angle', ok)
            # 5. Shear planes dip towards the crack: the steeper towards it of the two conjugates
            if mode != 'Mode I':
                side = -1 if res.x[i, j] <= disc.c()[0] else 1
                other = np.deg2rad(2*res.s1a[i, j]) - b
                record('dip', np.sin(2*b)*side >= np.sin(2*other)*side - 1e-12)
            # One example per mode (the node with the median failure step), and one jumped node
            key = 'jumped' if jumped else mode
            examples.setdefault(key, []).append((k, i, j))
        previous = res
    return dyn, m, checks, worst, examples


def draw_maps(name, dyn, out):
    """Final mode, first-failure step, jumped nodes and r/r_c at the last step."""
    title, ci, *_ = CASES[name]
    last = dyn[-1]
    x, y = last.x, last.y
    code = np.vectorize(MODES.index)(last['failure_mode'])
    first = dyn.failure_step().astype(float)
    E, nu, rho, cfi, T = MODEL_INPUTS
    m = model(XLIMS, YLIMS, RESOLUTION[0], RESOLUTION[1], 500, E, nu, rho, cfi, T, G_EARTH)
    res, _ = step_stresses(m, ci, last)
    ratio = failure_contact(res.s1, res.s3, cfi, T)[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    ax = axes[0, 0]
    ax.pcolormesh(x, y, code, cmap=ListedColormap(COLORS), norm=BoundaryNorm(np.arange(8) - 0.5, 7), shading='auto')
    ax.set_title('Final failure mode', fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='0.5') for c in COLORS]
    ax.legend(handles, MODES, fontsize=7, loc='lower left', framealpha=0.9)
    ax = axes[0, 1]
    pc = ax.pcolormesh(x, y, np.where(first > 0, first*STEP_MPA, np.nan), cmap='viridis', shading='auto')
    fig.colorbar(pc, ax=ax, label='driving stress at first failure (MPa)')
    ax.set_title('Driving stress at which each node first fails', fontsize=10)
    ax = axes[1, 0]
    jumped = last['failure_jumped'] == 1
    ax.pcolormesh(x, y, np.where(np.isin(last['failure_mode'], FAILED), 1, np.nan), cmap=ListedColormap(['0.8']),
                  shading='auto')
    ax.scatter(x[jumped], y[jumped], s=6, c='k')
    ax.set_title('Failed nodes (grey); jumped the band (black): %d of %d failed'
                 % (jumped.sum(), np.isin(last['failure_mode'], FAILED).sum()), fontsize=10)
    ax = axes[1, 1]
    pc = ax.pcolormesh(x, y, np.clip(ratio, 0, 2), cmap='RdBu_r', vmin=0, vmax=2, shading='auto')
    ax.contour(x, y, ratio, levels=[1 - FAILURE_TOLERANCE, 1 + FAILURE_TOLERANCE], colors='k', linewidths=0.8)
    fig.colorbar(pc, ax=ax, label='r / r$_c$ (clipped at 2)')
    ax.set_title('r/r$_c$ at the last step; black lines: edges of the failure band', fontsize=10)
    up, low = (ci[1], ci[2]), (ci[3], ci[4])
    for ax in axes.flat:
        ax.plot((up[0], low[0]), (up[1], low[1]), c='k', lw=2)
        ax.set_aspect('equal')
    for ax in axes[:, 0]:
        ax.set_ylabel('Y (m)')
    for ax in axes[1]:
        ax.set_xlabel('X (m)')
    fig.suptitle(title + ' — dynamic model, %.2f MPa per step, %d steps, g = %.2f, litho on'
                 % (STEP_MPA, len(dyn), G_EARTH), fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + '_maps.png'), dpi=110)
    plt.close(fig)


def draw_mohr(name, dyn, m, examples, out):
    """Mohr circles of one node per mode (median failure step), before and at failure."""
    title, ci, *_ = CASES[name]
    E, nu, rho, cfi, T = MODEL_INPUTS
    k_t = -T/1e6
    keys = [key for key in FAILED + ['jumped'] if key in examples]
    ncols = 3
    nrows = (len(keys) + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.4*nrows + 0.6), squeeze=False)
    for ax in axes.flat[len(keys):]:
        ax.axis('off')
    for ax, key in zip(axes.flat, keys):
        k, i, j = sorted(examples[key])[len(examples[key])//2]
        last = dyn[-1]
        for kk, style in ((k - 1, ':'), (k, '-')):
            if kk < 1:
                continue
            res, _ = step_stresses(m, ci, dyn[kk - 1])
            p, s = res.s1[i, j]/1e6, res.s3[i, j]/1e6
            c, r = (p + s)/2, (p - s)/2
            t = np.linspace(0, np.pi, 200)
            ax.plot(c + r*np.cos(t), r*np.sin(t), style, c='k', lw=1.4,
                    label='step %d (%.2f MPa)' % (kk, kk*STEP_MPA))
        ratio, csn, css, c2t, cmode = (np.asarray(v).item() for v in failure_contact(p*1e6, s*1e6, cfi, T))
        r_c = r/ratio
        for f in (1 - FAILURE_TOLERANCE, 1 + FAILURE_TOLERANCE):
            ax.plot(c + f*r_c*np.cos(t), f*r_c*np.sin(t), c='tab:green', lw=0.7, alpha=0.7)
        ax.plot([], [], c='tab:green', lw=0.7, label='band: (1 ± %.3f)·r$_c$' % FAILURE_TOLERANCE)
        # Envelope
        sg = np.linspace(-k_t, 0, 100)
        sl = np.linspace(0, c + 2*r, 100)
        ax.plot(sg, np.sqrt(4*k_t*(sg + k_t)), c='darkred', lw=1.8)
        ax.plot(sl, 2*k_t + cfi*sl, c='tomato', lw=1.8)
        ax.plot(csn, css, '*', ms=13, mfc='orange', mec='k', label='contact (σn=%.2f, τ=%.2f)' % (csn, css))
        ax.plot(last['failure_sn'][i, j], last['failure_ss'][i, j], 'o', ms=6, mfc='none', mec='b',
                label='recorded (2θ=%.1f°)' % (2*last['failure_theta'][i, j]))
        if key == 'jumped':
            # Why it is beyond the band: r/r_c > 1 + tolerance, or sigma3 < T
            why = 'σ$_3$ = %.2f < T' % s if s < T/1e6 else 'r/r$_c$ > 1 + %.3f' % FAILURE_TOLERANCE
            label = 'jumped the band (' + why + ') → ' + last['failure_mode'][i, j]
        else:
            label = key
        ax.set_title('%s\nnode (%.0f, %.0f) m, failure at step %d, r/r$_c$ = %.3f'
                     % (label, last.x[i, j], last.y[i, j], k, ratio), fontsize=10)
        ax.set_xlim(min(-k_t*1.3, c - 1.15*r), c + 1.15*r)
        ax.set_ylim(0, 1.15*r)
        ax.set_aspect('equal')
        ax.axvline(0, c='0.7', lw=0.6)
        ax.set_xlabel('σ$_n$ (MPa)')
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
        ax.set_ylabel('τ (MPa)')
    fig.suptitle(title + ' — Mohr circles at the step before failure (dotted) and at failure (solid)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + '_mohr.png'), dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='failure_check')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    all_ok = True
    for name in CASES:
        dyn, m, checks, worst, examples = run_case(name)
        draw_maps(name, dyn, args.out)
        draw_mohr(name, dyn, m, examples, args.out)
        modes = dyn.final_mode()
        print('\n' + CASES[name][0] + ': %d steps' % len(dyn))
        print('  final modes: ' + ', '.join('%s %d' % (mo, (modes == mo).sum()) for mo in MODES if (modes == mo).any()))
        print('  jumped the band: %d nodes' % (dyn[-1]['failure_jumped'] == 1).sum())
        for key, (n, bad) in checks.items():
            all_ok &= bad == 0
            print('  check %-15s %5d nodes, %d failures' % (key, n, bad))
        print('  worst |recorded - resolved| (MPa): %.2e ; worst direction error (deg): %.2e'
              % (worst['values'], worst['direction']))
    print('\nAll checks passed' if all_ok else '\nSOME CHECKS FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
