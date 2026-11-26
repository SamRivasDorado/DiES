"""
Comparison of the final failure maps of two dynamic models (vertical dike, opening
only; 60-degree fault, pure shear) computed with the sampling criterion used until
Nov 2026 (commit 4adce9e) and with the analytical criterion (current code).
See docs/failure_criterion.md, Section 6.

    python docs/figures/failure_criterion_comparison.py   # ~2 min; writes the PNG next to it

The old code is checked out in a temporary git worktree; each model runs in its own
process with the DiES modules of one version. Prints the nodes that change, the
first-failure steps and the run times.
"""

import collections
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OLD_COMMIT = '4adce9e'

MODEL_INPUTS = [30e9, 0.25, 2900, 0.57, -1.5e6]
# Mars gravity: the comparison of 25 Nov 2026 was made before the tests moved to Earth gravity
G = 3.71
FAULT_LOWER_Y = -1000 - 1000*np.tan(np.radians(60))
# name: (crack_inputs, dx, dy, dz, number of steps of 0.25 MPa)
CASES = {'dike': ([2, 0, -1000, 0, -5000], 0, 0, 0, 32),
         'fault': ([0, -500, -1000, 500, FAULT_LOWER_Y], 0, 15e6, 0, 60)}
TIPS = {'dike': ((0, -1000), (0, -5000)), 'fault': ((-500, -1000), (500, FAULT_LOWER_Y))}


def compute(code_root, out):
    """Runs the dynamic models with the DiES modules of code_root and saves the maps (child process)."""
    import time
    import warnings
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.abspath(code_root))
    from Modeling_functions import compute_dynamic_model
    data = {}
    for case, (ci, dx, dy, dz, n) in CASES.items():
        t = time.perf_counter()
        dyn = compute_dynamic_model(ci, MODEL_INPUTS, dx, dy, dz, 0, 0, G, True, [i/n for i in range(1, n + 1)],
                                    [-5000, 5000], [-6000, -10], [250, 250])
        data[case + '__seconds'] = np.asarray(time.perf_counter() - t)
        data[case + '__final_mode'] = np.asarray(dyn.final_mode()).astype(str)
        data[case + '__failure_step'] = dyn.failure_step()
        last = dyn[-1]
        # failure_jumped exists only with the analytical criterion
        data[case + '__jumped'] = last['failure_jumped'] if 'failure_jumped' in last else np.zeros(last.shape)
        data[case + '__x'], data[case + '__y'] = last.x, last.y
    np.savez(out, **data)


def run_both(tmp):
    """Computes the maps with the old commit (in a worktree) and with the current code."""
    worktree = os.path.join(tmp, 'old_code')
    subprocess.run(['git', '-C', ROOT, 'worktree', 'add', '--detach', worktree, OLD_COMMIT],
                   check=True, capture_output=True)
    try:
        for code_root, name in ((worktree, 'old.npz'), (ROOT, 'new.npz')):
            subprocess.run([sys.executable, os.path.abspath(__file__), '--compute', code_root,
                            os.path.join(tmp, name)], check=True, cwd=tmp)
    finally:
        subprocess.run(['git', '-C', ROOT, 'worktree', 'remove', '--force', worktree], capture_output=True)
    return np.load(os.path.join(tmp, 'old.npz')), np.load(os.path.join(tmp, 'new.npz'))


def draw(old, new):
    """Maps of the final failure mode, old and new, and of the nodes that change."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    modes = ['Below', 'Above', 'Mode I', 'Mixed-mode I-II', 'Mode II - tension-compression',
             'Mode II - compression', 'Singular']
    colors = ['#f2f2f2', '#d62728', 'blue', 'dodgerblue', 'purple', '#ff7f0e', 'black']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(modes) + 1) - 0.5, len(modes))
    code = np.vectorize(modes.index)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharex=True, sharey=True)
    for row, case in enumerate(CASES):
        x, y = new[case + '__x'], new[case + '__y']
        mo, mn = old[case + '__final_mode'], new[case + '__final_mode']
        jumped = new[case + '__jumped'] > 0
        changed = mo != mn
        panels = ((mo, 'sampling criterion (until Nov 2026)'), (mn, 'analytical criterion (Phase 2, 3B)'))
        for col, (m, title) in enumerate(panels):
            axes[row, col].pcolormesh(x, y, code(m), cmap=cmap, norm=norm, shading='auto')
            axes[row, col].set_title(case + ': ' + title, fontsize=10)
        if jumped.any():
            axes[row, 1].scatter(x[jumped], y[jumped], s=10, facecolor='none', edgecolor='k', lw=0.6,
                                 label='jumped the band (contact recorded)')
            axes[row, 1].legend(loc='lower left', fontsize=8)
        axes[row, 2].pcolormesh(x, y, np.where(changed, code(mn), np.nan), cmap=cmap, norm=norm, shading='auto')
        axes[row, 2].set_title(case + ': nodes that change (colour = new mode)\n%d of %d nodes'
                               % (changed.sum(), changed.size), fontsize=10)
        for ax in axes[row]:
            (x0, y0), (x1, y1) = TIPS[case]
            ax.plot((x0, x1), (y0, y1), c='k', lw=2.5)
            ax.set_aspect('equal')

        # Text summary: transitions, first-failure steps and run times
        print(case + ': %d of %d nodes change, %d jumped the band' % (changed.sum(), changed.size, jumped.sum()))
        for (a, b), n in collections.Counter(zip(mo[changed], mn[changed])).most_common():
            print('   %4d  %-30s -> %s' % (n, a, b))
        fo, fn = old[case + '__failure_step'], new[case + '__failure_step']
        both = (fo > 0) & (fn > 0)
        diff = np.abs(fo[both] - fn[both])
        print('   first-failure step (nodes failed with both): same %d, 1 step apart %d, more %d'
              % ((diff == 0).sum(), (diff == 1).sum(), (diff > 1).sum()))
        print('   run time: %.1f s -> %.1f s' % (old[case + '__seconds'], new[case + '__seconds']))

    for ax in axes[:, 0]:
        ax.set_ylabel('Y (m)')
    for ax in axes[1]:
        ax.set_xlabel('X (m)')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='0.5') for c in colors]
    fig.legend(handles, modes, loc='lower center', ncol=7, fontsize=9, frameon=False)
    fig.suptitle('Final failure mode of the dynamic models (0.25 MPa per step, 250 m grid, litho on): '
                 'sampling vs analytical criterion', fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(os.path.join(HERE, 'failure_criterion_comparison.png'), dpi=110)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--compute':
        compute(sys.argv[2], sys.argv[3])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            draw(*run_both(tmp))
