"""
Pixel-by-pixel comparison of the DiES figures between a reference commit and the
current code (Sam's rule for refactorings: same figures, pixel by pixel).

It renders tests/figure_cases.py twice, with the modules of the reference commit
(checked out in a temporary git worktree) and with the modules of this working copy,
and compares every pair of PNG files. The cases always come from this working
copy's tests/, so both renders draw exactly the same figures.

Usage, from the repository root (takes a few minutes):

    python tests/compare_figures.py                    # against 0755fe7
    python tests/compare_figures.py --ref 164b127 --keep figures_cmp

Exit code 0 if every figure is identical, 1 otherwise. With --keep, the two sets
of PNG files and an image of the differences of each changed figure are kept.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)


def render(code_root, out, cwd):
    """Runs figure_cases.py with the modules of code_root, in a clean process."""
    result = subprocess.run([sys.executable, os.path.join(TESTS, 'figure_cases.py'),
                             '--code-root', code_root, '--out', out],
                            capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        sys.exit('Rendering failed with ' + code_root + ':\n' + result.stdout + result.stderr)
    print(result.stdout.strip())


def compare(ref_dir, new_dir, diff_dir=None):
    """Compares the PNG files of two folders. Returns the list of differences, as text."""
    problems = []
    ref_names, new_names = set(os.listdir(ref_dir)), set(os.listdir(new_dir))
    for name in sorted(ref_names ^ new_names):
        problems.append(name + ': only in ' + ('reference' if name in ref_names else 'current'))
    for name in sorted(ref_names & new_names):
        a = plt.imread(os.path.join(ref_dir, name))
        b = plt.imread(os.path.join(new_dir, name))
        if a.shape != b.shape:
            problems.append(name + ': size ' + str(b.shape) + ' instead of ' + str(a.shape))
        elif not np.array_equal(a, b):
            changed = np.any(a != b, axis=-1)
            problems.append(name + ': ' + str(int(changed.sum())) + ' pixels differ')
            if diff_dir:
                plt.imsave(os.path.join(diff_dir, 'diff__' + name), changed, cmap='gray_r')
    return problems, len(ref_names & new_names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ref', default='0755fe7', help='reference commit')
    parser.add_argument('--keep', default=None, help='folder where the figures are kept')
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        work = args.keep or tmp
        ref_dir, new_dir, diff_dir = (os.path.join(work, n) for n in ('reference', 'current', 'differences'))
        for folder in (ref_dir, new_dir, diff_dir):
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(diff_dir)

        # Reference: a detached worktree of the commit, removed at the end
        worktree = os.path.join(tmp, 'ref_code')
        subprocess.run(['git', '-C', ROOT, 'worktree', 'add', '--detach', worktree, args.ref],
                       check=True, capture_output=True)
        try:
            render(worktree, ref_dir, tmp)
        finally:
            subprocess.run(['git', '-C', ROOT, 'worktree', 'remove', '--force', worktree], capture_output=True)
        render(ROOT, new_dir, tmp)

        problems, n = compare(ref_dir, new_dir, diff_dir)
        if problems:
            print(len(problems), 'of', n, 'figures differ from', args.ref + ':')
            print('\n'.join('  ' + p for p in problems))
        else:
            print('All', n, 'figures are identical, pixel by pixel, to those of', args.ref)
        return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
