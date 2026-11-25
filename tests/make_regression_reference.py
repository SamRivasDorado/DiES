"""
Builds the regression reference of tests/regression_cases.py from a given commit.

It checks the commit out in a temporary git worktree (the working copy is not
touched), runs the regression cases of the CURRENT tests/regression_cases.py
with the DiES modules of that commit, and saves the outputs. The worktree is
removed at the end.

Usage, from the repository root:

    python tests/make_regression_reference.py                      # commit 0755fe7
    python tests/make_regression_reference.py --commit 164b127 --out other.npz

The reference stored in the repository is
tests/reference_data/regression_0755fe7.npz (end of Phase 1, Nov 2026).
"""

import argparse
import os
import subprocess
import sys
import tempfile

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
DEFAULT_COMMIT = '0755fe7'


def run(command, **kwargs):
    """Runs a command, stopping with its output if it fails."""
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        sys.exit('Failed: ' + ' '.join(command) + '\n' + result.stdout + result.stderr)
    return result.stdout


def build_reference(commit, out):
    with tempfile.TemporaryDirectory() as tmp:
        worktree = os.path.join(tmp, 'ref')
        # A detached worktree of the commit: its DiES modules, untouched by this repository's changes
        run(['git', '-C', ROOT, 'worktree', 'add', '--detach', worktree, commit])
        try:
            # The cases come from this repository's tests/; the modules from the worktree
            print(run([sys.executable, os.path.join(TESTS, 'regression_cases.py'),
                       '--code-root', worktree, '--out', out], cwd=tmp))
        finally:
            run(['git', '-C', ROOT, 'worktree', 'remove', '--force', worktree])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--commit', default=DEFAULT_COMMIT)
    parser.add_argument('--out', default=None, help='default: tests/reference_data/regression_<commit>.npz')
    args = parser.parse_args(argv)
    out = args.out or os.path.join(TESTS, 'reference_data', 'regression_' + args.commit + '.npz')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    build_reference(args.commit, os.path.abspath(out))


if __name__ == '__main__':
    main()
