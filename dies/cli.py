"""
Command line of DiES (Phase 3, Nov 2026): runs a scenario file without writing Python.

    dies run examples/dike_example.yaml --out results/
    dies check examples/dike_example.yaml
    dies --version

(also `python -m dies ...`, which works without installing the command).

- `run` reads and checks the scenario, computes it and writes what its output section
  asks for (ScenarioResults.write: results, figures and a copy of the scenario).
- `check` only reads and checks it, and prints a summary of the model it describes
  (grid, layers, crack, driving stresses, steps, outputs), without computing it.

An invalid scenario prints every problem, one per line, without a Python traceback,
and exits with code 1. A folder that is not empty is not overwritten unless
--overwrite is given.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np


def _build_parser():
    """The command-line arguments: one subcommand per action."""
    import dies
    parser = argparse.ArgumentParser(
        prog='dies',
        description='DiES (Discontinuities in Elastic Space): run a model described in a scenario file '
                    '(YAML or JSON; format in docs/scenario.md).')
    version = dies.__version__ if dies.__version__ != 'unknown' else 'unknown (not installed; run from the repository)'
    parser.add_argument('--version', action='version', version='DiES ' + version)
    commands = parser.add_subparsers(dest='command', metavar='command')
    commands.required = True

    run = commands.add_parser('run', help='compute a scenario and write its results and figures',
                              description='Compute a scenario and write what its output section asks for.')
    run.add_argument('scenario', help='scenario file (.yaml, .yml or .json)')
    run.add_argument('--out', '-o', help='output folder (default: <scenario file name>_results, in the '
                                         'current folder)')
    run.add_argument('--overwrite', action='store_true',
                     help='write into the output folder even if it is not empty')
    run.add_argument('--quiet', '-q', action='store_true', help='print only problems')

    check = commands.add_parser('check', help='check a scenario and summarise it, without computing it',
                                description='Check a scenario and print a summary of the model it describes.')
    check.add_argument('scenario', help='scenario file (.yaml, .yml or .json)')
    return parser


def _print_warning(message, category, filename, lineno, file=None, line=None):
    # Warnings of the core (e.g. a crack crossing a layer boundary) as one plain line
    print('Warning: ' + str(message), file=sys.stderr)


def _with_unit(value, unit, factor=1.0):
    # A value of the core units shown in a readable unit, without needless decimals
    return '%g %s' % (value/factor, unit)


def summary(scenario):
    """The summary printed by `dies check`: a list of lines describing the scenario."""
    d = scenario.data
    g = d['grid']
    # Number of nodes, as model.grid() builds them (np.arange of the limits by the cell size)
    nx = len(np.arange(g['x'][0], g['x'][1], g['cell'][0]))
    ny = len(np.arange(g['y'][0], g['y'][1], g['cell'][1]))
    lines = ['Scenario: ' + d['name']]
    if d['description']:
        lines.append('  ' + d['description'])
    lines.append('Grid: x %s to %s, y %s to %s, cells %s x %s (%d x %d = %d nodes)' % (
        _with_unit(g['x'][0], 'km', 1e3), _with_unit(g['x'][1], 'km', 1e3),
        _with_unit(g['y'][0], 'km', 1e3), _with_unit(g['y'][1], 'km', 1e3),
        _with_unit(g['cell'][0], 'm'), _with_unit(g['cell'][1], 'm'), nx, ny, nx*ny))
    lines.append('Gravity: ' + _with_unit(d['gravity'], 'm/s2'))
    lines.append('Layers (from the top down):')
    for layer in d['layers']:
        lines.append('  %s: top %s, E %s, nu %g, density %s, friction %g, T %s' % (
            layer['name'], _with_unit(layer['top'], 'm'), _with_unit(layer['E'], 'GPa', 1e9), layer['nu'],
            _with_unit(layer['density'], 'kg/m3'), layer['friction'], _with_unit(layer['T'], 'MPa', 1e6)))
    c = d['cracks'][0]
    lines.append('Crack: %s, from (%s, %s) to (%s, %s)' % (
        c['name'], _with_unit(c['upper_tip'][0], 'm'), _with_unit(c['upper_tip'][1], 'm'),
        _with_unit(c['lower_tip'][0], 'm'), _with_unit(c['lower_tip'][1], 'm')))
    # Final driving stresses; with an aperture, sigma_I follows from it (as in the core)
    sI, sII, sIII = scenario.driving_stresses()
    origin = ' (from the aperture of %s)' % _with_unit(c['aperture'], 'm') if c['aperture'] is not None else ''
    lines.append('  driving stresses: sigma_I %s%s, sigma_II %s, sigma_III %s' % (
        _with_unit(sI, 'MPa', 1e6), origin, _with_unit(sII, 'MPa', 1e6), _with_unit(sIII, 'MPa', 1e6)))
    r = d['remote_stresses']
    lines.append('Loading: remote xx %s, yy %s; lithostatic %s' % (
        _with_unit(r['xx'], 'MPa', 1e6), _with_unit(r['yy'], 'MPa', 1e6), 'yes' if d['lithostatic'] else 'no'))
    a = d['analysis']
    if a['type'] == 'dynamic':
        fractions = scenario.investigation_range()
        largest = max(abs(s) for s in (sI, sII, sIII))
        lines.append('Analysis: dynamic, %d steps of %s (largest driving stress)' % (
            len(fractions), _with_unit(largest/len(fractions), 'MPa', 1e6)))
    else:
        lines.append('Analysis: static')
    out = d['output']
    figures = list(out['figures']['static'])
    dynamic = out['figures']['dynamic']
    lines.append('Output: results %s; static figures %s; dynamic figures %s; %s at %d dpi' % (
        ', '.join(out['results']) or 'none', ', '.join(figures) or 'none',
        ('one step in %d and the last one' % dynamic['show_every']) if dynamic else 'none',
        out['format'], out['dpi']))
    return lines


def _default_folder(path):
    # <scenario file name>_results in the current folder
    return os.path.splitext(os.path.basename(path))[0] + '_results'


def main(argv=None):
    """Entry point of the `dies` command. Returns the exit code (0 if all went well)."""
    import matplotlib
    import dies

    args = _build_parser().parse_args(argv)
    warnings.showwarning = _print_warning

    try:
        scenario = dies.load_scenario(args.scenario)
    except dies.ScenarioError as error:
        # Every problem of the scenario, without a traceback
        print(str(error), file=sys.stderr)
        return 1

    if args.command == 'check':
        print('\n'.join(summary(scenario)))
        print('The scenario is valid.')
        return 0

    # run: refuse to mix the files of this run with those already in a folder
    folder = args.out or _default_folder(args.scenario)
    if os.path.isdir(folder) and os.listdir(folder) and not args.overwrite:
        print("The output folder '" + folder + "' is not empty; use --overwrite to write into it, "
              "or choose another one with --out.", file=sys.stderr)
        return 1

    # Figures are only saved, never shown: no window opens
    matplotlib.use('Agg', force=True)
    start = time.time()
    if not args.quiet:
        print("Running '" + scenario.name + "' (" + scenario.analysis + ')...')
    try:
        run = scenario.run()
        written = run.write(folder)
    except dies.InputError as error:
        print('Error: ' + str(error), file=sys.stderr)
        return 1
    if not args.quiet:
        if run.dynamic is not None:
            print('  %d dynamic steps' % len(run.dynamic))
        print('  %d files written to %s (%.1f s)' % (len(written), os.path.abspath(folder), time.time() - start))
    return 0


if __name__ == '__main__':
    sys.exit(main())
