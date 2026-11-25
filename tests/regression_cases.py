"""
Regression cases of DiES: a fixed set of outputs that a refactoring must not change.

Phase 2 (Nov 2026) vectorizes the core. Every output listed here is stored in
tests/reference_data/regression_reference.npz (computed with the commit recorded in
tests/test_regression.py); tests/test_regression.py recomputes them with the current
code and compares them bit by bit (or with an explicit, documented tolerance).

The outputs are computed with whatever DiES modules come first on sys.path, so
the same file serves to build the reference from an older commit (in a git
worktree) and to check the current code:

    python tests/regression_cases.py --code-root <folder with DiES_core.py> --out file.npz

Use tests/make_regression_reference.py to rebuild the reference from a commit.

Outputs (keys of the .npz file, joined with '__'):
    static__<case>__litho<0|1>__<field>   every field of model.compute()
    legacy__<case>__litho<0|1>__<name>    the legacy methods (model_displacements,
                                          model_stresses, principal_stresses,
                                          plane_stresses) called directly
    dynamic__<case>__<field>              every field of every step of
                                          compute_dynamic_model, stacked as
                                          (step, row, column), plus final_mode
                                          and failure_step
    mohr__<sn|ss|angle|mode>              evaluate_stress_state on a sweep of
                                          (p, s) states
    planes__orientation                   failure_plane_orientation on a sweep
"""

import argparse
import os
import sys
import warnings

import numpy as np

# The cases are those of tests/visual_check.py (vertical dike, opening only;
# 60-degree pure-shear fault with sigma_II = 15 MPa) plus a dilatant fault with
# remote stresses and mode III, which exercises every term of the solution.
MODEL_INPUTS = [30e9, 0.25, 2900, 0.57, -1.5e6]   # E, nu, density, cfi, T
# Earth gravity for the tests since 25 Nov 2026 (Sam's convention); Mars (3.71) until then
G_EARTH = 9.81
XLIMS = [-5000, 5000]
YLIMS = [-6000, -10]
# A coarser grid than visual_check (250 m instead of 100 m) keeps the test fast;
# it still has ~1000 nodes, nodes on the crack and nodes on its tips.
RESOLUTION = [250, 250]
# The dynamic models evaluate the failure criterion node by node (until Phase 2,
# point 3), so they run on a 500 m grid (~250 nodes) to keep the test under a minute.
RESOLUTION_DYNAMIC = [500, 500]
FAULT_LOWER_Y = -1000 - 1000*np.tan(np.radians(60))

# name: (crack_inputs, dx, dy, dz, rx, ry)
# crack_inputs: [total aperture, x upper tip, y upper tip, x lower tip, y lower tip]
CASES = {
    'dike': ([2, 0, -1000, 0, -5000], 0, 0, 0, 0, 0),
    'fault': ([0, -500, -1000, 500, FAULT_LOWER_Y], 0, 15e6, 0, 0, 0),
    'dilatant_fault': ([0, -500, -1000, 500, FAULT_LOWER_Y], 2e6, 10e6, 5e6, -3e6, -1e6),
}

# Driving stress added at each step of the dynamic models (Sam's convention for tests)
STEP_MPA = 0.25


def import_dies(code_root):
    """Puts the DiES modules of code_root first on sys.path and imports them."""
    if code_root:
        sys.path.insert(0, os.path.abspath(code_root))
    import DiES_core, Helper_functions, Modeling_functions, Failure_Determination
    return DiES_core, Helper_functions, Modeling_functions, Failure_Determination


def driving_stress_x(case, core, helpers):
    """dx of the crack: the input dx for a fault, or derived from the aperture (as single_model)."""
    ci, dx = CASES[case][0], CASES[case][1]
    if ci[0] == 0:
        return dx
    E, nu = MODEL_INPUTS[0], MODEL_INPUTS[1]
    b = helpers.points_distance((ci[1], ci[2]), (ci[3], ci[4]))
    return ci[0]*helpers.G(E, nu)/(b*(1 - nu))


def investigation_range(case, core, helpers):
    """Fractions of the final driving stress so that each step adds STEP_MPA MPa."""
    dy, dz = CASES[case][2], CASES[case][3]
    final = max(abs(driving_stress_x(case, core, helpers)), abs(dy), abs(dz))
    n = max(1, int(round(final/(STEP_MPA*1e6))))
    return [i/n for i in range(1, n + 1)]


def static_outputs(core, helpers, out):
    """model.compute() and the legacy methods, for every case, with and without lithostatic load."""
    E, nu, density, cfi, T = MODEL_INPUTS
    for case, (ci, dx, dy, dz, rx, ry) in CASES.items():
        disc = core.crack((ci[1], ci[2]), (ci[3], ci[4]), driving_stress_x(case, core, helpers), dy, dz)
        m = core.model(XLIMS, YLIMS, RESOLUTION[0], RESOLUTION[1], 500, E, nu, density, cfi, T, G_EARTH)
        for litho in (False, True):
            prefix = case + '__litho' + str(int(litho)) + '__'
            res = m.compute(disc, rx, ry, litho)
            for name in res.names():
                out['static__' + prefix + name] = np.asarray(res[name])
            # The legacy methods, called directly (they are what the old scripts use)
            for name, value in zip(('ux', 'uy', 'uz'), m.model_displacements(disc)):
                out['legacy__' + prefix + 'model_displacements_' + name] = value
            for name, value in zip(('sxx', 'syy', 'sav', 'sxy', 'sxz', 'szz', 'tmax'),
                                   m.model_stresses(disc, rx, ry, litho)):
                out['legacy__' + prefix + 'model_stresses_' + name] = value
            for name, value in zip(('s1', 's3', 's1a', 's3a'), m.principal_stresses(disc, rx, ry, litho)):
                out['legacy__' + prefix + 'principal_stresses_' + name] = value
            for name, value in zip(('sn', 'ss'), m.plane_stresses(disc, rx, ry, litho)):
                out['legacy__' + prefix + 'plane_stresses_' + name] = value


def dynamic_outputs(core, helpers, modeling, out):
    """Every field of every step of compute_dynamic_model, stacked as (step, row, column)."""
    for case, (ci, dx, dy, dz, rx, ry) in CASES.items():
        steps = investigation_range(case, core, helpers)
        dyn = modeling.compute_dynamic_model(ci, MODEL_INPUTS, dx, dy, dz, rx, ry, G_EARTH, True,
                                             steps, XLIMS, YLIMS, RESOLUTION_DYNAMIC)
        for name in dyn[0].names():
            out['dynamic__' + case + '__' + name] = np.stack([np.asarray(step[name]) for step in dyn])
        out['dynamic__' + case + '__final_mode'] = np.asarray(dyn.final_mode())
        out['dynamic__' + case + '__failure_step'] = np.asarray(dyn.failure_step())


def mohr_outputs(failure, out):
    """evaluate_stress_state on a sweep of (p, s) states, from tension to compression."""
    cfi, T = MODEL_INPUTS[3], MODEL_INPUTS[4]
    # A coarse sweep over the whole range, and a fine one (0.1 MPa) near the
    # Griffith parabola, where the narrow tension-shear modes ('Mode I',
    # 'Mixed-mode I-II') are found
    coarse = np.arange(-4e6, 30.01e6, 0.85e6)
    pairs = [(p, s) for p in coarse for s in coarse]
    pairs += [(p, s) for p in np.arange(0, 8.01e6, 0.1e6) for s in np.arange(-1.6e6, 1.01e6, 0.1e6)]
    sn, ss, angle, mode = [], [], [], []
    for p, s in pairs:
        # Only p >= s is a valid input (p is the maximum compressive stress)
        if p < s:
            continue
        try:
            result = failure.evaluate_stress_state(p, s, cfi, T, 'values')
        except UnboundLocalError:
            # Bug of the sampling algorithm used until Phase 2, point 3B (found on
            # 25 Nov 2026): when a sample of the circle fell exactly on sigma_n = 0
            # (e.g. s = 0), no branch assigned the results and the function crashed.
            # Kept so that the cases can still run with older commits.
            result = (np.nan, np.nan, np.nan, 'Error')
        sn.append(result[0]) ; ss.append(result[1]) ; angle.append(result[2]) ; mode.append(result[3])
    out['mohr__sn'] = np.asarray(sn, dtype=float)
    out['mohr__ss'] = np.asarray(ss, dtype=float)
    out['mohr__angle'] = np.asarray(angle, dtype=float)
    out['mohr__mode'] = np.asarray(mode)


def plane_outputs(failure, out):
    """failure_plane_orientation on a sweep of sigma1 orientations, angles, modes and sides."""
    orientations = []
    for mode in ('Mode I', 'Mixed-mode I-II', 'Mode II - tension-compression', 'Mode II - compression', 'Below'):
        for s1a in np.arange(-89.5, 90.1, 7.5):
            for theta in (55.0, 62.5, 70.0, 90.0):
                for side in (-1, 1):
                    orientations.append(failure.failure_plane_orientation(s1a, theta, mode, side))
    out['planes__orientation'] = np.asarray(orientations, dtype=float)


def compute_all(code_root=None):
    """Returns a dict with every regression output, computed with the code in code_root."""
    warnings.filterwarnings('ignore')
    core, helpers, modeling, failure = import_dies(code_root)
    out = {}
    static_outputs(core, helpers, out)
    dynamic_outputs(core, helpers, modeling, out)
    mohr_outputs(failure, out)
    plane_outputs(failure, out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--code-root', required=True, help='folder with the DiES modules to run')
    parser.add_argument('--out', required=True, help='.npz file to write')
    args = parser.parse_args(argv)
    out = compute_all(args.code_root)
    np.savez_compressed(args.out, **out)
    print('Saved', len(out), 'outputs to', args.out)


if __name__ == '__main__':
    main()
