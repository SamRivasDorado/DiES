"""
Regression test: the current code must reproduce the outputs of commit 0755fe7
(end of Phase 1) stored in tests/reference_data/regression_0755fe7.npz.

The comparison is bit by bit (NaN equal to NaN, same NaN pattern), except for
the outputs listed in TOLERANCES, where a documented change in the floating-point
operations is allowed to change the last digits. A deliberate change of the
results (e.g. a new failure criterion) must update the reference with
tests/make_regression_reference.py and be recorded in the roadmap.

The cases and outputs are defined in tests/regression_cases.py.
"""

import os
import sys

import numpy as np
import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
for folder in (ROOT, TESTS):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import regression_cases

REFERENCE = os.path.join(TESTS, 'reference_data', 'regression_0755fe7.npz')

# Key prefix -> largest allowed |difference|, relative to the largest |value| of
# the reference field (so 1e-13 is 1e-5 Pa for a field of 100 MPa).
#
# Phase 2, point 2 (25 Nov 2026): model_displacements and model_stresses evaluate the
# whole grid in one call instead of node by node. numpy computes np.arctan2 (used by
# tripolar_coordinates) with vectorized routines for arrays, which round the last bit
# differently from the scalar routine in ~8% of the values. As a result 0.6% of the
# static values differ, by at most 3.1e-16 of the field maximum (float64 precision).
# The NaN pattern, the failure modes and the dynamic models are identical bit by bit.
TOLERANCES = {
    'static__': 1e-13,
    'legacy__': 1e-13,
}


@pytest.fixture(scope='module')
def outputs():
    """The reference and the current outputs, computed once for every test of this module."""
    with np.load(REFERENCE) as data:
        reference = {key: data[key] for key in data.files}
    return reference, regression_cases.compute_all()


def tolerance(key):
    """Relative tolerance of a key (see TOLERANCES), or None for a bit-by-bit comparison."""
    for prefix, value in TOLERANCES.items():
        if key.startswith(prefix):
            return value
    return None


def test_same_outputs(outputs):
    reference, current = outputs
    assert sorted(current) == sorted(reference), 'the set of outputs changed'


@pytest.mark.parametrize('group', ['static', 'legacy', 'dynamic', 'mohr', 'planes'])
def test_outputs_match_reference(outputs, group):
    reference, current = outputs
    mismatches = []
    for key in sorted(k for k in reference if k.startswith(group + '__')):
        ref, cur = reference[key], current[key]
        if ref.shape != cur.shape:
            mismatches.append(key + ': shape ' + str(cur.shape) + ' instead of ' + str(ref.shape))
            continue
        # Failure modes and other labels: exact equality
        if ref.dtype.kind in 'US':
            if not np.array_equal(ref, cur):
                mismatches.append(key + ': ' + str(int(np.sum(ref != cur))) + ' labels differ')
            continue
        tol = tolerance(key)
        if tol is None:
            same = np.array_equal(ref, cur, equal_nan=True)
        else:
            # Same NaN pattern, and differences below tol times the largest |value|
            scale = np.nanmax(np.abs(ref)) if np.isfinite(ref).any() else 0.0
            same = (np.array_equal(np.isnan(ref), np.isnan(cur))
                    and np.allclose(cur, ref, rtol=0, atol=tol*scale, equal_nan=True))
        if not same:
            finite = np.isfinite(ref) & np.isfinite(cur)
            worst = np.max(np.abs(ref[finite] - cur[finite])) if finite.any() else float('nan')
            nan_changes = int(np.sum(np.isnan(ref) != np.isnan(cur)))
            mismatches.append(key + ': max |difference| = ' + repr(float(worst))
                              + ', NaN pattern changes = ' + str(nan_changes))
    assert not mismatches, '\n'.join(mismatches)
