"""
Tests of results.DiESResults and model.compute() (Phase 1, 2026-11-24).

Run with pytest, or directly: python tests/test_results.py
"""

import csv
import os
import sys
import tempfile
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from DiES_core import model, crack
from Helper_functions import G
from results import DiESResults

warnings.filterwarnings('ignore')

E, NU, RHO, FRICTION, T = 30e9, 0.25, 2900, 0.57, -1.5e6


def small_case():
    m = model((-2000, 2000), (-3000, -10), 250, 250, 500, E, NU, RHO, FRICTION, T, 9.81)
    up, lo = (0, -500), (0, -2000)
    disc = crack(up, lo, 2*G(E, NU)/(1500*(1 - NU)), 0, 0)
    return m, disc


def test_compute_matches_the_existing_functions():
    m, disc = small_case()
    for litho in (True, False):
        res = m.compute(disc, 1e6, -2e6, litho)
        ux, uy, uz = m.model_displacements(disc)
        sxx, syy, sav, sxy, sxz, szz, tmax = m.model_stresses(disc, 1e6, -2e6, litho)
        s1, s3, s1a, s3a = m.principal_stresses(disc, 1e6, -2e6, litho)
        sn, ss = m.plane_stresses(disc, 1e6, -2e6, litho)
        expected = dict(ux=ux, uy=uy, uz=uz, sxx=sxx, syy=syy, sav=sav, sxy=sxy, sxz=sxz, szz=szz,
                        tmax=tmax, s1=s1, s3=s3, s1a=s1a, s3a=s3a, sn=sn, ss=ss)
        for name, values in expected.items():
            assert np.array_equal(res[name], values, equal_nan=True), (litho, name)
        grid = m.grid()
        assert np.array_equal(res.x, grid[0]) and np.array_equal(res.y, grid[1])
        assert np.allclose(res.cohesion, 2*abs(T)) and np.allclose(res.T, T)
        assert res.metadata['litho'] is litho and res.metadata['crack']['dpx'] == disc.x()
        assert res.units['s1'] == 'Pa' and res.units['s1a'] == 'deg' and res.units['ux'] == 'm'


def test_partial_compute():
    m, disc = small_case()
    res = m.compute(disc, displacements=True, stresses=False)
    assert 'ux' in res and 's1' not in res
    res = m.compute(disc, displacements=False, stresses=True)
    assert 'ux' not in res and 's1' in res


def synthetic():
    x, y = np.meshgrid(np.arange(0., 50., 10.), np.arange(-40., 0., 10.))
    fields = {'a': 2*x + 3*y, 'b': x*0 + 7.0}
    return DiESResults(x, y, fields, {'note': 'synthetic'}, units={'a': 'Pa'})


def test_access_and_errors():
    res = synthetic()
    assert np.array_equal(res.a, res['a']) and res.names() == ['a', 'b']
    assert res.shape == (4, 5) and 'a' in res and 'c' not in res
    try:
        res.c
    except AttributeError:
        pass
    else:
        raise AssertionError('missing field did not raise AttributeError')
    try:
        DiESResults(res.x, res.y, {'bad': np.zeros((2, 2))})
    except ValueError:
        pass
    else:
        raise AssertionError('wrong shape accepted')


def test_probe_and_profile():
    res = synthetic()
    p = res.probe(21, -19)                         # nearest node (20, -20)
    assert (p['x'], p['y'], p['a'], p['b']) == (20.0, -20.0, 2*20 + 3*(-20), 7.0)
    prof = res.profile((0, -40), (40, -10), n=31)
    # A linear field is interpolated exactly
    assert np.allclose(prof['a'], 2*prof['x'] + 3*prof['y'])
    assert np.isclose(prof['distance'][-1], 50.0)
    outside = res.profile((100, -40), (120, -10), n=5)
    assert np.all(np.isnan(outside['a']))


def test_mask():
    res = synthetic()
    masked = res.mask(res.x > 20, 'a')
    assert np.all(np.isnan(masked[res.x <= 20])) and np.allclose(masked[res.x > 20], res.a[res.x > 20])


def test_save_load_and_csv():
    m, disc = small_case()
    res = m.compute(disc, 0, 0, True)
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, 'model.npz')
    res.save(path)
    back = DiESResults.load(path)
    assert back.names() == res.names() and back.metadata == res.metadata and back.units == res.units
    for name in res.names():
        assert np.array_equal(back[name], res[name], equal_nan=True), name
    csv_path = os.path.join(folder, 'model.csv')
    res.to_csv(csv_path, names=['s1', 'ux'])
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['x [m]', 'y [m]', 's1 [Pa]', 'ux [m]']
    assert len(rows) == 1 + res.x.size
    assert np.isclose(float(rows[1][2]), res.s1.ravel()[0])


def test_to_xarray_is_optional():
    res = synthetic()
    try:
        import xarray  # noqa: F401
    except ImportError:
        try:
            res.to_xarray()
        except ImportError:
            return
        raise AssertionError('to_xarray did not raise ImportError without xarray')
    ds = res.to_xarray()
    assert ds['a'].attrs['units'] == 'Pa' and np.array_equal(ds['a'].values, res.a)

def test_files_saved_with_old_names_load_with_the_new_ones(tmp_path):
    """Results saved before cfi was renamed friction (Nov 2026) open with the new names."""
    import json
    m = model((-2000, 2000), (-3000, -10), 250, 250, 500, E, NU, RHO, FRICTION, T, 9.81)
    res = m.compute(crack((0, -500), (0, -2000), 5e6, 0, 0), 0, 0, True)
    path = str(tmp_path / 'old.npz')
    res.save(path)
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    # Rewrite the file as one saved with the old names
    arrays['field_cfi'] = arrays.pop('field_friction')
    units = json.loads(str(arrays['units_json'])) ; units['cfi'] = units.pop('friction')
    meta = json.loads(str(arrays['metadata_json'])) ; meta['layers_cfi'] = meta.pop('layers_friction')
    arrays['units_json'] = np.array(json.dumps(units)) ; arrays['metadata_json'] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    old = DiESResults.load(path)
    assert 'friction' in old and 'cfi' not in old and 'layers_friction' in old.metadata
    assert np.array_equal(old['friction'], res['friction'])


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('OK  ', test.__name__)
    print(len(tests), 'tests passed')
