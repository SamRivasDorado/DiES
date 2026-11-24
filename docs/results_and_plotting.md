# Results and figures in DiES (Phase 1, 2026-11-24)

Since Phase 1 (branch `phase1-separate-plotting`), DiES separates **computing** a model from
**drawing** it. A model is computed once into a results object; figures, exports and the future
application read from that object instead of recomputing. Everything that worked before still
works the same way: `model.plot_*`, `single_model` and `dynamic_model` keep their signatures,
figures and return values, and now use the new pieces internally.

Verified on 2026-11-24 against commit `164b127`, the starting point of the branch, with 177
outputs identical pixel by pixel or bit for bit: the figures of `single_model` (13 parameters, two
cases, with and without lithostatic stresses), the 42 figures of `tests/plot_gallery.py`, 58
numerical fields, the figures and failure modes of `dynamic_model` in three cases, and the Mohr
circle in its six possible states.

## The pieces

| File | What it contains |
|---|---|
| `results.py` | `DiESResults` (one evaluation of a model) and `DynamicResults` (one `DiESResults` per step of the dynamic model). DiES's own classes on numpy arrays; no new dependency. |
| `DiES_core.py` | `model.compute()`, which computes every field once, and the pure functions `principal_from_stresses()` and `optimal_plane_stresses()`. |
| `Modeling_functions.py` | `compute_dynamic_model()`, the dynamic model without drawing. |
| `plotting.py` | Every figure, drawn from results: `plot_displacement`, `plot_stresses`, `plot_principal_stresses`, `plot_stress_trajectories`, `plot_plane_stresses`, `plot_crack_opening`, `plot_dynamic_step`, `plot_mohr_circle`. |

## Static model

```python
from DiES_core import model, crack
import plotting

current_model = model(xlims, ylims, dx, dy, 500, E, nu, density, cfi, T, g)
disc = crack(upper_tip, lower_tip, dpx, dpy, dpz)

res = current_model.compute(disc, rxx, ryy, litho=True)    # computed once

res.s1                  # 2D array, Pa (compression positive)
res['sxy']              # access by name (tension positive)
res.units['s1a']        # 'deg'
res.names()             # every field held
res.metadata            # model, crack, remote stresses, litho, g...

res.probe(500, -1200)                         # every field at the nearest node
res.profile((-3000, -1500), (3000, -1500))    # every field along A -> B (bilinear)
res.mask(res.s3 < 0, 's1')                    # sigma1 only where sigma3 is tensile

res.save('model.npz') ; res = DiESResults.load('model.npz')
res.to_csv('model.csv', names=['s1', 's3', 'ux'])
res.to_xarray()         # optional: only needs xarray when called

fig = plotting.plot_stresses(res, 'sxy', 1)             # or the former key 'sxya'
fig = plotting.plot_principal_stresses(res, 'angle', 10)
fig = plotting.plot_stress_trajectories(res, planes=True)
fig = plotting.plot_plane_stresses(res, 'normal', 60, 5)
fig = plotting.plot_displacement(res, 'uxy', 'vectors', 0.25)
```

`compute(..., displacements=True, stresses=False)` computes only part of the fields, for
instance for a displacement figure.

**Fields** (the units and descriptions are in `results.FIELD_INFO`):

- material at each node: `E`, `nu`, `density`, `cohesion` (2|T|), `cfi`, `coula`, `T`;
- displacements, in m: `ux`, `uy`, `uz`;
- stresses, in Pa, tension positive, including the remote stresses and, with `litho=True`, the
  lithostatic ones: `sxx`, `syy`, `sav`, `sxy`, `sxz`, `szz`, `tmax`;
- principal stresses, in Pa, compression positive: `s1`, `s3`;
- orientations, in degrees counterclockwise from +x, in (−90°, 90°]: `s1a`, `s3a`;
- stresses on the optimal shear planes, in Pa, compression positive: `sn`, `ss`.

## Dynamic model

```python
from Modeling_functions import compute_dynamic_model
from results import DynamicResults

dyn = compute_dynamic_model(crack_inputs, model_inputs, dx, dy, dz, rx, ry, g, litho,
                            investigation_range, xlims, ylims, resolution)
len(dyn) ; dyn[-1]                 # number of steps ; last step (a DiESResults)
dyn.final_mode()                   # failure mode of each node at the last step
dyn.failure_step()                 # step at which each node first failed (0 = never)
dyn[k].metadata['dx']              # driving stresses, aperture/slip... of step k
fig = plotting.plot_dynamic_step(dyn[k], downscaling=2)
dyn.save('dynamic.npz') ; dyn = DynamicResults.load('dynamic.npz')
```

Each step holds `failure_sn`, `failure_ss` (MPa), `failure_theta` (degrees), `failure_mode`,
`above_failure` and the seven orientations of the failure planes `plane_*` (radians,
counterclockwise from +x). `dynamic_model(..., plotting=True, folder=...)` still saves the same
figures, with the same file names.

## What changed for existing code

- `model.plot_*`, `crack.plot_mode_I_crack_displacements`, `single_model`, `dynamic_model`:
  - same signatures, same figures and same return values;
  - each figure computes the model once (`plot_principal_stresses` used to compute the stresses
    up to three times, and `plot_plane_stresses` twice);
  - an invalid option raises `InputError` before computing (`plot_stresses` used to raise
    `UnboundLocalError`).
- `evaluate_stress_state(..., 'plot')` returns the same figure, drawn by `plotting.plot_mohr_circle`.
  An `output` other than `'values'` or `'plot'` raises `InputError` (it used to return `None`).
- The figures of `plotting.py` raise `plotting.PlotError` for an invalid option or a field that
  the results do not contain.
- No aesthetic change: the improvements listed in the roadmap are planned for Phase 2.

## Tests

- `tests/test_results.py` (7): `compute()` against the existing functions, probe, profile, mask,
  save/load, CSV and xarray.
- `tests/test_plotting.py` (5): every wrapper draws the same figure as `plotting.py`, pixel by
  pixel; the wrappers compute once; invalid requests; figures from saved results; Mohr circle.
- `tests/test_dynamic_results.py` (4): `compute_dynamic_model` against `dynamic_model`, first
  failure kept, `failure_step`, save/load, saved figures equal to `plot_dynamic_step`.
- `pytest tests/`: 42 tests in total.
