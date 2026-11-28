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
| `plotting.py` | Every figure, drawn from results: `plot_displacement`, `plot_stresses`, `plot_principal_stresses`, `plot_stress_trajectories`, `plot_plane_stresses`, `plot_crack_displacements`, `plot_dynamic_step`, `plot_mohr_circle`. |

## Static model

```python
from DiES_core import model, crack
import plotting

current_model = model(xlims, ylims, dx, dy, 500, E, nu, density, friction, T, g)
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

fig = plotting.plot_stresses(res, 'sxy')               # automatic contour step
fig = plotting.plot_stresses(res, 'sxy', 1)            # or a step of your choice (MPa)
fig = plotting.plot_principal_stresses(res, 'angle')
fig = plotting.plot_stress_trajectories(res, planes=True)
fig = plotting.plot_plane_stresses(res, 'normal', 60)
fig = plotting.plot_displacement(res, 'uxy', 'vectors')
fig = plotting.plot_crack_displacements(crack, nu, E)  # opening and slip of the crack walls
s, walls = crack.wall_displacements(nu, E)             # the same profiles, as arrays
```

**Contour steps** (since Phase 2, Nov 2026): the `step` of every figure is optional. Without
it, a round step (1, 2, 2.5 or 5 times a power of ten) giving at most ~12 contours between the
1st and 99th percentiles of the field is chosen (`Helper_functions.auto_contour_step`), and the
contour levels are multiples of the step. It is a starting point: pass a step to choose another
one after looking at the figure. A field that is the same everywhere (e.g. `sxz` without mode III
loading) shows a note instead of an empty colour scale. The orientation of σ1 (`'angle'`) uses
a cyclic colour scale (`twilight`, −90° to 90°, the same colour for both ends, which are the same
orientation) and no contours.

`compute(..., displacements=True, stresses=False)` computes only part of the fields, for
instance for a displacement figure.

**Fields** (the units and descriptions are in `results.FIELD_INFO`):

- material at each node: `E`, `nu`, `density`, `cohesion` (2|T|), `friction` (the coefficient of internal
  friction, μ′; named `cfi` until Nov 2026, and older files load with the new name), `coula`, `T`;
- displacements, in m: `ux`, `uy`, `uz`;
- stresses, in Pa, tension positive, including the remote stresses and, with `litho=True`, the
  lithostatic ones: `sxx`, `syy`, `sav`, `sxy`, `sxz`, `szz`, `tmax`;
- principal stresses, in Pa, compression positive: `s1`, `s3`;
- orientations, in degrees counterclockwise from +x, in (−90°, 90°]: `s1a`, `s3a`;
- stresses on the optimal shear planes, in Pa, compression positive: `sn`, `ss`.

**Drawing options** (Phase 4, Nov 2026; the plotting options of the app). The map figures take
keyword options; their defaults give the figures of before, pixel for pixel
(`plotting.STYLE_DEFAULTS`; an unknown or invalid option raises `PlotError`):

| Option | Default | Effect | Figures |
|---|---|---|---|
| `cmap` | the figure's own (`viridis`; `twilight` for the angle) | any matplotlib colour map | all maps |
| `limits` | 1st and 99th percentiles | `(vmin, vmax)` of the colours | colour maps, arrows |
| `center_zero` | `False` | colours symmetric around 0 (ignored with `limits`); with a diverging map (`RdBu_r`, `coolwarm`), tension and compression get opposite colours and 0 is white | colour maps, arrows |
| `contours` | `True` | draw the contours | colour maps |
| `contour_labels` | `True` | label the contours | colour maps |
| `info_box` | `True` | the box with the driving stresses (and the material) | displacements, `plot_plane_stresses` |
| `arrow_scale` | `1.0` | length of the arrows and trajectory segments, relative to the default | displacement arrows, trajectories |

```python
fig = plotting.plot_stresses(res, 'sxx', cmap='RdBu_r', center_zero=True, contour_labels=False)
fig = plotting.plot_plane_stresses(res, 'normal', 60, limits=(-5, 40), info_box=False)
fig = plotting.plot_displacement(res, 'uxy', 'vectors', arrow_scale=2)
```

For the angle of σ1 only `cmap` is used (it should stay a cyclic map); the trajectories take only
`arrow_scale`.

**Failure state of a single model** (Phase 4, Nov 2026): how close each node is to the
Griffith-Coulomb envelope, with the crack loaded at once (no steps).

```python
from Modeling_functions import compute_failure_state     # also dies.compute_failure_state

state = compute_failure_state(res)          # a DiESResults
state['failure_ratio']        # r/r_c: radius of the Mohr circle over the critical radius (1: touching)
state['failure_mode']         # 'Below', 'Above', a failure mode or 'Singular' (evaluate_stress_states)
state['failure_contact_mode'] # the mode where the circle would touch the envelope, at every node
state['failure_sn'], state['failure_ss'], state['failure_theta']   # on the failure plane, where at failure

fig = plotting.plot_failure_modes(state)    # one colour per state; legend under the map with the share
                                            # of each state among the nodes that fail
plotting.failure_mode_shares(state['failure_mode'])   # [(state, nodes, percent or None), ...]
fig = plotting.plot_failure_ratio(state)    # r/r_c, white at 1, with the edges of the failure band
```

The shares are relative to the nodes that fail, i.e. all but `Below` and `Singular`, which are
counted but get no share; the legend leaves out `Below` (the grey of the map) (Sam's choice,
27 Nov 2026). `plot_failure_modes` also draws the `failure_mode` of a step of a dynamic model. Its colours are
those of `tests/failure_check.py`: below failure light grey, above it red, Mode I blue, mixed-mode
I-II dodgerblue, Mode II tension-compression purple, Mode II compression orange, singular black.

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
color_range = plotting.dynamic_color_range(dyn)  # sigma_n range of the final step
fig = plotting.plot_dynamic_step(dyn[k], 2, color_range)
dyn.save('dynamic.npz') ; dyn = DynamicResults.load('dynamic.npz')

# Phase 4, step 3 (Nov 2026)
dyn.step_loads()                   # driving stress of each step: the largest of |dx|, |dy|, |dz| (Pa)
dyn.failure_load()                 # driving stress at which each node first failed (Pa, NaN = never)
dyn.failure_load(10)               # the same, counting only the first 10 steps
fig = plotting.plot_dynamic_step(dyn[k], 2, color_range, cmap='magma', traces=False)
fig = plotting.plot_field(dyn[k], 'failure_ss', factor=1.0)   # any field as a colour map
fig = plotting.plot_first_failure(dyn, upto=10)               # load of the first failure, up to step 10
fig = plotting.plot_failure_evolution(dyn, current=10)        # nodes of each failure state vs the load
```

Figures of the dynamic model (the app plays them step by step, see `docs/app.md`):

| Function | What it draws | Options |
|---|---|---|
| `plot_dynamic_step(step, downscaling, color_range)` | σn on the failure planes, with the traces of the planes (one every `downscaling` nodes), the line σn = 0 and a red mesh over the nodes above failure. Unchanged by default. | `cmap`, `limits` (instead of `color_range`), `contours` (σn = 0), `info_box`, `arrow_scale` (trace length), `above` (red mesh), `traces` |
| `plot_field(res, field, label, factor, step)` | Any field as a colour map with contours (e.g. `failure_ss`, τ on the failure planes). | as `plot_stresses` |
| `plot_first_failure(dyn, upto, unit)` | The driving stress (`unit='MPa'`), the share of the final load (`'percent'`) or the step (`'step'`) at which each node first failed, counting the first `upto` steps; light grey where it has not failed. The colours keep the range of the whole run, so that the failure zone can be watched growing. | `cmap` (viridis), `limits`, `info_box` |
| `plot_failure_evolution(dyn, current, share)` | Number of nodes (or % of all, `share=True`) in each failure state against the driving stress, in the colours of `plot_failure_modes`; a dashed line at step `current`. | |

The failure-state map of a step is `plot_failure_modes(dyn[k])`.

Each step holds `failure_sn`, `failure_ss` (MPa), `failure_theta` (degrees), `failure_mode`,
`above_failure`, `failure_jumped` and the seven orientations of the failure planes `plane_*`
(radians, counterclockwise from +x). `dynamic_model(..., plotting=True, folder=...)` still saves
the same figures, with the same file names.

Since Phase 2, point 3B (Nov 2026), the failure criterion is analytical (see
`docs/failure_criterion.md`). A node that goes from `'Below'` to `'Above'` between two steps
(it jumped over the failure band) keeps the mode, stresses and plane of the point where its Mohr
circle touched the envelope, and `failure_jumped` is 1 there (0 elsewhere). The metadata of the
run include `failure_criterion` (`'analytical'`) and `failure_tolerance`.

## Classes and their methods

`model.compute()` and `compute_dynamic_model()` are functions, not classes: they compute and
return an object of one of the two result classes of `results.py`.

### `DiESResults` — one evaluation of a model

Returned by `model.compute()`, and one per step inside `DynamicResults`.

| Method or attribute | What it does |
|---|---|
| `res['sxx']`, `res.sxx` | A field (2D array) by name, as an item or as an attribute. |
| `'sxx' in res` | Whether the field exists. |
| `names()` | Names of the fields it holds. |
| `shape` | Shape of the grid (property). |
| `x_nodes()`, `y_nodes()` | x of each grid column and y of each grid row (1D arrays). |
| `units`, `metadata` | Units of each field; inputs of the model (JSON-serializable dict). |
| `probe(x, y, names=None)` | Values at the grid node nearest to (x, y), as a dict. |
| `profile(A, B, n=200, names=None)` | Values along the segment A→B at `n` points, interpolated bilinearly, as a dict. |
| `mask(condition, name)` | The field `name` as an array, with `NaN` where `condition` is False. |
| `save(path)`, `DiESResults.load(path)` | Save to and load from `.npz` plus JSON metadata. |
| `to_csv(path, names=None)` | One row per node: x, y and the chosen fields, with units in the header. |
| `to_xarray()` | Optional export to an `xarray.Dataset` (only if xarray is installed). |

### `DynamicResults` — the steps of a dynamic model

Returned by `compute_dynamic_model()`. It holds one `DiESResults` per step, with the failure
fields listed above, and the metadata of the whole run.

| Method or attribute | What it does |
|---|---|
| `len(dyn)` | Number of steps. |
| `dyn[k]` | The `DiESResults` of step `k` (from 0; `dyn[-1]` is the last one). |
| `for step in dyn` | Iterates over the steps. |
| `steps`, `metadata` | The list of `DiESResults`; inputs of the run. |
| `final_mode()` | 2D array with the failure mode of each node at the last step (what `dynamic_model` returns). |
| `failure_step()` | 2D array with the step (1, 2, ...) at which each node first failed, including `'Above'`; 0 where it never did. |
| `step_loads()` | The driving stress of each step (Pa): the largest absolute value of dx, dy and dz in its metadata. |
| `failure_load(upto=None)` | 2D array with the driving stress (Pa) at which each node first failed, NaN where it never did (or not in the first `upto` steps). |
| `save(path)`, `DynamicResults.load(path)` | Save to and load from `.npz` plus JSON metadata. |

Each step also has every `DiESResults` method (`probe`, `profile`, `mask`, `to_csv`...).

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
- No aesthetic change in Phase 1. Phase 2 (Nov 2026) changed the figures:
  - automatic contour steps (above), in `single_model` too; levels at multiples of the step;
  - major ticks at round intervals that do not overlap (at most 6 labels in x);
  - σ1 orientation with a cyclic colour scale;
  - stress trajectories and failure-plane traces scaled with the spacing of the grid (until then
    111 m and 250 m whatever the grid), and the two colour bars of the trajectories under the map;
  - the dynamic model draws every step with the σn colour range of the final step
    (`dynamic_model` and `dynamic_color_range`; `plot_dynamic_step` without it keeps the range of
    the step), with its text box inside the axes;
  - `plot_crack_displacements` (and `crack.plot_crack_displacements`; until then
    `plot_crack_opening` and `crack.plot_mode_I_crack_displacements`, names kept as aliases)
    draws the opening and the
    slip of both walls, and mode III if any, from the core solution (`crack.wall_displacements`);
    until then only the mode I opening, so a fault came out flat. Rounded values in its legend,
    and the half-length of the crack labelled as such (it was 'a', the symbol of the aperture);
  - fields that are the same everywhere show a note;
  - colour bars with automatic tick labels (decimals only when the range needs them; the dynamic
    model had a fixed '%.2f');
  - the text boxes with the driving stresses, anchored at their bottom inside the axes.

## Tests

- `tests/test_results.py` (7): `compute()` against the existing functions, probe, profile, mask,
  save/load, CSV and xarray.
- `tests/test_plotting.py` (5): every wrapper draws the same figure as `plotting.py`, pixel by
  pixel; the wrappers compute once; invalid requests; figures from saved results; Mohr circle.
- `tests/test_dynamic_results.py` (4): `compute_dynamic_model` against `dynamic_model`, first
  failure kept, `failure_step`, save/load, saved figures equal to `plot_dynamic_step`.
- `pytest tests/`: 42 tests at the end of Phase 1. Phase 2 adds the regression against a stored
  reference (`tests/test_regression.py`, with `tests/compare_figures.py` for the figures) and the
  tests of the analytical failure criterion (`tests/test_failure_criterion.py`).
