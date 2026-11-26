# Scenarios: a DiES model in one file

_Phase 3 (Nov 2026). Code: `dies/scenario.py`. Examples: `examples/dike_example.yaml` (dynamic)
and `examples/fault_example.yaml` (static). Tests: `tests/test_scenario.py`._

A scenario describes a complete model in one YAML (or JSON) file: the grid, the gravity, the
layers, the crack, the loading, the analysis and what to write. DiES reads it, checks it and runs
it with the functions of the core, which are not modified:

```python
import dies

scenario = dies.load_scenario('examples/dike_example.yaml')
run = scenario.run()        # ScenarioResults
run.static                  # DiESResults of the crack at its final load
run.dynamic                 # DynamicResults (None for a static analysis)
run.write('results/')       # results files, figures and a copy of the scenario
```

The results are identical, bit for bit, to those of `model.compute` and
`compute_dynamic_model` called with the same values (checked in `tests/test_scenario.py` for a
dike, a pure-shear fault, a dilatant fault with remote stresses and mode III, and two layers).

## Units

Every quantity with dimensions is written with its unit; **a number without a unit is an error**,
so that a forgotten factor of 1e6 cannot pass. Internally every value is converted to the units of
the core (m, Pa, kg m-3, m s-2, degrees).

| Kind | Accepted units (case sensitive) | Example |
|---|---|---|
| length | `m`, `km`, `cm`, `mm` | `5 km` |
| stress | `Pa`, `kPa`, `MPa`, `GPa` | `-1.5 MPa` |
| density | `kg/m3`, `kg/m^3`, `kg m-3`, `g/cm3`, `g/cm^3` | `2900 kg/m3` |
| acceleration | `m/s2`, `m/s^2`, `m s-2` | `9.81 m/s2` |
| angle | `deg`, `°`, `rad` | `60 deg` |

`nu` and `friction` are dimensionless: plain numbers (a unit there is an error).
`dies.parse_quantity('30 GPa', 'stress')` gives 3e10.

## Sections

| Key | Required | Content |
|---|---|---|
| `name`, `description` | no | Texts kept in the metadata of the results. |
| `grid` | yes | `x: [min, max]`, `y: [base, top]` (negative downwards, top < 0), `cell: [along x, along y]`. |
| `gravity` | yes | No default, so that Earth (9.81 m/s2) or Mars (3.71 m/s2) is chosen knowingly. |
| `layers` | yes | List, from the top down. Each one: `top`, `E`, `nu`, `density`, `friction`, `T`, and optionally `name`. |
| `cracks` | yes | List with one crack (superposing several is not supported yet). |
| `remote_stresses` | no | `xx`, `yy`: constant remote stresses, tension positive. Default 0. |
| `lithostatic` | no | `true` or `false`. Default `true`. |
| `analysis` | yes | `type: static` or `dynamic`; see below. |
| `output` | no | Results files, figures, format and resolution; see below. |

Unknown keys are errors, with a suggestion for typos (`frictoin` → `friction`). **Every problem
found is reported together**, one per line, with its place in the file (e.g. `layers[0].E: ...`),
in one `ScenarioError` (a subclass of `InputError`).

### Layers

- From the top down; each layer goes from its `top` to the top of the next one, and the last one
  down to the base of the model and below.
- The first layer starts at the surface (`top: 0 m`): the lithostatic load of the core takes the
  uppermost layer up to y = 0, so any other value would be misleading.
- The tops must decrease, and no layer may start at or below the base of the model.
- A node exactly on a boundary belongs to the layer above (as in `model.layer_index`).
- Limits: `E > 0`; `0 <= nu < 0.5`; `density > 0`; `friction > 0`; **`T < 0`** (tension is
  negative on the Mohr diagram; the cohesion is implicit, C = 2|T|).
- They become the inputs of the core: one layer gives numbers and no `layer_tops`; several give
  one list per property and `layer_tops` (`Scenario.model_inputs()`, `Scenario.layer_tops()`),
  converted by `layered_inputs`.

### The crack

- `upper_tip`, `lower_tip`: `[x, y]`, at or below the surface.
- Opening: **either** `aperture` (total aperture, > 0; dikes) **or** `sigma_I` (opening driving
  stress, > 0 opens, < 0 closes), not both: one follows from the other,
  σI = a·G/(b·(1−ν)) (eq. 2 of Rivas-Dorado et al., 2023), with E and ν of the layer at the centre
  of the crack, as in `single_model` (a warning says so if the crack crosses a layer boundary).
- `sigma_II` (slip, > 0 normal sense: hanging wall down-dip) and `sigma_III` (out of the X-Y
  plane): optional, default 0. A fault is given by its driving stresses.

### Analysis

- `type: static`: one evaluation of the model (`model.compute`).
- `type: dynamic`: the driving stresses grow together step by step (`compute_dynamic_model`), with
  either `step` (the driving stress added at each step, applied to the largest of the three) or
  `steps` (their number). With `step`, the number of steps is rounded so that the last one reaches
  the final load exactly (8 MPa with 0.25 MPa → 32 steps; 8 MPa with 0.3 MPa → 27 steps of
  0.296 MPa). A dynamic analysis also gives the static results of the crack at its final load.
- `plane_angle` (optional, 0–90 deg): the angle written in the label of `plane_normal` and
  `plane_shear`. **Note:** in the current code those figures always show the stresses on the
  optimal planes of the friction (`coula`); the angle is only a label. By default it is the
  optimal angle of the layer at the centre of the crack (90° − coula, e.g. 59.84° for μ′ = 0.57).

### Output

| Key | Default | Content |
|---|---|---|
| `results` | `[npz]` | `npz` and/or `csv`. |
| `figures.static` | none | List of figure names (table below), or `all`. |
| `figures.dynamic` | none | `show_every` (default 4: one step in 4 and the last one) and `downscaling` (default 1: one plane trace every n nodes). Only in a dynamic analysis. All steps use the colour range of the final step. |
| `format` | `png` | `png`, `pdf` or `svg`. |
| `dpi` | 200 | Resolution of the figures. |

Static figures (drawn with `plotting.py` from the results, without recomputing):

| Name | Figure |
|---|---|
| `ux`, `uy`, `uxy` | Displacement vectors (on the coarser grid, see below). |
| `ux_contours`, `uy_contours`, `uxy_contours` | Displacement contours. |
| `sxx`, `syy`, `sxy`, `sxz`, `szz`, `sav`, `tmax` | Stress components, tension positive. |
| `s1`, `s3`, `angle` | Principal stresses and the orientation of σ1. |
| `trajectories`, `trajectories_planes` | Orientations of σ1 and σ3, without and with the optimal shear planes (coarser grid). |
| `plane_normal`, `plane_shear` | Normal and shear stress on the optimal shear planes. |
| `crack_displacements` | Profile of the crack walls (opening and slip). |

As in `single_model`, the vector and trajectory figures use a grid 2.5 times coarser along x and 3
times along y than `grid.cell`, so that the arrows can be read (`ScenarioResults.coarse()`).

### Files written by `run.write(folder)`

| File | Content |
|---|---|
| `scenario.yaml` (or `.json`) | The scenario as given (a copy of its file, or the dict as JSON). |
| `static_results.npz` / `.csv` | `DiESResults.save` / `to_csv` of the static model. |
| `dynamic_results.npz` | `DynamicResults.save`, every step. |
| `dynamic_final_step.csv` | The last step, with `failure_step` (step at which each node first failed, 0 if never). |
| `<figure>.<format>` | Each static figure, and `dynamic_step_<step>` for the dynamic ones. |

The metadata of every results file keep the scenario in the units of the core
(`metadata['scenario']`).

## API

| Name | Content |
|---|---|
| `load_scenario(path)` | Reads and checks a `.yaml`, `.yml` or `.json` file; returns a `Scenario`. |
| `Scenario.from_dict(data)` | The same, from a dict. |
| `Scenario.data`, `to_dict()` | The checked scenario, in the units of the core. |
| `model_inputs()`, `layer_tops()`, `crack_inputs()`, `driving_stresses()`, `investigation_range()`, `build_model()`, `build_crack()` | The inputs of the core functions. |
| `Scenario.run()` | Computes it; returns a `ScenarioResults`. |
| `ScenarioResults.static`, `.dynamic`, `.model`, `.crack` | The results and the objects used. |
| `figure(name)`, `figures()`, `write(folder)` | One figure, those of the output section, everything to a folder. |
| `parse_quantity(value, kind)` | One value with its unit, in the units of the core. |
| `ScenarioError` | Invalid scenario; `.problems` lists every problem. |

Reading YAML needs PyYAML (a dependency in `pyproject.toml`).
