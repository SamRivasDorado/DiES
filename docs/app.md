# The DiES app

_Phase 4 (Nov 2026), in progress. Code: `dies/app/`. Tests: `tests/test_app_state.py` and
`tests/test_app_ui.py`._

The app is a Streamlit interface on top of the `dies` package. It does not touch the core: the
forms build a scenario (`dies.Scenario.from_dict`, the same format as the YAML files of
`docs/scenario.md`), the problems of `ScenarioError.problems` are shown next to their fields, and
the results come from `Scenario.run()`, as with `dies run`.

## Starting it

Streamlit is an optional dependency:

```
pip install -e .[app]        # once, from the repository folder
dies-app                     # opens the app in the browser
```

Without installing, from the repository folder: `streamlit run dies/app/streamlit_app.py`.
Step by step on Windows (without Anaconda or git, Python off the PATH): `docs/installation.md`.
Extra arguments go to Streamlit, e.g. `dies-app --server.port 8502`.

## Layout

- **Sidebar: the definitions common to a project.**
  - Scenario: open a `.yaml`, `.yml` or `.json` file, save the current one as YAML (with the
    analysis chosen, static or dynamic), or go back to the example. The app opens with the dike of
    `examples/dike_example.yaml`. The problems of a file opened are listed under it, and the values
    that cannot be read are left empty, so that a file with problems can be fixed in the app.
  - Grid (m): x limits, y base and top (the top below the surface, y < 0), cell size.
  - Gravity (m/s²): any value (9.81 Earth, 3.71 Mars, 1.62 Moon, 0...).
  - Loading: remote stresses (MPa, tension positive) and the lithostatic stresses.
- **Model setup** (above the tabs): the layers (from the top down, the first one from the surface;
  rows can be added and removed) and the crack (aperture **or** σI, plus σII and σIII), as editable
  tables. In the crack table, 0 (or an empty cell) in the aperture or in σI means "not given": give
  one of them and leave 0 in the other; with both at 0 the crack has no opening (e.g. a fault with
  shear only). The cohesion is implicit, C = 2|T|, and is shown under the layers.
- **The models**, in a box below the model setup, with two tabs:
- **Single model** tab: one evaluation of the model, with its Run button, and its views:
  - **Model**: sketch of the model and profile of the crack walls, without computing;
  - **Displacements** (ux, uy or |u|, as arrows or colour maps), **Stresses** (σxx, σyy, σxy, σxz,
    σzz, mean stress, maximum shear), **Principal & trajectories** (σ1, σ3, orientation of σ1,
    trajectories with or without the optimal shear planes) and **Planes & failure** (σn and τ on
    the optimal shear planes, with the angle α of their label; the failure state of each node,
    with the share of each failure mode among the nodes that fail, and r/r_c, how close each node
    is to failure). Each view, from top to bottom (layout of Sam's sketch, 27 Nov 2026):
    - a selector of its figures and the **drawing options** (colour map, colours centred on 0,
      manual colour limits, contours and their labels, manual contour step, text box, arrow length
      and **arrow density**, one arrow every n nodes; those a figure does not take are disabled);
    - the figure, with its PNG, PDF and SVG downloads;
    - the **probes**, in one table with a row per probe (none at first): x, y and *On the map* are
      edited in the table, and the node used and its displacements, stresses, principal stresses
      and orientation of σ1, σn and τ, r/r_c and failure state follow; rows are added with *Add
      probe* (or the + of the table) and removed by selecting them and deleting them in the table;
    - the **profiles**, added with *Add profile* (none at first; the first one runs along the top
      row of nodes, the surface profile; the second one through the centre of the crack), two per
      row: A and B, the plot of the fields of the figure along the line, its CSV, and a button back
      to the surface profile; the **profile plot options** (major and minor grids, axis labels,
      sampled points, line width) apply to all of them.
    The probes and profiles are shared by the views (moved in one, moved in all), each with its
    colour and label on the map (P1, P2…; A1–B1, A2–B2…), kept inside the map.
  - **Export**: a ZIP with what `dies run` writes: `scenario.yaml`, the results files (npz, csv)
    and the figures chosen, drawn with their default options.
- **Dynamic model** tab: the driving stresses grow step by step (the stress added at each step, or
  the number of steps), with its Run button and a summary of the last step: the nodes of each
  failure state and their share of the nodes that fail (those below failure are counted apart).

Units in the forms: m, MPa, GPa (E), kg/m³, m/s². The saved YAML writes every value with its unit.

## How it works

- At every interaction Streamlit runs `dies/app/streamlit_app.py` again from the top. The widgets
  are read into a form (`dies/app/session.py`); the form gives two scenarios, static and dynamic
  (`dies/app/state.py`), which are checked at once. Each problem is shown next to its field, in the
  words of the forms (e.g. `Layer 2, E (GPa): must be larger than 0`); a Run button is disabled
  while its scenario has problems.
- Nothing is computed until Run is pressed. Each computation is cached with the scenario as its
  key (`dies/app/compute.py`): the checked scenario in the units of the core, without its name,
  description, output section and plane angle (only a label), so that `5 km` and `5000 m` give
  the same key. The cache keeps
  the last 4 scenarios and gives back the same results object (`st.cache_resource`), without
  copying it at every interaction. When the inputs change after a run, the tab says that the
  results shown are those of the previous inputs.
- Figures are drawn only for the view that is open, cached as images, and can be downloaded as
  PNG, PDF or SVG. pyplot is not safe when several sessions draw at the same time, so the app draws
  one figure at a time (`dies/app/figures.py`, `DRAW_LOCK`).
- Warnings of the core (e.g. a crack crossing a layer boundary) are shown in the tab.

## Modules

| Module | Content | Streamlit |
|---|---|---|
| `dies/app/defaults.py` | The scenario the app opens with (the dike example; `examples/` is not installed with the package). | no |
| `dies/app/state.py` | Forms ⇄ scenario dict, display units, problems → sections and readable messages, cache key, reading and writing YAML/JSON. | no |
| `dies/app/figures.py` | Figures of the app (`plot_model`, `plot_crack`, `plot_profile` with its options), the marks of the probes and profiles (`add_overlays`, labels kept inside the map) and `render()` to image bytes. | no |
| `dies/app/views.py` | The figures of each view of the single model: how they are drawn (with `plotting.py`), their drawing options and the fields of their profile; the arrow density (`subsample`), the probe table, the default profile and the optimal plane angle. | no |
| `dies/app/export.py` | The ZIP of the Export view, written by `ScenarioResults.write()` from the results of the run. | no |
| `dies/app/session.py` | Widget keys (with a version number) and session state. | yes |
| `dies/app/compute.py` | Cached runs and figures. | yes |
| `dies/app/ui/` | Sidebar, model setup, Single model and Dynamic model tabs; `result_view.py`: one view of the results (selector, drawing options, figure, probes, profiles); `style.py`: a few CSS rules (faint −, + and × buttons in the number fields, larger main tabs, different backgrounds for the model setup and the box of the models). | yes |
| `dies/app/streamlit_app.py` | Entry point. | yes |
| `dies/app/launcher.py` | The `dies-app` command. | yes |

The modules without Streamlit hold the logic of the app and are tested with pytest; another
interface (e.g. Shiny) could reuse them.

## Status

| Step | Content | Status |
|---|---|---|
| 1 | Skeleton: sidebar, model setup, the two tabs with Run and cache, Model view, `dies-app`. | done |
| 2 | Views of the single model (displacements, stresses, principal stresses and trajectories, planes and failure) with drawing options, probe and profile; failure state of the single model; export. | done |
| 3 | Views of the dynamic model: steps, failure-mode map, step of first failure. | next |
| 4 | Export of the dynamic model, deployment. | |
