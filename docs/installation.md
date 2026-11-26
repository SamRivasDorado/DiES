# Using DiES: as scripts or as a package

_Phase 3 (Nov 2026). DiES can be used in two ways, from the same code: the modules at the
root of the repository are the only copy, and the package only adds a layer on top._

## 1. As scripts (as until now)

Open the repository folder in Spyder (or any IDE), so that it is the working directory, and run
`Running_Models.py` or your own scripts. Nothing has to be installed beyond numpy, scipy and
matplotlib. The modules import each other by name (`from Modeling_functions import single_model`),
and Python finds them because they are in the working folder.

## 2. As a package

From the repository folder, preferably inside a virtual environment:

```
pip install -e .
```

`-e` (editable) installs links to the files of the repository instead of copies, so any change
to the code takes effect immediately, without installing again. After that, DiES can be imported
from any folder, either through the root modules, as in the scripts, or through the `dies`
package, which re-exports their public API:

```python
import dies

m = dies.model([-5000, 5000], [-6000, -10], 100, 100, 500,
               30e9, 0.25, 2900, 0.57, -1.5e6, 9.81)
disc = dies.crack((0, -1000), (0, -5000), 8e6, 0, 0)
res = m.compute(disc, 0, 0, True)
fig = dies.plotting.plot_stresses(res, 'sxy')

dyn = dies.compute_dynamic_model(...)
```

`dies` contains no code of its own: `dies.model` is `DiES_core.model`, `dies.plotting` is
`plotting`, and so on (`dies/__init__.py` lists what it exports). Optional extras:
`pip install -e .[test]` (pytest) and `pip install -e .[xarray]` (for `DiESResults.to_xarray()`).

A normal installation (`pip install .`, without `-e`) copies the modules and works the same,
but then changes to the repository need a new installation.

## Notes

- The root modules are installed with their current names (`DiES_core`, `Modeling_functions`,
  `Failure_Determination`, `Helper_functions`, `results`, `plotting`), so that they keep
  importing each other as they do when run as scripts. `results` and `plotting` are generic
  names that another installed package could also use: a virtual environment for DiES avoids
  any clash. If DiES is ever published (e.g. on PyPI), the modules can be moved into `dies/`
  and the root files reduced to one-line imports, without breaking any script.
- New modules of the core must be added to `py-modules` in `pyproject.toml`
  (`tests/test_package.py` checks it).
- The tests run from the repository folder, with or without installing: `pytest tests/`.
