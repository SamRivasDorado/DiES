# Using DiES: as scripts or as a package

_Phase 3 (Nov 2026). DiES can be used in two ways, from the same code: the modules at the
root of the repository are the only copy, and the package only adds a layer on top._

## 1. As scripts (as until now)

Open the repository folder in Spyder (or any IDE), so that it is the working directory, and run
`Running_Models.py` or your own scripts. Nothing has to be installed beyond numpy, scipy and
matplotlib (and PyYAML, only for the scenarios). The modules import each other by name (`from Modeling_functions import single_model`),
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

`dies` re-exports the core without copying it: `dies.model` is `DiES_core.model`,
`dies.plotting` is `plotting`, and so on (`dies/__init__.py` lists what it exports). Its own
modules are `dies/scenario.py`, which reads a complete model from a YAML or JSON file (see
`docs/scenario.md`), and `dies/cli.py`, the command line:

```python
scenario = dies.load_scenario('examples/dike_example.yaml')
run = scenario.run()
run.write('results/')
```

The installation also adds the `dies` command, which does the same from a terminal:

```
dies check examples/dike_example.yaml              # check the scenario and summarise it
dies run examples/dike_example.yaml --out results/ # compute it and write its output
```

Without installing, the same command works from the repository folder as `python -m dies ...`.

Optional extras:
`pip install -e .[test]` (pytest), `pip install -e .[xarray]` (for `DiESResults.to_xarray()`) and
`pip install -e .[app]` (Streamlit, for the app).

A normal installation (`pip install .`, without `-e`) copies the modules and works the same,
but then changes to the repository need a new installation.

## 3. The app

With the `app` extra installed, `dies-app` opens the DiES app in the browser (without installing,
from the repository folder: `streamlit run dies/app/streamlit_app.py`). See `docs/app.md`.

### Step by step on Windows

Without Anaconda and without git (Python from python.org, or the one Spyder uses):

1. **Download the code.** On GitHub, the green *Code* button → *Download ZIP*, or the ZIP of a
   branch: `https://github.com/SamRivasDorado/DiES/archive/refs/heads/<branch>.zip`. Unzip it in
   a local folder, **not in a synchronised one** (iCloud Drive, OneDrive, Dropbox): the virtual
   environment holds thousands of files, and the synchronisation can lock them and break the
   installation.
2. **Open a terminal in that folder**: right click on the folder → *Open in Terminal* (PowerShell),
   or type `cmd` in the address bar of the File Explorer.
3. **Find your Python.** If `py --version` or `python --version` gives an error, or `python` opens
   the Microsoft Store, Python is not on the PATH: use its full path. One way to find it: in the
   console of Spyder, `import sys; print(sys.executable)`. Write that full path in quotes, with
   `&` before it in PowerShell (without it in `cmd`), e.g.:
   ```
   & "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
   ```
   This creates a virtual environment in the folder (`.venv`), apart from your other packages.
4. **Install DiES and the app** into that environment (one or two minutes):
   ```
   .venv\Scripts\python.exe -m pip install -e ".[app]"
   ```
5. **Open the app**:
   ```
   .venv\Scripts\python.exe -m dies.app.launcher
   ```
   A tab opens in the browser (`http://localhost:8501`); the app runs on your computer. To close
   it, press Ctrl + C in the terminal. Next time, only step 5 is needed (in a terminal open in the
   folder).

With a newer ZIP of the same branch, copy its files over the folder (keeping `.venv`); there is
nothing to install again unless `pyproject.toml` changed. Calling `.venv\Scripts\python.exe`
directly avoids activating the environment, which PowerShell may refuse to do.

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
