"""
DiES (Discontinuities in Elastic Space) as a package (Phase 3, Nov 2026).

This package is a facade: it re-exports the public API of the modules at the root of
the repository (DiES_core.py, Modeling_functions.py, Failure_Determination.py,
results.py, plotting.py), which stay as they are and can still be run as scripts.
Nothing is copied: dies.model is DiES_core.model, and so on. The only module of its own
is dies/scenario.py, which reads a model described in a YAML or JSON file (Phase 3):

    scenario = dies.load_scenario('examples/dike_example.yaml')
    run = scenario.run() ; run.write('results/')

    import dies
    m = dies.model([-5000, 5000], [-6000, -10], 100, 100, 500, 30e9, 0.25, 2900, 0.57, -1.5e6, 9.81)
    disc = dies.crack((0, -1000), (0, -5000), 8e6, 0, 0)
    res = m.compute(disc, 0, 0, True)
    fig = dies.plotting.plot_stresses(res, 'sxy')

It needs the root modules to be importable: either `pip install -e .` from the
repository (see pyproject.toml), or running from the repository folder.
"""

# Version, from the installed metadata (pyproject.toml); 'unknown' when DiES is used
# from the repository folder without being installed
try:
    from importlib.metadata import version as _version, PackageNotFoundError as _NotFound
    try:
        __version__ = _version('dies')
    except _NotFound:
        __version__ = 'unknown'
except ImportError:
    __version__ = 'unknown'

# Model geometry and analytical solutions
from DiES_core import crack, model, InputError, displacements, stresses

# Models: single figures, dynamic models (computation, and the legacy wrapper that also draws)
from Modeling_functions import single_model, dynamic_model, compute_dynamic_model, compute_failure_state

# Failure criterion (analytical Griffith-Coulomb tangency since Phase 2) and failure planes
from Failure_Determination import (evaluate_stress_state, evaluate_stress_states, failure_contact,
                                   failure_plane_orientation, FAILURE_TOLERANCE)

# Results of a model and of a dynamic model
from results import DiESResults, DynamicResults

# Figures, as a module: dies.plotting.plot_stresses(...), dies.plotting.plot_dynamic_step(...)
import plotting

# Scenarios: a complete model in one YAML/JSON file (Phase 3; the only code of the package itself)
from dies.scenario import load_scenario, Scenario, ScenarioResults, ScenarioError, parse_quantity

__all__ = ['crack', 'model', 'InputError', 'displacements', 'stresses',
           'single_model', 'dynamic_model', 'compute_dynamic_model', 'compute_failure_state',
           'evaluate_stress_state', 'evaluate_stress_states', 'failure_contact',
           'failure_plane_orientation', 'FAILURE_TOLERANCE',
           'DiESResults', 'DynamicResults', 'plotting',
           'load_scenario', 'Scenario', 'ScenarioResults', 'ScenarioError', 'parse_quantity', '__version__']
