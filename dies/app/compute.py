"""
Cached computations of the DiES app (Phase 4, Nov 2026).

A scenario is computed once for each key (dies.app.state.scenario_key: the checked
scenario in the units of the core, without its name, description and output); the
same inputs give the results back from the cache at once.

st.cache_resource is used, not st.cache_data: cache_data would copy the results
(pickling them) every time the app reads them, i.e. at every interaction, and the
results of a dynamic analysis are large (38 MB for the dike example: 6000 nodes,
32 steps). cache_resource gives back the same object, which the app only reads.
The cache keeps the last few scenarios (MAX_RUNS) and is shared by the sessions of
the same server, which is fine: the same scenario always gives the same results.
"""

import time
import warnings

import streamlit as st

from dies.scenario import Scenario

# Scenarios kept in the cache (each one with its static and, if any, dynamic results)
MAX_RUNS = 4


@st.cache_resource(max_entries=MAX_RUNS, show_spinner=False)
def _run_cached(key, _raw):
    """
    Computes the scenario _raw. key identifies it in the cache (the argument _raw is not
    hashed by Streamlit, because its name starts with '_'). Returns the ScenarioResults,
    the time taken (s) and the warnings of the core (e.g. a crack crossing a layer boundary).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        results = Scenario.from_dict(_raw).run()
        elapsed = time.perf_counter() - start
    messages = list(dict.fromkeys(str(w.message) for w in caught))
    return results, elapsed, messages


def run(raw, key):
    """
    Results of the scenario raw (from the cache if it was computed before), as a dict:
    key, results (ScenarioResults), elapsed (s, of the computation itself), warnings,
    and cached (True if they came from the cache).
    """
    start = time.perf_counter()
    results, elapsed, messages = _run_cached(key, raw)
    # A call much shorter than the computation means that the results came from the cache
    cached = (time.perf_counter() - start) < 0.5*elapsed
    return {'key': key, 'results': results, 'elapsed': elapsed, 'warnings': messages, 'cached': cached}


@st.cache_data(max_entries=64, show_spinner=False)
def figure_bytes(name, key, label, fmt, dpi, _draw):
    """
    Bytes of a figure, cached by (name, key, label, fmt, dpi). _draw is the function
    that draws it (not hashed); key is the scenario key of what it shows, and label any
    other text it shows that is not in the key (e.g. the scenario name in the title).
    """
    from dies.app.figures import render
    return render(_draw, fmt, dpi)
