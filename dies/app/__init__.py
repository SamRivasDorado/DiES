"""
The DiES app (Phase 4, Nov 2026): a Streamlit interface on top of the dies package.

The app does not touch the core: the forms build a scenario (dies.Scenario.from_dict),
the problems of ScenarioError.problems are shown next to their fields, and the results
and figures come from ScenarioResults, as with `dies run`.

Modules:
    defaults.py         the scenario the app opens with (the dike of examples/)
    state.py            forms <-> scenario dict, problems -> fields, cache key (no Streamlit)
    figures.py          figures of the app itself, e.g. the sketch of the model (no Streamlit)
    session.py          widget keys and session state (Streamlit)
    compute.py          cached runs (Streamlit)
    ui/                 the sidebar, the model setup and the tabs (Streamlit)
    streamlit_app.py    the entry point: `streamlit run dies/app/streamlit_app.py`
    launcher.py         the `dies-app` command

Streamlit is an optional dependency: `pip install -e .[app]`.
"""
