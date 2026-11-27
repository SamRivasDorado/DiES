"""
Style of the DiES app: a few CSS rules on top of Streamlit's theme (Sam's comments of
27 Nov 2026). The colours are translucent, so that they work in the light and the
dark themes.

- Number fields: the −, + and × (clear) buttons are faint until the pointer is over
  the field.
- Backgrounds: the model setup (st.container with key 'setup') bluish, the box of the
  models neutral grey, so that they are told apart.
- The box of the models (st.container with key 'models'): its two tabs, Single model
  and Dynamic model, are larger and drawn as tabs with a background; the tabs inside
  them (Model, Displacements...) keep Streamlit's look.

The selectors use Streamlit's data-testid attributes (checked with Streamlit 1.67);
if a later version renames them, the app still works, only with Streamlit's own look.
"""

import streamlit as st

CSS = """
<style>
/* Number fields: faint -, + and x (clear) buttons, clearer when the pointer is over the field */
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputClearButton"] {
    opacity: 0.25;
    background: transparent;
}
[data-testid="stNumberInput"]:hover [data-testid="stNumberInputStepDown"],
[data-testid="stNumberInput"]:hover [data-testid="stNumberInputStepUp"],
[data-testid="stNumberInput"]:hover [data-testid="stNumberInputClearButton"] {
    opacity: 0.75;
}

/* Backgrounds of the two boxes, slightly different so that they are told apart (Sam,
   27 Nov 2026): the model setup bluish, the box of the models neutral grey */
.st-key-setup [data-testid="stExpander"] details {
    background: rgba(70, 130, 220, 0.09);
}
.st-key-models {
    background: rgba(128, 128, 128, 0.07);
}

/* Main tabs of the box of the models: larger, bold, each one on its own background */
.st-key-models [data-testid="stTab"] {
    padding: 0.45rem 1.2rem;
    margin-right: 0.35rem;
    border-radius: 0.5rem 0.5rem 0 0;
    background: rgba(128, 128, 128, 0.12);
}
.st-key-models [data-testid="stTab"] p {
    font-size: 1.15rem;
    font-weight: 600;
}
.st-key-models [data-testid="stTab"][aria-selected="true"] {
    background: rgba(255, 75, 75, 0.16);
}

/* Tabs inside a tab (the views of each model): back to Streamlit's look */
.st-key-models [role="tabpanel"] [data-testid="stTab"] {
    padding: revert;
    margin-right: revert;
    border-radius: 0;
    background: transparent;
}
.st-key-models [role="tabpanel"] [data-testid="stTab"] p {
    font-size: revert;
    font-weight: revert;
}
</style>
"""


def apply():
    """Adds the style rules to the page (once per run of the script)."""
    st.markdown(CSS, unsafe_allow_html=True)
