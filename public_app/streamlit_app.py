"""Public demo of the park crowd forecast: a model lab, a crowd explorer and the Epic Universe study.

Runs on a frozen snapshot (data/, written by scripts/export_data.py) with the same parks_lib code as
the Databricks jobs, so clicks cost nothing: no warehouse, no model API.

    .venv\\Scripts\\python -m pip install -r public_app\\requirements.txt
    .venv\\Scripts\\streamlit run public_app\\streamlit_app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
CODE_DIRS = (APP_DIR, APP_DIR.parent / "databricks")   # lab/viz/cached, parks_lib and the shared charts


def _our_modules():
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and any(Path(path).resolve().is_relative_to(d) for d in CODE_DIRS):
            yield name, module, path


def drop_stale_modules():
    """Community Cloud pulls new code on a merge but keeps imported modules in memory, so a new page could
    run against an old lab.py. If any of this app's modules changed on disk since it was imported, forget
    them all (together, so none keeps a reference to an old one) and let this run import them fresh."""
    ours = list(_our_modules())
    if any(m.__dict__.get("_imported_mtime", os.path.getmtime(p)) != os.path.getmtime(p) for _, m, p in ours):
        for name, _, _ in ours:
            del sys.modules[name]


def stamp_modules():
    for _, module, path in _our_modules():
        module.__dict__.setdefault("_imported_mtime", os.path.getmtime(path))


drop_stale_modules()
import cached  # noqa: E402  (after the stale check)

FAVICON = APP_DIR.parent / "databricks" / "app" / "assets" / "favicon.png"

st.set_page_config(page_title="Park crowd forecast", page_icon=str(FAVICON), layout="wide")

page = st.navigation([
    st.Page("app_pages/model_lab.py", title="Model lab", icon=":material/tune:"),
    st.Page("app_pages/crowd_explorer.py", title="Crowd explorer", icon=":material/calendar_month:"),
    st.Page("app_pages/epic_impact.py", title="Epic Universe impact", icon=":material/science:"),
], position="top")

st.title("Universal Orlando crowd forecast")
# The brand's short amber rule under the headline
st.markdown('<div style="width:56px;height:4px;background:#f0a742;border-radius:2px;margin:-0.5rem 0 1rem"></div>',
            unsafe_allow_html=True)

page.run()
stamp_modules()   # pages import more of our modules (viz, charts, transforms)

st.divider()
meta = cached.meta()
st.caption(f"Data snapshot exported {meta['exported_at'][:10]}: daily crowd scores {meta['daily_range'][0]} to "
           f"{meta['daily_range'][1]}. Crowd scores: [Powered by Queue-Times.com](https://queue-times.com/). "
           "Weather: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0). Everything runs in this app; nothing is "
           "fetched or billed when you click. "
           "[Code on GitHub](https://github.com/livingdw67/park-crowd-forecast)")
