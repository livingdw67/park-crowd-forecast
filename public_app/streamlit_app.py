"""Public demo of the park crowd forecast: a model lab, a crowd explorer and the Epic Universe study.

Runs on a frozen snapshot (data/, written by scripts/export_data.py) with the same parks_lib code as
the Databricks jobs, so clicks cost nothing: no warehouse, no model API.

    .venv\\Scripts\\python -m pip install -r public_app\\requirements.txt
    .venv\\Scripts\\streamlit run public_app\\streamlit_app.py
"""

from pathlib import Path

import streamlit as st

import cached

APP_DIR = Path(__file__).resolve().parent
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

st.divider()
meta = cached.meta()
st.caption(f"Data snapshot exported {meta['exported_at'][:10]}: daily crowd scores {meta['daily_range'][0]} to "
           f"{meta['daily_range'][1]}. Crowd scores: [Powered by Queue-Times.com](https://queue-times.com/). "
           "Weather: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0). Everything runs in this app; nothing is "
           "fetched or billed when you click. "
           "[Code on GitHub](https://github.com/livingdw67/park-crowd-forecast)")
