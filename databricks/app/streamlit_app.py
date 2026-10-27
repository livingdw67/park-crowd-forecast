"""Park crowd forecast app (Databricks Apps): crowd calendar, staffing plans, what-if scenarios and
the Epic Universe impact study.

Runs from the bundle root (databricks/), so the theme in .streamlit/ applies and the app shares
parks_lib and the seeds with the jobs. Locally: see "App" in the README.
"""

import os
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SRC = str(APP_DIR.parent / "src")  # parks_lib, the optimizer the daily job plans with
if SRC not in sys.path:
    sys.path.insert(0, SRC)

st.set_page_config(page_title="Park crowd forecast", page_icon=str(APP_DIR / "assets" / "favicon.png"), layout="wide")

page = st.navigation([
    st.Page("app_pages/crowd_calendar.py", title="Crowd calendar", icon=":material/calendar_month:"),
    st.Page("app_pages/staffing_plan.py", title="Staffing plan", icon=":material/groups:"),
    st.Page("app_pages/what_if.py", title="What if", icon=":material/tune:"),
    st.Page("app_pages/epic_impact.py", title="Epic Universe impact", icon=":material/science:"),
], position="top")

st.title("Universal Orlando crowd forecast")
# The brand's short amber rule under the headline
st.markdown('<div style="width:56px;height:4px;background:#f0a742;border-radius:2px;margin:-0.5rem 0 1rem"></div>',
            unsafe_allow_html=True)

if not os.getenv("DATABRICKS_WAREHOUSE_ID"):
    st.error("Set DATABRICKS_WAREHOUSE_ID to the SQL warehouse to query. On Databricks Apps, app.yaml sets it "
             "from the app's sql-warehouse resource.", icon=":material/error:")
    st.stop()

page.run()

st.divider()
st.caption("Wait times and crowd scores: [Powered by Queue-Times.com](https://queue-times.com/). "
           "Weather: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0). Staffing plans rest on assumed attendance, "
           "wages and wait costs, not measured data.")
