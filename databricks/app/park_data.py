"""Data for the app: Unity Catalog tables through the SQL warehouse, plus the staffing seed.

On Databricks Apps the app's service principal signs in from the environment the platform sets;
locally, set DATABRICKS_CONFIG_PROFILE (e.g. `free`). DATABRICKS_WAREHOUSE_ID names the warehouse
(app.yaml fills it from the app's sql-warehouse resource).
"""

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = os.getenv("PARKS_SCHEMA", "workspace.parks")
EASTERN = ZoneInfo("America/New_York")
PARKS = {65: "Universal Studios Florida", 64: "Islands of Adventure", 334: "Epic Universe", 67: "Volcano Bay"}
PARK_IDS = {name: park_id for park_id, name in PARKS.items()}


def today():
    """The parks' local date (app servers run on UTC)."""
    return datetime.now(EASTERN).date()


def query(*statements):
    """Run statements on one warehouse connection; one DataFrame per statement."""
    cfg = Config()
    http_path = f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}"
    with sql.connect(server_hostname=cfg.host.removeprefix("https://"), http_path=http_path,
                     credentials_provider=lambda: cfg.authenticate) as conn, conn.cursor() as cur:
        frames = []
        for statement in statements:
            cur.execute(statement)
            frames.append(cur.fetchall_arrow().to_pandas())
    return frames


@st.cache_data(ttl="1h", show_spinner="Loading forecasts and plans...")
def load():
    """Everything the pages show, in one round trip. The jobs refresh it daily (6:30am ET)."""
    forecast, measured, summary, shifts, hourly = query(
        f"""SELECT f.park_id, f.date, f.predicted_crowd, f.queue_times_forecast, f.scored_at,
                   c.holiday_name, c.event_hhn
            FROM {SCHEMA}.crowd_forecast f LEFT JOIN {SCHEMA}.calendar_features c USING (date)""",
        f"""SELECT park_id, date, crowd_percent AS measured_crowd FROM {SCHEMA}.raw_crowd_calendar
            WHERE NOT is_forecast AND crowd_percent IS NOT NULL AND date >= date_sub(current_date(), 90)""",
        f"SELECT * FROM {SCHEMA}.staffing_plan_summary",
        f"SELECT * FROM {SCHEMA}.staffing_plan_shifts",
        f"SELECT * FROM {SCHEMA}.staffing_plan_hourly",
    )
    frames = {"forecast": forecast, "measured": measured, "summary": summary, "shifts": shifts, "hourly": hourly}
    for frame in frames.values():
        frame["date"] = pd.to_datetime(frame["date"])
    return frames


@st.cache_data
def load_locations():
    """Rides and food outlets the optimizer staffs: the same seed the daily-features job plans from."""
    return pd.read_csv(BUNDLE_ROOT / "seeds" / "staffing_locations.csv")
