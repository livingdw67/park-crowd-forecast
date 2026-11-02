"""Export the public app's data from Databricks into public_app/data/ (committed).

    databricks auth login --host <your workspace URL> --profile free
    .venv\\Scripts\\python public_app\\scripts\\export_data.py

Only derived data leaves the workspace: no scraped pages, ride waits or raw JSON.

- daily.parquet: one row per park and measured day, the daily crowd score plus the model features
  (built with parks_lib.forecast.build_frame, the same code the training job runs)
- monthly.parquet: monthly average crowd score per park, Universal and comparison parks
  (parks_lib.causal.monthly_panel), for the Epic Universe impact study
- forecast.parquet: the champion model's 90-day forecast and queue-times' own forecast

Reads through the first SQL warehouse in the workspace with the CLI profile's credentials.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from databricks import sql
from databricks.sdk import WorkspaceClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "databricks" / "src"))

from parks_lib.causal import monthly_panel  # noqa: E402
from parks_lib.forecast import FEATURE_SETS, build_frame  # noqa: E402

SCHEMA = os.getenv("PARKS_SCHEMA", "workspace.parks")
OUT = Path(__file__).resolve().parents[1] / "data"
QUERIES = {
    "crowd": f"SELECT * FROM {SCHEMA}.raw_crowd_calendar",
    "comparison": f"SELECT park_id, date, crowd_percent, is_forecast, is_closed FROM {SCHEMA}.raw_crowd_calendar_comparison",
    "calendar": f"SELECT * FROM {SCHEMA}.calendar_features",
    "weather": f"SELECT * EXCEPT (is_forecast, fetched_at) FROM {SCHEMA}.raw_weather_daily",
    "forecast": f"""SELECT f.park_id, f.date, f.predicted_crowd, f.queue_times_forecast, c.holiday_name, c.event_hhn
                    FROM {SCHEMA}.crowd_forecast f LEFT JOIN {SCHEMA}.calendar_features c USING (date)""",
}


def fetch():
    w = WorkspaceClient(profile=os.getenv("DATABRICKS_CONFIG_PROFILE", "free"))
    warehouse = next(iter(w.warehouses.list()))
    frames = {}
    with sql.connect(server_hostname=w.config.host.removeprefix("https://"),
                     http_path=f"/sql/1.0/warehouses/{warehouse.id}",
                     credentials_provider=lambda: w.config.authenticate) as conn, conn.cursor() as cur:
        for name, statement in QUERIES.items():
            cur.execute(statement)
            frames[name] = cur.fetchall_arrow().to_pandas()
            print(f"{name}: {len(frames[name])} rows")
    return frames


def daily(crowd, calendar, weather):
    features = FEATURE_SETS["calendar+hours+weather"]
    df = build_frame(crowd, calendar, weather).dropna(subset=["crowd_percent"])
    out = df[["date", "crowd_percent", "holiday_name"] + features].copy()
    out["park_id"] = out["park_id"].astype(int)
    out["date"] = pd.to_datetime(out["date"])
    floats = out.select_dtypes("float").columns
    out[floats] = out[floats].round(2)
    return out.sort_values(["park_id", "date"]).reset_index(drop=True)


def main():
    frames = fetch()
    OUT.mkdir(exist_ok=True)

    days = daily(frames["crowd"], frames["calendar"], frames["weather"])
    days.to_parquet(OUT / "daily.parquet", index=False)

    panel = monthly_panel(pd.concat([frames["crowd"], frames["comparison"]]))
    monthly = (panel.round(2).rename_axis("month").reset_index()
               .melt(id_vars="month", var_name="park_id", value_name="crowd").dropna())
    monthly["park_id"] = monthly["park_id"].astype(int)
    monthly.to_parquet(OUT / "monthly.parquet", index=False)

    forecast = frames["forecast"].assign(date=lambda f: pd.to_datetime(f["date"]))
    forecast["predicted_crowd"] = forecast["predicted_crowd"].astype(float).round(1)
    forecast.to_parquet(OUT / "forecast.parquet", index=False)

    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"daily": len(days), "monthly": len(monthly), "forecast": len(forecast)},
        "daily_range": [str(days["date"].min().date()), str(days["date"].max().date())],
        "measured_days_per_park": {str(k): int(v) for k, v in days.groupby("park_id").size().items()},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
