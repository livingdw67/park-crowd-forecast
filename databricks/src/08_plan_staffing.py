# Databricks notebook source
# MAGIC %md
# MAGIC # 08 Staffing: crew plans for the next 14 days from the crowd forecast
# MAGIC
# MAGIC For each park and day, solves the staffing MILP (`parks_lib/staffing.py`, OR-Tools SCIP) on the
# MAGIC forecast crowd score and the published park hours, then replays that plan at the forecast plus
# MAGIC 15 points (about one typical forecast error) to show what an underestimate would cost.
# MAGIC
# MAGIC Locations and their demand shares are in `seeds/staffing_locations.csv`. Ride shares were
# MAGIC calibrated so the plan's midday waits match observed posted waits on 2026-10-22; the
# MAGIC attendance, wage and wait-cost figures are assumptions (`ASSUMPTIONS`), not data.
# MAGIC
# MAGIC Writes `staffing_plan_summary`, `staffing_plan_shifts` and `staffing_plan_hourly` (latest run).

# COMMAND ----------

# MAGIC %pip install ortools==9.11.4210 --quiet

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
dbutils.widgets.text("days", "14")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
days = int(dbutils.widgets.get("days"))

# COMMAND ----------

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import parks_lib
from parks_lib.staffing import evaluate_plan, plan_staffing

FORECAST_MISS = 15
LATEST_CLOSE = 23
EVENT_NIGHT_DAY_CLOSE = 17  # hours listed past midnight are event nights (Halloween Horror Nights),
                            # when the park closes to day guests around 5pm

locations = pd.read_csv(Path(parks_lib.__file__).parents[2] / "seeds" / "staffing_locations.csv")
today = date.today()
end = today + timedelta(days=days - 1)
forecast = spark.sql(f"""
    SELECT f.park_id, f.date, f.predicted_crowd, c.open_time, c.close_time, coalesce(c.is_closed, false) AS is_closed
    FROM {catalog}.{schema}.crowd_forecast f
    LEFT JOIN {catalog}.{schema}.raw_crowd_calendar c USING (park_id, date)
    WHERE f.date BETWEEN '{today}' AND '{end}'
""").toPandas()
forecast = forecast[forecast["park_id"].isin(locations["park_id"].unique()) & ~forecast["is_closed"]]

def hour(t, default):
    return default if t is None or pd.isna(t) else int(t.split(":")[0])

# COMMAND ----------

summary, shifts, hourly = [], [], []
for r in forecast.sort_values(["date", "park_id"]).itertuples():
    open_hour = hour(r.open_time, 9)
    close_hour = hour(r.close_time, 21)
    if close_hour <= open_hour:
        close_hour = EVENT_NIGHT_DAY_CLOSE
    close_hour = min(close_hour, LATEST_CLOSE)
    locs = locations[locations["park_id"] == r.park_id].to_dict("records")
    plan = plan_staffing(locs, r.predicted_crowd, open_hour, close_hour)
    if plan.status == "infeasible":
        print(f"No feasible plan for park {r.park_id} on {r.date}")
        continue
    miss = evaluate_plan(plan, locs, min(100, r.predicted_crowd + FORECAST_MISS), open_hour, close_hour)
    key = {"park_id": int(r.park_id), "date": r.date}
    summary.append({
        **key, "predicted_crowd": float(r.predicted_crowd), "open_hour": open_hour, "close_hour": close_hour,
        "workers": plan.total_workers, "labor_cost": plan.labor_cost, "wait_guest_hours": plan.wait_guest_hours,
        "worst_wait_minutes": max((h["wait_minutes"] or 0) for h in plan.hourly),
        "skipped_guests": plan.skipped_guests,
        "wait_guest_hours_if_15_busier": miss.wait_guest_hours,
        "worst_wait_minutes_if_15_busier": max((h["wait_minutes"] or 0) for h in miss.hourly),
        "skipped_guests_if_15_busier": miss.skipped_guests,
    })
    shifts += [{**key, **s} for s in plan.shifts]
    hourly += [{**key, **h} for h in plan.hourly]

planned_at = datetime.now(timezone.utc)
for name, rows in [("staffing_plan_summary", summary), ("staffing_plan_shifts", shifts), ("staffing_plan_hourly", hourly)]:
    (spark.createDataFrame(pd.DataFrame(rows).assign(planned_at=planned_at))
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{name}"))
print(f"Planned {len(summary)} park-days, {today} to {end}")
display(spark.table(f"{catalog}.{schema}.staffing_plan_summary").orderBy("date", "park_id"))
