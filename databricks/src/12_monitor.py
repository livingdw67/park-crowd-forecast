# Databricks notebook source
# MAGIC %md
# MAGIC # 12 Monitor: live accuracy, drift and the retune triggers
# MAGIC
# MAGIC Runs daily at the end of `daily-features`, even when an earlier task failed. Steps:
# MAGIC 1. Score every past forecast in `crowd_forecast_history` against the measured crowd.
# MAGIC 2. Compare the 28-day error with the band from the replay (11), and measure drift by month.
# MAGIC 3. Decide whether to start `tune-forecast`.
# MAGIC
# MAGIC Writes `monitor_accuracy` and `monitor_drift` (rebuilt daily) and appends the decision to `retune_log`.
# MAGIC With `auto_retune` false it logs what it would have done and starts nothing. Rules and reasoning are in
# MAGIC `parks_lib/monitoring.py`; `ops/replay_triggers.py` replays them on 2019-2026.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
dbutils.widgets.text("tune_job_id", "")
dbutils.widgets.text("auto_retune", "true")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
tune_job_id = dbutils.widgets.get("tune_job_id")
auto_retune = dbutils.widgets.get("auto_retune").lower() == "true"

# COMMAND ----------

from datetime import datetime, timezone

import pandas as pd

from parks_lib import monitoring as mon


def read(name):
    full = f"{catalog}.{schema}.{name}"
    return spark.table(full).toPandas() if spark.catalog.tableExists(full) else None


history = read("crowd_forecast_history")
measured = spark.sql(f"""SELECT park_id, date, crowd_percent FROM {catalog}.{schema}.raw_crowd_calendar
                         WHERE NOT is_forecast AND NOT is_closed AND crowd_percent IS NOT NULL""").toPandas()
replay = read("forecast_replay")
if replay is None:
    print("No forecast_replay yet (train-forecast writes it weekly): no band or error drift until it exists.")
    replay = pd.DataFrame(columns=["park_id", "date", "trained_through", "horizon", "bucket", "actual", "pred"])
replay["date"] = pd.to_datetime(replay["date"])
settings = read("forecast_settings")
log = read("retune_log")

live = mon.live_errors(history, measured)
accuracy = mon.accuracy_table(live, replay)
drift = pd.concat([mon.error_drift(replay, live) if len(replay) else None, mon.crowd_drift(measured)],
                  ignore_index=True)
print(f"{len(live):,} measured forecasts, {live['date'].nunique()} days; {len(drift)} month-lines of drift")

# COMMAND ----------

now = datetime.now(timezone.utc)
last_run = settings.sort_values("run_at").iloc[-1].to_dict() if settings is not None and len(settings) else None
last_tuned = last_run["run_at"] if last_run else None
started = [] if log is None else list(log.loc[log["action"] == "start", "checked_at"])
last_started = max([t for t in [last_tuned, *started] if t is not None and not pd.isna(t)], key=mon.as_day, default=None)

triggers = [mon.error_trigger(accuracy, since=last_tuned), mon.drift_trigger(drift, since=last_tuned),
            mon.backstop_trigger(last_tuned, now)]
action, reason = mon.decide(triggers, now, last_started, last_run)
run_id = None
if action == "start":
    if auto_retune and tune_job_id:
        from databricks.sdk import WorkspaceClient

        run_id = WorkspaceClient().jobs.run_now(job_id=int(tune_job_id),
                                                job_parameters={"trigger": mon.first_trigger(triggers)}).run_id
    else:
        action = "start (dry run)"
for t in triggers:
    print(f"{t.name:9} {'FIRED' if t.fired else 'quiet':6} {t.detail}")
print(f"Decision: {action}. {reason}" + (f" (tune-forecast run {run_id})" if run_id else ""))

# COMMAND ----------

ACCURACY = ("bucket STRING, date DATE, forecasts BIGINT, live_mae DOUBLE, queue_times_mae DOUBLE, "
            "band_low DOUBLE, band_high DOUBLE, computed_at TIMESTAMP")
DRIFT = "month DATE, line STRING, source STRING, n BIGINT, psi DOUBLE, threshold DOUBLE, computed_at TIMESTAMP"
LOG = ("checked_at TIMESTAMP, error_fired BOOLEAN, error_detail STRING, drift_fired BOOLEAN, drift_detail STRING, "
       "backstop_fired BOOLEAN, backstop_detail STRING, action STRING, reason STRING, tune_run_id BIGINT, "
       "auto_retune BOOLEAN")


def save(frame, schema_ddl, name, mode="overwrite"):
    columns = [c.split()[0] for c in schema_ddl.split(", ")]
    frame = frame.reindex(columns=columns)   # schema order; an empty frame gets the right columns
    writer = spark.createDataFrame(frame, schema=schema_ddl).write.mode(mode)
    writer = writer.option("overwriteSchema", "true") if mode == "overwrite" else writer.option("mergeSchema", "true")
    writer.saveAsTable(f"{catalog}.{schema}.{name}")


save(accuracy.assign(date=pd.to_datetime(accuracy["date"]).dt.date, forecasts=accuracy["forecasts"].astype("int64"),
                     computed_at=now), ACCURACY, "monitor_accuracy")
save(drift.assign(month=pd.to_datetime(drift["month"]).dt.date, n=drift["n"].astype("int64"), computed_at=now)
     if len(drift) else pd.DataFrame(), DRIFT, "monitor_drift")
by_name = {t.name: t for t in triggers}
save(pd.DataFrame([{
    "checked_at": now, **{f"{n}_{k}": getattr(by_name[n], "fired" if k == "fired" else "detail")
                          for n in ("error", "drift", "backstop") for k in ("fired", "detail")},
    "action": action, "reason": reason, "tune_run_id": run_id, "auto_retune": auto_retune,
}]), LOG, "retune_log", mode="append")
display(accuracy[accuracy["bucket"] == mon.TRIGGER_HORIZON].tail(14))
