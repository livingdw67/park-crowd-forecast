# Databricks notebook source
# MAGIC %md
# MAGIC # 11 Replay: production-like forecasts that set the monitor's expected error
# MAGIC
# MAGIC The monitor (12) compares live accuracy with what the model should manage. That expectation comes
# MAGIC from this replay, not the yearly backtest, because a model fit once a year is staler than production:
# MAGIC every Monday over the last three-plus years, fit production's settings on every earlier day and
# MAGIC forecast the next 90 days, the way production does. Runs weekly after `train-forecast` retrains
# MAGIC (about 170 fits) and overwrites `forecast_replay`. Why and how: `parks_lib/monitoring.py`.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 --quiet

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from datetime import date, datetime, timedelta, timezone

from parks_lib.forecast import build_frame
from parks_lib.monitoring import REFERENCE_DAYS, REFERENCE_GAP_DAYS, WINDOW_DAYS, walk_forward
from parks_lib.tuning import current_settings

crowd = spark.table(f"{catalog}.{schema}.raw_crowd_calendar").toPandas()
calendar = spark.table(f"{catalog}.{schema}.calendar_features").toPandas()
weather = spark.table(f"{catalog}.{schema}.raw_weather_daily").drop("is_forecast", "fetched_at").toPandas()
df = build_frame(crowd, calendar, weather)

settings_table = f"{catalog}.{schema}.forecast_settings"
history = spark.table(settings_table).toPandas() if spark.catalog.tableExists(settings_table) else None
params, features, source = current_settings(history)

# Enough replay for a band and a drift reference over the past year: each month's reference period (two
# years, ending 90 days before it) plus one rolling window.
start = date.today() - timedelta(days=365 + REFERENCE_DAYS + REFERENCE_GAP_DAYS + WINDOW_DAYS)
replay = walk_forward(df, features, params, start=start)
print(f"{replay['trained_through'].nunique()} weekly fits from {start} with {source}: {len(replay):,} forecasts")

# COMMAND ----------

out = replay.assign(date=replay["date"].dt.date, trained_through=replay["trained_through"].dt.date,
                    settings=source, replayed_at=datetime.now(timezone.utc))
(spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(f"{catalog}.{schema}.forecast_replay"))
display(out.assign(error=(out["actual"] - out["pred"]).abs()).groupby("bucket")["error"].describe().reset_index())
