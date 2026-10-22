# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Calendar features: one row per date with holidays, school breaks and events
# MAGIC
# MAGIC Builds `calendar_features` from 2018 to 18 months ahead. Every column is known in advance, so
# MAGIC the forecast can use it for any future date. Event dates come from `seeds/events.csv`, which
# MAGIC cites a source for each row; school breaks are rule-based (see `parks_lib/calendar.py`).
# MAGIC Rebuilt in full on every run, since the whole table is a few thousand rows.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import parks_lib
from parks_lib.calendar import build_calendar_features

# The notebook's working directory isn't its own folder on serverless, so find seeds/ from the package
events = pd.read_csv(Path(parks_lib.__file__).parents[2] / "seeds" / "events.csv", parse_dates=["start_date", "end_date"])
events["start_date"] = events["start_date"].dt.date
events["end_date"] = events["end_date"].dt.date
spark.createDataFrame(events.astype({"park_id": "Int64"})).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{schema}.seed_events")

features = build_calendar_features(date(2018, 1, 1), date.today() + timedelta(days=548), events)
features = features.astype({"days_to_next_holiday": "Int64", "days_since_last_holiday": "Int64", "week_of_year": "int32"})
spark.createDataFrame(features).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{schema}.calendar_features")
print(f"calendar_features: {len(features):,} days, {features['date'].min()} to {features['date'].max()}")
print("Days flagged:", {c: int(features[c].sum()) for c in features.columns if features[c].dtype == bool})
