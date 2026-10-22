# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Score: next 90 days of crowd forecasts per park
# MAGIC
# MAGIC Loads the `champion` model and scores every park for today through 90 days out. Writes
# MAGIC `crowd_forecast` (latest run only) and appends to `crowd_forecast_history`, so forecasts can
# MAGIC later be compared with what was measured. queue-times' own forecast is carried alongside where
# MAGIC they publish one. Days the calendar already marks closed are left out.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 --quiet

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
dbutils.widgets.text("horizon_days", "90")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
horizon_days = int(dbutils.widgets.get("horizon_days"))

# COMMAND ----------

from datetime import date, datetime, timedelta, timezone

import mlflow
import pandas as pd

from parks_lib.forecast import build_frame

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{schema}.crowd_forecast"
version = mlflow.MlflowClient().get_model_version_by_alias(model_name, "champion").version
model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")

today = date.today()
dates = [today + timedelta(days=i) for i in range(horizon_days + 1)]
parks = [65, 64, 334, 67]

# Future days the crowd calendar already covers bring their hours, closures and queue-times' forecast
crowd = spark.table(f"{catalog}.{schema}.raw_crowd_calendar").where(f"date >= '{today}'").toPandas()
skeleton = pd.DataFrame([(p, d) for p in parks for d in dates], columns=["park_id", "date"])
future = skeleton.merge(crowd, on=["park_id", "date"], how="left")
future["queue_times_forecast"] = future["crowd_percent"].where(future["is_forecast"].fillna(False).astype(bool))
future["is_closed"] = future["is_closed"].fillna(False)
future["is_forecast"] = True

calendar = spark.table(f"{catalog}.{schema}.calendar_features").toPandas()
weather = spark.table(f"{catalog}.{schema}.raw_weather_daily").drop("is_forecast", "fetched_at").toPandas()
frame = build_frame(future, calendar, weather)

# COMMAND ----------

inputs = model.metadata.get_input_schema().input_names()
frame["predicted_crowd"] = model.predict(frame[inputs].astype({"park_id": int}))
out = frame[["park_id", "date", "predicted_crowd", "queue_times_forecast"]].copy()
out["park_id"] = out["park_id"].astype(int)
out["model_version"] = int(version)
out["scored_at"] = datetime.now(timezone.utc)

result = spark.createDataFrame(out)
result.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.crowd_forecast")
result.write.mode("append").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.crowd_forecast_history")
print(f"Scored {len(out):,} park-days with model version {version}, {today} to {dates[-1]}")
display(result.where(f"date <= '{today + timedelta(days=13)}'").orderBy("date", "park_id"))
