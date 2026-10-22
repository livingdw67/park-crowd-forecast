# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Train: daily crowd forecast with an out-of-time backtest
# MAGIC
# MAGIC Backtests each test year (train on every earlier day, score that year) for two baselines and
# MAGIC three LightGBM feature sets, logging every result to MLflow. Then trains the production model on
# MAGIC all measured days and registers it in Unity Catalog with the `champion` alias.
# MAGIC
# MAGIC The production feature set is calendar-only: in the backtest, park hours and weather added
# MAGIC nothing measurable, and calendar features exist for any horizon.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 --quiet

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from datetime import date
from pathlib import Path

import lightgbm
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import parks_lib
from parks_lib.forecast import CALENDAR_FEATURES, FEATURE_SETS, backtest, build_frame, fit
from parks_lib.serving import CrowdForecastModel

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/park-crowd-forecast")
model_name = f"{catalog}.{schema}.crowd_forecast"

crowd = spark.table(f"{catalog}.{schema}.raw_crowd_calendar").toPandas()
calendar = spark.table(f"{catalog}.{schema}.calendar_features").toPandas()
weather = spark.table(f"{catalog}.{schema}.raw_weather_daily").drop("is_forecast", "fetched_at").toPandas()
df = build_frame(crowd, calendar, weather)
print("Measured days per park:", df.groupby("park_id", observed=True)["crowd_percent"].count().to_dict())

# COMMAND ----------

test_years = [y for y in range(2023, date.today().year + 1)]
production_features = CALENDAR_FEATURES

with mlflow.start_run(run_name="crowd-forecast") as run:
    results = backtest(df, test_years)
    mlflow.log_table(results, "backtest.json")
    for r in results.itertuples():
        key = r.method.replace(": ", "_").replace(" ", "_").replace("+", "_")
        mlflow.log_metrics({f"mae_{r.test_year}_{key}": r.mae, f"within10_{r.test_year}_{key}": r.within_10})
    display(results.pivot(index="method", columns="test_year", values="mae").round(1).reset_index())

    model = fit(df, production_features)
    mlflow.log_params({"features": ",".join(production_features), "train_rows": int(df["crowd_percent"].notna().sum()),
                       "train_end": str(df.dropna(subset=["crowd_percent"])["date"].max()), "test_years": test_years})
    sample = df.dropna(subset=["crowd_percent"]).tail(50)[production_features].astype({"park_id": int})
    wrapped = CrowdForecastModel(model, production_features)
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapped,
        code_paths=[str(Path(parks_lib.__file__).parent)],
        signature=infer_signature(sample, wrapped.predict(None, sample)),
        pip_requirements=[f"lightgbm=={lightgbm.__version__}", f"pandas=={pd.__version__}", "python-dateutil"],
        registered_model_name=model_name,
    )

client = MlflowClient()
version = info.registered_model_version
client.set_registered_model_alias(model_name, "champion", version)
print(f"Registered {model_name} version {version} as champion (run {run.info.run_id})")
