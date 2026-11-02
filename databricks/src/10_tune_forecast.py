# Databricks notebook source
# MAGIC %md
# MAGIC # 10 Tune: Optuna search, one look at the holdout, promote only on a reliable gain
# MAGIC
# MAGIC 1. Optuna searches LightGBM settings and the calendar feature groups by mean absolute error on test
# MAGIC    years 2023-2024, each predicted by a model trained only on earlier days. Every trial is a nested
# MAGIC    MLflow run; a trial whose first year trails the median stops there.
# MAGIC 2. 2025 onward, which the search never saw, is predicted once with the tuned settings and once with
# MAGIC    the settings production uses now (the latest promoted row of `forecast_settings`, else the repo's
# MAGIC    `LGBM_PARAMS`). A paired block bootstrap says whether the difference is more than holdout luck.
# MAGIC 3. The model refit on every measured day is registered as `crowd_forecast@challenger`. It becomes
# MAGIC    `@champion` only if its holdout error is reliably lower (the 95% interval entirely below zero).
# MAGIC    Every run is appended to `forecast_settings`; the weekly `train-forecast` job retrains with the
# MAGIC    latest promoted settings, so a promotion survives Monday's retrain.
# MAGIC
# MAGIC Runs on demand (the `tune-forecast` job has no schedule). The first study (2026-10-22, 100 trials, an
# MAGIC Assistant notebook) found no reliable gain: -0.11 points, interval -0.42 to +0.18.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 optuna==4.9.0 --quiet

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
dbutils.widgets.text("n_trials", "60")
dbutils.widgets.text("timeout_minutes", "30")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
n_trials = int(dbutils.widgets.get("n_trials"))
timeout = 60 * int(dbutils.widgets.get("timeout_minutes"))

# COMMAND ----------

import json
from datetime import date, datetime, timezone
from pathlib import Path

import lightgbm
import mlflow
import optuna
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import parks_lib
from parks_lib.evaluation import block_bootstrap, robust_to_block_length
from parks_lib.forecast import backtest_predictions, build_frame, features_for, fit
from parks_lib.serving import CrowdForecastModel
from parks_lib.tuning import HOLDOUT_FROM, TUNE_YEARS, current_settings, decide, from_best, tune

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/park-crowd-forecast")
model_name = f"{catalog}.{schema}.crowd_forecast"
settings_table = f"{catalog}.{schema}.forecast_settings"
optuna.logging.set_verbosity(optuna.logging.WARNING)

crowd = spark.table(f"{catalog}.{schema}.raw_crowd_calendar").toPandas()
calendar = spark.table(f"{catalog}.{schema}.calendar_features").toPandas()
weather = spark.table(f"{catalog}.{schema}.raw_weather_daily").drop("is_forecast", "fetched_at").toPandas()
df = build_frame(crowd, calendar, weather)

history = spark.table(settings_table).toPandas() if spark.catalog.tableExists(settings_table) else None
prod_params, prod_features, prod_source = current_settings(history)
holdout_years = list(range(HOLDOUT_FROM, date.today().year + 1))
print(f"Tuning on {TUNE_YEARS}, holdout {holdout_years}; production settings: {prod_source}")

# COMMAND ----------


def log_trial(study, trial):
    """Each trial as a nested run: its settings, how many years it ran, and its tuning-years MAE."""
    with mlflow.start_run(run_name=f"trial {trial.number}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.set_tag("state", trial.state.name.lower())
        mlflow.log_metric("years_run", len(trial.intermediate_values))
        if trial.value is not None:
            mlflow.log_metric("tuning_mae", trial.value)


with mlflow.start_run(run_name="optuna-crowd-forecast") as run:
    study = tune(df, n_trials=n_trials, timeout=timeout, callbacks=[log_trial])
    params, groups = from_best(study.best_params)
    features = features_for(groups)
    states = pd.Series([t.state.name.lower() for t in study.trials]).value_counts()
    mlflow.log_params({**params, "feature_groups": ",".join(groups), "trials": len(study.trials),
                       "trials_pruned": int(states.get("pruned", 0)), "tune_years": TUNE_YEARS,
                       "holdout_years": holdout_years, "production_settings": prod_source,
                       "lightgbm": lightgbm.__version__, "optuna": optuna.__version__})
    mlflow.log_metric("tuning_mae", study.best_value)
    print(f"{len(study.trials)} trials ({int(states.get('pruned', 0))} stopped early), "
          f"best tuning MAE {study.best_value:.2f}; groups: {groups}")

    # One look at the holdout: tuned settings against production's, on the same park-days
    tuned, _ = backtest_predictions(df, features, params, holdout_years)
    prod, _ = backtest_predictions(df, prod_features, prod_params, holdout_years)
    mae = {name: float((p["actual"] - p["pred"]).abs().mean()) for name, p in [("tuned", tuned), ("production", prod)]}
    boot = block_bootstrap(tuned, prod)
    robustness = robust_to_block_length(tuned, prod)
    promote, reason = decide(boot)
    mlflow.log_metrics({"holdout_mae_tuned": mae["tuned"], "holdout_mae_production": mae["production"],
                        "holdout_diff": boot["diff"], "holdout_ci_low": boot["lo"], "holdout_ci_high": boot["hi"]})
    mlflow.log_table(robustness, "bootstrap_by_block_length.json")
    trials = study.trials_dataframe().drop(columns=["datetime_start", "datetime_complete", "duration"], errors="ignore")
    mlflow.log_table(trials, "trials.json")
    display(robustness.round(3))
    print(f"Holdout MAE: tuned {mae['tuned']:.2f}, production {mae['production']:.2f}. {reason}")

    # The registered challenger: the tuned settings refit on every measured day
    model = fit(df, features, params)
    wrapped = CrowdForecastModel(model, features)
    sample = df.dropna(subset=["crowd_percent"]).tail(50)[features].astype({"park_id": int})
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapped,
        code_paths=[str(Path(parks_lib.__file__).parent)],
        signature=infer_signature(sample, wrapped.predict(None, sample)),
        pip_requirements=[f"lightgbm=={lightgbm.__version__}", f"pandas=={pd.__version__}", "python-dateutil"],
        registered_model_name=model_name,
    )

# COMMAND ----------

client = MlflowClient()
version = info.registered_model_version
client.set_registered_model_alias(model_name, "challenger", version)
if promote:
    client.set_registered_model_alias(model_name, "champion", version)

row = pd.DataFrame([{
    "run_id": run.info.run_id, "run_at": datetime.now(timezone.utc), "model_version": int(version),
    "params": json.dumps(params), "features": ",".join(features), "trials": len(study.trials),
    "tuning_mae": float(study.best_value), "holdout_mae": mae["tuned"], "production_holdout_mae": mae["production"],
    "holdout_diff": boot["diff"], "holdout_ci_low": boot["lo"], "holdout_ci_high": boot["hi"],
    "production_settings": prod_source, "promoted": promote, "reason": reason,
}])
spark.createDataFrame(row).write.mode("append").saveAsTable(settings_table)
print(f"Registered {model_name} v{version} as challenger; {'promoted to' if promote else 'not'} champion: {reason}")
