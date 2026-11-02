"""Optuna search over LightGBM settings and calendar feature groups, and the rule for promoting the result.

Protocol (the same as the first study, 2026-10-22): tune on test years 2023-2024, each predicted by a model
trained only on earlier days; then look once at 2025 onward, which the search never saw, and compare with
the settings production uses now. Park hours and weather stay out of the search: production forecasts 90
days ahead, and neither is known that far out.

Promotion: the tuned settings replace production's only when a paired block bootstrap on the holdout days
says their error is reliably lower (the whole 95% interval below zero; see evaluation.py). Each run that
looks at the holdout spends a little of its independence, so the bar stays strict.
"""

import json

import numpy as np
import pandas as pd

from parks_lib.forecast import CALENDAR_FEATURES, CALENDAR_GROUPS, features_for, fit, predict

# optuna is imported inside the functions that search: the weekly training job imports this module for
# current_settings and installs only lightgbm.

TUNE_YEARS = [2023, 2024]
HOLDOUT_FROM = 2025
OPTIONAL_GROUPS = list(CALENDAR_GROUPS)[1:]


def search_space(trial):
    """(params, groups) for one trial. Loss stays absolute error, the measure the backtest reports."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 128, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }
    groups = [g for g in OPTIONAL_GROUPS if trial.suggest_categorical(f"use {g.lower()}", [True, False])]
    return params, groups


def from_best(best_params):
    """Split a study's best_params back into LightGBM params and the feature groups switched on."""
    params = {k: v for k, v in best_params.items() if not k.startswith("use ")}
    groups = [g for g in OPTIONAL_GROUPS if best_params.get(f"use {g.lower()}", True)]
    return params, groups


def tuning_mae(df, params, features, years=TUNE_YEARS, trial=None):
    """Mean daily absolute error over `years`, each predicted by a model trained on the days before it.
    Given a trial, reports after each year so Optuna can stop a weak one."""
    dates = pd.to_datetime(df["date"])
    errors = []
    for step, year in enumerate(years):
        train, test = df[dates.dt.year < year], df[dates.dt.year == year].dropna(subset=["crowd_percent"])
        pred = predict(fit(train, features, params), test, features)
        errors.append(np.abs(test["crowd_percent"].to_numpy() - pred))
        if trial is not None:
            trial.report(float(np.concatenate(errors).mean()), step)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
    return float(np.concatenate(errors).mean())


def tune(df, n_trials=60, timeout=None, seed=42, callbacks=None):
    """TPE search for the lowest tuning-years MAE. After 8 full trials, a trial whose first year's error
    is above the median of earlier trials' stops there (median pruning)."""
    import optuna

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))

    def objective(trial):
        params, groups = search_space(trial)
        return tuning_mae(df, params, features_for(groups), trial=trial)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)
    return study


def current_settings(rows):
    """Production's settings: the latest promoted row of the forecast_settings table, else the repo's.

    `rows`: that table as a DataFrame (or None before it exists). Returns (params, features, source).
    """
    if rows is not None and len(rows):
        promoted = rows[rows["promoted"]].sort_values("run_at")
        if len(promoted):
            last = promoted.iloc[-1]
            return json.loads(last["params"]), last["features"].split(","), f"tuning run {last['run_id']}"
    return {}, list(CALENDAR_FEATURES), "repo defaults (LGBM_PARAMS)"


def decide(boot):
    """Promote only when the tuned model's holdout error is reliably lower. Returns (promote, reason)."""
    interval = f"{boot['diff']:+.2f} points a day, 95% interval {boot['lo']:+.2f} to {boot['hi']:+.2f}"
    if boot["hi"] < 0:
        return True, f"reliably lower holdout error: {interval}"
    return False, f"not reliably better on the holdout: {interval}"

