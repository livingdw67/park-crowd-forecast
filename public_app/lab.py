"""Model lab and study logic for the public app, kept free of Streamlit so it can be unit-tested.

Everything runs on the exported snapshot in data/ with the same parks_lib code the Databricks jobs
run: no workspace, no model API, nothing billed per click.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for path in (ROOT / "databricks" / "src", ROOT / "databricks" / "app"):  # parks_lib; the Databricks app's charts
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parks_lib.causal import deseasonalize, months_between, placebo_test  # noqa: E402
from parks_lib.evaluation import block_bootstrap  # noqa: E402,F401  (the pages call it as lab.block_bootstrap)
from parks_lib.forecast import (CALENDAR_GROUPS, HOURS_FEATURES, LGBM_PARAMS, WEATHER_FEATURES,  # noqa: E402
                                backtest_predictions, fit, predict, profile_mean, seasonal_naive)

DATA = APP_DIR / "data"
PARKS = {65: "Universal Studios Florida", 64: "Islands of Adventure", 334: "Epic Universe", 67: "Volcano Bay"}
TEST_YEARS = [2023, 2024, 2025, 2026]

# Feature groups the lab can switch on and off: the calendar groups the tuning job searches, plus the two
# measured ones. The first is always on. The champion (production model) uses the four calendar groups.
FEATURE_GROUPS = {**CALENDAR_GROUPS, "Park hours": HOURS_FEATURES, "Weather": WEATHER_FEATURES}
CHAMPION_GROUPS = list(CALENDAR_GROUPS)
# The settings the lab exposes, at the champion's values. LGBM_PARAMS leaves depth and regularization at
# LightGBM's defaults (no depth limit, no L1/L2 penalty); they are spelled out so a preset can change them.
CHAMPION_PARAMS = {**{k: LGBM_PARAMS[k] for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples",
                                                  "subsample", "colsample_bytree")},
                   "max_depth": -1, "reg_alpha": 0.0, "reg_lambda": 0.0}
# Best trial of the 100-trial Optuna study run in Databricks on 2026-10-22 (MLflow run
# e45c4b4427e444628df292d66c0d7c0c, "crowd-forecast-tuned-2023-2024"): tuned on test years 2023-2024 only,
# champion features, absolute-error loss. Exact values: rounding them moves the 2025-2026 result.
OPTUNA_RUN = "e45c4b4427e444628df292d66c0d7c0c"
OPTUNA_TUNED_ON = [2023, 2024]
OPTUNA_PARAMS = {"n_estimators": 860, "learning_rate": 0.0452130149175768, "num_leaves": 56, "max_depth": 5,
                 "min_child_samples": 6, "subsample": 0.8788647169098125, "colsample_bytree": 0.6086232578580403,
                 "reg_alpha": 1.0038881495789473, "reg_lambda": 2.446785580104569}
OBJECTIVES = {"Absolute error (L1)": "regression_l1", "Squared error (L2)": "regression", "Huber": "huber"}

# Why-this-day explanation: model features grouped the way a reader thinks about a day
EXPLAIN_GROUPS = {
    "Park": ["park_id"], "Year (trend)": ["year"], "Time of year": ["month", "day_of_year", "week_of_year"],
    "Day of week": ["day_of_week", "is_weekend"], **{g: FEATURE_GROUPS[g] for g in CHAMPION_GROUPS[1:]},
}

# Epic Universe impact study (databricks/src/09_epic_impact_study.py, the weekly job)
EPIC_OPENING = pd.Timestamp("2025-05-22")
BOTH = 6564
IMPACT_UNITS = {"Universal Studios Florida": 65, "Islands of Adventure": 64, "Both parks, averaged": BOTH}
DONORS = {16: "Disneyland", 17: "Disney California Adventure", 66: "Universal Studios Hollywood",
          61: "Knott's Berry Farm", 279: "Legoland California", 20: "SeaWorld San Diego",
          32: "Six Flags Magic Mountain", 24: "Busch Gardens Tampa"}
DEFAULT_DONORS = [d for d in DONORS if d != 24]   # Busch Gardens is near Orlando, so out by default
NAMES = {**PARKS, **DONORS, BOTH: "Universal Studios + Islands of Adventure"}


# ---------------------------------------------------------------------------------------- data

def load_daily():
    df = pd.read_parquet(DATA / "daily.parquet")
    df["park_id"] = df["park_id"].astype("category")
    return df


def load_monthly():
    m = pd.read_parquet(DATA / "monthly.parquet")
    panel = m.pivot(index="month", columns="park_id", values="crowd").sort_index()
    panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="MS"))
    panel[BOTH] = panel[[65, 64]].mean(axis=1, skipna=False)
    return panel


def load_forecast():
    return pd.read_parquet(DATA / "forecast.parquet")


def day_label(ts, weekday="%a", month="%b", year=False):
    """'Sat Oct 31' (or with the year), without strftime's %-d, which Windows lacks."""
    ts = pd.Timestamp(ts)
    return f"{ts.strftime(weekday)} {ts.strftime(month)} {ts.day}" + (f", {ts.year}" if year else "")


def features_for(groups):
    return [f for g in FEATURE_GROUPS if g in groups for f in FEATURE_GROUPS[g]]


# ---------------------------------------------------------------------------------------- model lab

def run_backtest(df, features, params, test_years=TEST_YEARS):
    """(per-day predictions, gain importance): parks_lib's backtest, the one the tuning job runs."""
    return backtest_predictions(df, features, params, test_years)


def baselines(df, test_years=TEST_YEARS):
    """MAE per test year for the two baselines the production backtest reports."""
    dates = pd.to_datetime(df["date"])
    out = []
    for year in test_years:
        train, test = df[dates.dt.year < year], df[dates.dt.year == year]
        actual = test["crowd_percent"].to_numpy()
        for name, pred in [("Same weekday last year", seasonal_naive(train, test)),
                           ("Park x month x weekday average", profile_mean(train, test))]:
            err = np.abs(actual - np.asarray(pred, dtype=float))
            out.append({"test_year": year, "method": name, "mae": float(np.nanmean(err))})
    return pd.DataFrame(out)


def mae_by_year(preds):
    return (preds.assign(err=(preds["actual"] - preds["pred"]).abs())
            .groupby("test_year")["err"].mean().rename("mae").reset_index())


def verdict(boot):
    """Plain-language reading of the bootstrap interval."""
    if boot["hi"] < 0:
        return "better", f"Your model's errors are smaller than the champion's: {abs(boot['diff']):.2f} points a day on average."
    if boot["lo"] > 0:
        return "worse", f"Your model's errors are larger than the champion's: {boot['diff']:.2f} points a day on average."
    return "tie", ("Can't tell apart from the champion: the interval includes zero, so a difference this size "
                   "could come from which days happened to be in the test years.")


# ---------------------------------------------------------------------------------------- why this day

def contributions(model, df, features):
    """SHAP contributions per day, summed into EXPLAIN_GROUPS. Columns: the groups plus `baseline`."""
    from parks_lib.forecast import _matrix

    raw = model.predict(_matrix(df, features), pred_contrib=True)
    frame = pd.DataFrame(raw[:, :-1], columns=features, index=df.index)
    out = pd.DataFrame({g: frame[[f for f in fs if f in frame]].sum(axis=1) for g, fs in EXPLAIN_GROUPS.items()})
    out["baseline"] = raw[:, -1]
    return out


# ---------------------------------------------------------------------------------------- Epic study

def run_study(panel, unit, donors, pre_start, method, fake_opening=False):
    """The impact study for one park and setup, as the weekly job runs it, plus the chart frames."""
    last = panel[unit].last_valid_index()
    if fake_opening:   # a fake opening a year early, before Epic existed: should find nothing
        pre, post = months_between(pre_start, "2024-04-01"), months_between("2024-06-01", "2025-04-01")
        opening = pd.Timestamp("2024-05-22")
    else:
        pre, post = months_between(pre_start, "2025-04-01"), months_between("2025-06-01", last)
        opening = EPIC_OPENING
    adj = deseasonalize(panel, pre)
    result = placebo_test(adj, unit, donors, pre, post, method)
    est = result["treated"]
    offset = est.gap.loc[pre].mean()
    actual = panel.loc[pre + post, unit]
    counterfactual = actual - (est.gap - offset)
    series = pd.concat([
        pd.DataFrame({"month": actual.index, "series": "Actual", "value": actual.to_numpy()}),
        pd.DataFrame({"month": counterfactual.index, "series": "Without Epic (estimated)",
                      "value": counterfactual.to_numpy()}),
    ]).dropna()
    placebos = pd.DataFrame([(NAMES[p.unit], p.effect, False) for p in result["placebos"]]
                            + [(NAMES[unit], est.effect, True)], columns=["park", "effect", "is_treated"])
    pre_mean = float(panel.loc[pre, unit].mean())
    return {
        "effect": est.effect, "effect_pct": 100 * est.effect / pre_mean, "pre_mean": pre_mean,
        "p_effect": result["p_effect"], "p_ratio": result["p_ratio"], "n_placebos": len(result["placebos"]),
        "pre_rmspe": est.pre_rmspe, "weights": {NAMES[d]: w for d, w in est.weights.items() if w >= 0.005},
        "series": series, "placebos": placebos.sort_values("effect", ignore_index=True), "opening": opening,
        "months": (len(pre), len(post)), "through": post[-1],
    }
