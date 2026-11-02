import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "databricks" / "src"))

from parks_lib.forecast import CALENDAR_FEATURES, CALENDAR_GROUPS  # noqa: E402
from parks_lib.tuning import OPTIONAL_GROUPS, current_settings, decide, from_best, tune  # noqa: E402


def test_best_params_split_back_into_settings_and_groups():
    best = {"n_estimators": 300, "learning_rate": 0.05, "use holidays": True, "use school breaks": False,
            "use events": True}
    params, groups = from_best(best)
    assert params == {"n_estimators": 300, "learning_rate": 0.05}
    assert groups == ["Holidays", "Events"]
    assert OPTIONAL_GROUPS == list(CALENDAR_GROUPS)[1:]


def test_production_uses_the_latest_promoted_settings():
    assert current_settings(None) == ({}, CALENDAR_FEATURES, "repo defaults (LGBM_PARAMS)")
    rows = pd.DataFrame([
        {"run_id": "a", "run_at": pd.Timestamp("2026-11-01"), "promoted": True, "params": json.dumps({"num_leaves": 8}),
         "features": "park_id,year"},
        {"run_id": "b", "run_at": pd.Timestamp("2026-11-03"), "promoted": False, "params": json.dumps({"num_leaves": 99}),
         "features": "park_id"},
    ])
    params, features, source = current_settings(rows)
    assert params == {"num_leaves": 8} and features == ["park_id", "year"] and source == "tuning run a"
    assert current_settings(rows.assign(promoted=False))[2].startswith("repo defaults")


def test_promotion_needs_the_whole_interval_below_zero():
    assert decide({"diff": -0.5, "lo": -0.9, "hi": -0.1})[0] is True
    assert decide({"diff": -0.1, "lo": -0.42, "hi": 0.18})[0] is False     # the first study's result
    assert decide({"diff": 0.4, "lo": 0.1, "hi": 0.7})[0] is False


def test_a_short_study_runs_end_to_end():
    df = pd.read_parquet(ROOT / "public_app" / "data" / "daily.parquet").astype({"park_id": "category"})
    study = tune(df, n_trials=3, seed=0)
    params, groups = from_best(study.best_params)
    assert {"n_estimators", "learning_rate", "max_depth", "reg_alpha"} <= set(params)
    assert set(groups) <= set(OPTIONAL_GROUPS)
    assert 5 < study.best_value < 30
