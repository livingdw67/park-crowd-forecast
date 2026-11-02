import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[1] / "public_app"
sys.path.insert(0, str(APP))

import lab  # noqa: E402


@pytest.fixture(scope="module")
def daily():
    return lab.load_daily()


def test_snapshot_holds_only_derived_columns(daily):
    allowed = {"date", "crowd_percent", "holiday_name"} | {f for fs in lab.FEATURE_GROUPS.values() for f in fs}
    assert set(daily.columns) == allowed
    assert daily["crowd_percent"].between(0, 100).all()
    assert set(daily["park_id"].astype(int).unique()) <= set(lab.PARKS)


def test_feature_groups_keep_the_production_feature_order():
    from parks_lib.forecast import CALENDAR_FEATURES, FEATURE_SETS

    assert lab.features_for(lab.CHAMPION_GROUPS) == CALENDAR_FEATURES
    assert lab.features_for(list(lab.FEATURE_GROUPS)) == FEATURE_SETS["calendar+hours+weather"]


def test_bootstrap_of_a_model_against_itself_is_zero():
    preds = pd.DataFrame({"park_id": 65, "date": pd.date_range("2025-01-01", periods=60), "actual": 50.0,
                          "pred": np.linspace(30, 70, 60)})
    boot = lab.block_bootstrap(preds, preds)
    assert boot["diff"] == boot["lo"] == boot["hi"] == 0
    assert lab.verdict(boot)[0] == "tie"


def test_bootstrap_finds_a_clearly_better_model():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=300)
    actual = rng.uniform(20, 80, 300)
    champion = pd.DataFrame({"park_id": 65, "date": dates, "actual": actual, "pred": actual + rng.normal(0, 15, 300)})
    better = champion.assign(pred=actual + rng.normal(0, 5, 300))
    boot = lab.block_bootstrap(better, champion)
    assert boot["hi"] < 0 and lab.verdict(boot)[0] == "better"
    assert lab.verdict(lab.block_bootstrap(champion, better))[0] == "worse"


def test_day_contributions_add_up_to_the_prediction(daily):
    features = lab.features_for(lab.CHAMPION_GROUPS)
    model = lab.fit(daily, features, {"n_estimators": 50})
    rows = daily.sample(20, random_state=0)
    contrib = lab.contributions(model, rows, features)
    np.testing.assert_allclose(contrib.sum(axis=1), lab.predict(model, rows, features), atol=1e-6)


def test_day_label_has_no_padding():
    assert lab.day_label("2026-11-02") == "Mon Nov 2"
    assert lab.day_label("2026-11-02", "%A,", "%B", year=True) == "Monday, November 2, 2026"


def test_impact_study_runs_and_the_fake_opening_finds_less():
    panel = lab.load_monthly()
    real = lab.run_study(panel, 65, lab.DEFAULT_DONORS, "2022-01-01", "synthetic_control")
    fake = lab.run_study(panel, 65, lab.DEFAULT_DONORS, "2022-01-01", "synthetic_control", fake_opening=True)
    assert real["effect"] < 0
    assert abs(fake["effect"]) < abs(real["effect"])
    assert set(real["series"]["series"]) == {"Actual", "Without Epic (estimated)"}
    assert real["placebos"]["is_treated"].sum() == 1


@pytest.mark.parametrize("page", ["app_pages/model_lab.py", "app_pages/crowd_explorer.py", "app_pages/epic_impact.py"])
def test_pages_render_without_errors(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=180)
    at.run()
    at.switch_page(page).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 3


def test_lab_retrains_with_new_settings():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=180)
    at.run()
    at.slider(key="lab_n_estimators").set_value(200)
    at.pills(key="lab_groups").set_value(list(lab.FEATURE_GROUPS))
    at.button[0].click().run()     # the form's submit button
    assert not at.exception, at.exception
    assert at.metric[0].label == "Your model, average error"
    assert at.metric[0].value != at.metric[1].value      # no longer the champion's settings


def test_epic_page_needs_three_comparison_parks():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=180)
    at.run()
    at.switch_page("app_pages/epic_impact.py").run()
    at.pills(key="epic_donors").set_value([16, 17]).run()
    assert not at.exception
    assert "at least three" in at.warning[0].value
    at.pills(key="epic_donors").set_value(lab.DEFAULT_DONORS)
    at.toggle(key="epic_fake").set_value(True).run()
    assert not at.exception and not at.warning


def test_lab_on_champion_settings_says_so():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Retrain and backtest" in i.value for i in at.info)   # the "these are the champion's settings" note
    assert at.metric[0].value == at.metric[1].value


def test_spelled_out_defaults_leave_the_champion_unchanged(daily):
    features = lab.features_for(lab.CHAMPION_GROUPS)
    rows = daily.sample(50, random_state=1)
    plain = lab.fit(daily, features, {"n_estimators": 60})
    spelled = lab.fit(daily, features, {**lab.CHAMPION_PARAMS, "n_estimators": 60})
    np.testing.assert_array_equal(lab.predict(plain, rows, features), lab.predict(spelled, rows, features))


def test_optuna_preset_scores_only_unseen_years():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=180)
    at.run()
    next(b for b in at.button if b.label == "Load Optuna's best").click().run()
    assert at.slider(key="lab_learning_rate").value == lab.OPTUNA_PARAMS["learning_rate"]   # exact, not snapped
    next(b for b in at.button if b.label == "Retrain and backtest").click().run()
    assert not at.exception, at.exception
    assert at.segmented_control(key="lab_score_on").value == "2025-2026 only"
    assert any("Optuna's best" in c.value for c in at.caption)
    assert at.metric[0].value != at.metric[1].value


def test_app_reimports_a_helper_module_that_changed_on_disk():
    """Community Cloud keeps imported modules across a code pull; the app must notice and reload them."""
    import os
    import time

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=180)
    at.run()
    before = sys.modules["lab"]
    path = APP / "lab.py"
    stat = path.stat()
    try:
        os.utime(path, (stat.st_atime, time.time() + 60))   # as if a pull rewrote lab.py
        at.run()
        assert not at.exception
        assert sys.modules["lab"] is not before
    finally:
        os.utime(path, (stat.st_atime, stat.st_mtime))
