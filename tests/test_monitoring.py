import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "src"))

from parks_lib import monitoring as mon  # noqa: E402
from parks_lib.monitoring import Trigger  # noqa: E402

TODAY = pd.Timestamp("2026-11-18")


def errors_frame(dates, error, bucket=mon.TRIGGER_HORIZON, parks=(64, 65), horizon=3):
    """Forecast rows for every park and date with a fixed absolute error."""
    rows = [{"park_id": p, "date": pd.Timestamp(d), "horizon": horizon, "bucket": bucket, "actual": 50.0 + error,
             "pred": 50.0} for d in dates for p in parks]
    return pd.DataFrame(rows)


def test_horizons_split_at_two_weeks():
    assert list(mon.bucket_of(pd.Series([0, 14, 15, 90, 91]))) == ["0-14 days", "0-14 days", "15-90 days",
                                                                  "15-90 days", pd.NA]


def test_live_errors_keep_the_last_run_of_a_day_and_count_horizon_from_it():
    history = pd.DataFrame({
        "park_id": [65, 65, 65, 65], "date": ["2026-11-10"] * 3 + ["2026-11-11"],
        "predicted_crowd": [40.0, 45.0, 30.0, 50.0], "queue_times_forecast": [None, None, 55.0, None],
        "scored_at": pd.to_datetime(["2026-11-01 11:30", "2026-11-01 18:00", "2026-11-05 11:30", "2026-11-05 11:30"],
                                    utc=True)})
    measured = pd.DataFrame({"park_id": [65], "date": ["2026-11-10"], "crowd_percent": [60.0]})
    live = mon.live_errors(history, measured)
    assert sorted(live["pred"]) == [30.0, 45.0]            # Nov 1's later run replaced its earlier one
    assert sorted(live["horizon"]) == [5, 9]               # Nov 11 isn't measured, so it's left out
    assert set(live["bucket"]) == {"0-14 days"}
    assert live.loc[live["horizon"] == 5, "qt_pred"].item() == 55.0


def test_rolling_error_weights_every_forecast_and_needs_enough_days():
    dates = pd.date_range("2026-10-01", periods=20)
    e = pd.concat([errors_frame(dates[:10], 10.0), errors_frame(dates[10:], 20.0, parks=(64,))])
    r = mon.rolling_mae(e)
    assert np.isnan(r[dates[12]])                          # 13 measured days: fewer than MIN_WINDOW_DAYS
    assert r[dates[19]] == pytest.approx((20 * 10 + 10 * 20) / 30)   # 20 rows at 10 and 10 rows at 20


def test_band_uses_two_years_ending_90_days_back():
    dates = pd.date_range("2023-01-01", "2026-11-18")
    rolling = pd.Series(np.where(dates >= TODAY - pd.Timedelta(days=90), 99.0, 10.0), index=dates)
    rolling[(dates >= "2025-06-01") & (dates < "2025-08-01")] = 20.0   # 61 of 730 days: above the 95th percentile
    rolling[dates < "2024-08-20"] = 99.0                   # older than two years before Aug 20, 2026
    lo, hi = mon.band(rolling, TODAY)
    assert lo == 10.0 and hi == 20.0                       # neither the last 90 days nor the old 99s are in it
    assert np.isnan(mon.band(rolling[-100:], TODAY)[0])    # not enough reference


def test_psi_is_zero_for_the_same_distribution_and_grows_with_a_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(50, 10, 5000)
    assert mon.psi(ref, ref) == pytest.approx(0, abs=1e-9)
    assert mon.psi(ref, ref + 5) < mon.psi(ref, ref + 15)
    assert np.isfinite(mon.psi(ref, np.full(50, 500.0)))   # every value in one bin: smoothed, not infinite


def monthly(start, months, shift_month=None, shift=0.0, seed=0):
    """Daily values for two parks over `months`, with an optional shift in one month."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=months, freq="MS")
    days = pd.date_range(dates[0], dates[-1] + pd.offsets.MonthEnd(0))
    rows = [{"park_id": p, "date": d, "crowd_percent": 50 + 10 * rng.standard_normal()
             + (shift if shift_month and d.strftime("%Y-%m") == shift_month else 0)} for d in days for p in (64, 65)]
    return pd.DataFrame(rows)


def test_calibrated_threshold_sits_above_the_textbook_line_for_a_month_of_data():
    ref = monthly("2024-01-01", 24)
    cur = monthly("2026-01-01", 1, seed=9)
    t = mon.psi_test(ref["crowd_percent"], cur["crowd_percent"], mon._week(ref["date"]),
                     ref["date"].dt.to_period("M"), mon._week(cur["date"]))
    assert t["threshold"] > 0.2 and t["psi"] < t["threshold"]   # an unchanged month isn't flagged


def test_crowd_drift_flags_a_shifted_month_against_the_same_month_in_earlier_years():
    data = monthly("2022-01-01", 48, shift_month="2025-07", shift=25)
    d = mon.crowd_drift(data).set_index("month")
    assert d.loc["2025-07-01", "psi"] > d.loc["2025-07-01", "threshold"]
    assert d.loc["2025-06-01", "psi"] < d.loc["2025-06-01", "threshold"]
    assert d.index.min() == pd.Timestamp("2024-01-01")     # needs two earlier years of the same month


def test_error_drift_uses_live_months_once_they_have_enough_days():
    rng = np.random.default_rng(2)
    days = pd.date_range("2023-06-01", "2026-10-31")
    replay = pd.DataFrame([{"park_id": p, "date": d, "horizon": 3, "bucket": mon.TRIGGER_HORIZON,
                            "actual": 50 + 8 * rng.standard_normal(), "pred": 50.0} for d in days for p in (64, 65)])
    live = errors_frame(pd.date_range("2026-10-01", "2026-10-31"), 30.0)
    live["qt_pred"] = np.nan
    d = mon.error_drift(replay, live).set_index("month")
    assert d.loc["2026-10-01", "source"] == "live" and d.loc["2026-09-01", "source"] == "replay"
    assert d.loc["2026-10-01", "psi"] > d.loc["2026-10-01", "threshold"]    # every live miss is 30 points


def accuracy_frame(live_mae, band_high=20.0, start="2026-10-01"):
    dates = pd.date_range(start, periods=len(live_mae))
    return pd.DataFrame({"bucket": mon.TRIGGER_HORIZON, "date": dates, "live_mae": live_mae, "band_high": band_high})


def test_error_trigger_needs_14_days_in_a_row_above_the_band():
    assert not mon.error_trigger(accuracy_frame([25.0] * 13)).fired
    t = mon.error_trigger(accuracy_frame([25.0] * 14))
    assert t.fired and "14 days in a row" in t.detail
    assert not mon.error_trigger(accuracy_frame([25.0] * 10 + [15.0] + [25.0] * 10)).fired
    assert mon.fire_dates(accuracy_frame([25.0] * 20)) == [pd.Timestamp("2026-10-14")]


def test_only_evidence_after_the_last_tuning_run_counts():
    acc = accuracy_frame([25.0] * 30)
    assert mon.error_trigger(acc).fired
    assert not mon.error_trigger(acc, since=pd.Timestamp("2026-10-20", tz="UTC")).fired   # 10 days since


def test_drift_trigger_reads_the_latest_live_month():
    drift = pd.DataFrame({"month": pd.to_datetime(["2026-09-01", "2026-10-01", "2026-10-01"]),
                          "line": ["model errors", "model errors", "crowd level"],
                          "source": ["live", "live", "measured"], "psi": [3.0, 0.4, 9.0], "threshold": [1.0] * 3})
    assert not mon.drift_trigger(drift).fired                     # October's errors are fine; crowd level doesn't count
    assert mon.drift_trigger(drift.iloc[:1]).fired
    assert not mon.drift_trigger(drift.iloc[:1], since="2026-09-15").fired
    assert not mon.drift_trigger(pd.DataFrame()).fired


def test_backstop_after_90_days():
    assert not mon.backstop_trigger(pd.Timestamp("2026-11-02 14:00", tz="UTC"), TODAY).fired
    assert mon.backstop_trigger(pd.Timestamp("2026-08-20"), TODAY).fired
    assert mon.backstop_trigger(None, TODAY).fired


ERROR = Trigger("error", True, "28-day error high")
BACKSTOP = Trigger("backstop", True, "95 days")
QUIET = Trigger("drift", False, "fine")


def test_decide():
    assert mon.decide([QUIET], TODAY)[0] == "none"
    assert mon.decide([ERROR, QUIET], TODAY)[0] == "start"
    assert mon.decide([ERROR], TODAY, last_started=TODAY - pd.Timedelta(days=5))[0] == "cooldown"
    failed_auto = {"trigger": "error", "promoted": False}
    action, reason = mon.decide([ERROR], TODAY, last_started=TODAY - pd.Timedelta(days=30), last_run=failed_auto)
    assert action == "needs a person" and "missing feature" in reason
    assert mon.decide([ERROR], TODAY, last_run={"trigger": "error", "promoted": True})[0] == "start"
    assert mon.decide([ERROR], TODAY, last_run={"trigger": "manual", "promoted": False})[0] == "start"
    assert mon.decide([BACKSTOP], TODAY, last_run=failed_auto)[0] == "start"   # the backstop isn't a problem
    assert mon.first_trigger([BACKSTOP, ERROR]) == "error" and mon.first_trigger([QUIET]) == "manual"


def test_walk_forward_refits_on_mondays_and_forecasts_90_days():
    days = pd.date_range("2024-01-01", "2024-12-31")
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"park_id": pd.Categorical(np.repeat([64, 65], len(days))), "date": np.tile(days, 2),
                       "day_of_week": np.tile(days.dayofweek, 2)})
    df["crowd_percent"] = 40 + 5 * (df["day_of_week"] >= 5) + rng.normal(0, 1, len(df))
    out = mon.walk_forward(df, ["park_id", "day_of_week"], {"n_estimators": 20}, start="2024-10-02",
                           end="2024-10-31", step_days=14)
    assert set(out["trained_through"].dt.dayofweek) == {0}            # Oct 2 snaps back to Monday Sep 30
    assert out["horizon"].between(0, 90).all() and (out["date"] >= out["trained_through"]).all()
    assert out["trained_through"].nunique() == 3 and set(out["bucket"]) == set(mon.HORIZONS)
