import sys
from pathlib import Path

import pandas as pd
import pytest

BUNDLE = Path(__file__).resolve().parents[1] / "databricks"
sys.path[:0] = [str(BUNDLE / "src"), str(BUNDLE / "app")]

import park_data  # noqa: E402
import transforms  # noqa: E402
from parks_lib.staffing import evaluate_plan, plan_staffing  # noqa: E402

LOCATIONS = pd.read_csv(BUNDLE / "seeds" / "staffing_locations.csv")


def test_hour_labels():
    assert [transforms.hour_label(h) for h in (0, 9, 12, 13, 23)] == ["12am", "9am", "12pm", "1pm", "11pm"]


def test_signed_deltas_keep_zero_numeric():
    assert transforms.signed(0.4) == 0
    assert transforms.signed(12135) == "+12,135"
    assert transforms.signed(-1500, prefix="$") == "-$1,500"
    assert transforms.signed(19, suffix=" min") == "+19 min"


def test_stamp_shows_park_time():
    assert transforms.stamp(pd.Timestamp("2026-10-22 18:11:58", tz="UTC")) == "Oct 22, 2:11pm ET"


def test_calendar_grid_starts_weeks_on_monday():
    forecast = pd.DataFrame({"park_id": 65, "date": pd.date_range("2026-10-20", periods=10), "predicted_crowd": 50.0})
    grid = transforms.calendar_grid(forecast, 65, "2026-10-22")
    assert grid["date"].min() == pd.Timestamp("2026-10-22")
    assert (grid["week"].dt.dayofweek == 0).all()
    assert grid.loc[grid["date"] == "2026-10-26", "week_label"].item() == "Oct 26"


def test_trend_frame_skips_days_without_a_queue_times_forecast():
    forecast = pd.DataFrame({"park_id": 65, "date": pd.date_range("2026-10-22", periods=3),
                             "predicted_crowd": [30.0, 40.0, 50.0], "queue_times_forecast": [36.0, None, None]})
    measured = pd.DataFrame({"park_id": 65, "date": pd.date_range("2026-10-19", periods=3), "measured_crowd": 60.0})
    counts = transforms.trend_frame(forecast, measured, 65)["series"].value_counts()
    assert counts.to_dict() == {"Forecast": 3, "Measured": 3, "Queue-times forecast": 1}


def test_published_plan_replays_to_the_stored_numbers():
    locs = LOCATIONS[LOCATIONS["park_id"] == 64].to_dict("records")
    plan = plan_staffing(locs, 40, 9, 17)
    shifts = pd.DataFrame(plan.shifts).assign(park_id=64, date=pd.Timestamp("2026-10-22"))
    replay = evaluate_plan(transforms.published_plan(shifts, 64, "2026-10-22"), locs, 40, 9, 17)
    assert transforms.headline(replay) == transforms.headline(plan)


def test_at_max_crew_lists_capped_locations():
    hourly = [{"location": "Minion Cafe", "hour": 12, "crew": 14}, {"location": "Leaky Cauldron", "hour": 12, "crew": 5}]
    assert transforms.at_max_crew(hourly, LOCATIONS) == ["Minion Cafe"]


@pytest.fixture(scope="module")
def frames():
    """Small stand-in for the warehouse tables: real optimizer plans for two days at each staffed park."""
    today = pd.Timestamp(park_data.today())
    days = pd.date_range(today, periods=2)
    forecast = pd.DataFrame([(p, d, 40.0 + i, None, pd.Timestamp("2026-10-22 18:00", tz="UTC"), None, False)
                             for p in park_data.PARKS for i, d in enumerate(pd.date_range(today, periods=30))],
                            columns=["park_id", "date", "predicted_crowd", "queue_times_forecast", "scored_at",
                                     "holiday_name", "event_hhn"])
    measured = pd.DataFrame([(p, d, 55.0) for p in park_data.PARKS for d in pd.date_range(today - pd.Timedelta(days=30), periods=30)],
                            columns=["park_id", "date", "measured_crowd"])
    summary, shifts, hourly = [], [], []
    for park_id in LOCATIONS["park_id"].unique():
        locs = LOCATIONS[LOCATIONS["park_id"] == park_id].to_dict("records")
        for day in days:
            plan = plan_staffing(locs, 40, 9, 17)
            busier = evaluate_plan(plan, locs, 55, 9, 17)
            key = {"park_id": int(park_id), "date": day}
            summary.append({**key, "predicted_crowd": 40.0, "open_hour": 9, "close_hour": 17,
                            **transforms.headline(plan), "wait_guest_hours_if_15_busier": busier.wait_guest_hours,
                            "worst_wait_minutes_if_15_busier": transforms.headline(busier)["worst_wait_minutes"],
                            "skipped_guests_if_15_busier": busier.skipped_guests,
                            "planned_at": pd.Timestamp("2026-10-22 18:40", tz="UTC")})
            shifts += [{**key, **s} for s in plan.shifts]
            hourly += [{**key, **h} for h in plan.hourly]
    return {"forecast": forecast, "measured": measured, "summary": pd.DataFrame(summary),
            "shifts": pd.DataFrame(shifts), "hourly": pd.DataFrame(hourly)}


@pytest.fixture
def app(frames, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "test")
    monkeypatch.setattr(park_data, "load", lambda: frames)
    return AppTest.from_file("../databricks/app/streamlit_app.py", default_timeout=60).run()


@pytest.mark.parametrize("page", ["app_pages/crowd_calendar.py", "app_pages/staffing_plan.py", "app_pages/what_if.py"])
def test_pages_render(app, page):
    if page != "app_pages/crowd_calendar.py":
        app.switch_page(page).run()
    assert not app.exception
    assert len(app.metric) >= 4


def test_staffing_follows_the_chosen_park(app):
    app.switch_page("app_pages/staffing_plan.py").run()
    first = app.metric[0].value
    app.segmented_control(key="park").set_value("Epic Universe").run()
    assert not app.exception
    assert app.metric[0].value != first


def test_what_if_warns_when_the_worker_cap_is_too_low(app):
    app.switch_page("app_pages/what_if.py").run()
    app.number_input(key="cap").set_value(5)
    app.form_submit_button[0].click().run()
    assert not app.exception
    assert any("worker cap" in w.value for w in app.warning)
