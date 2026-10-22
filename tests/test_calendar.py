import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "src"))

from parks_lib.calendar import break_windows, build_calendar_features

EVENTS = pd.DataFrame({
    "event": ["hhn"],
    "start_date": [date(2024, 8, 30)],
    "end_date": [date(2024, 11, 3)],
})


def features_for(start, end):
    return build_calendar_features(start, end, EVENTS).set_index("date")


def test_thanksgiving_week_runs_saturday_to_sunday():
    # Thanksgiving 2024 was Thursday Nov 28
    assert break_windows(2024)["thanksgiving_week"] == (date(2024, 11, 23), date(2024, 12, 1))


def test_winter_break_spans_new_year():
    first, last = break_windows(2024)["winter_break"]
    assert first == date(2024, 12, 21) and last == date(2025, 1, 5)


def test_easter_week_moves_with_easter():
    # Easter 2024 was March 31, Easter 2025 April 20
    assert break_windows(2024)["easter_week"] == (date(2024, 3, 23), date(2024, 4, 1))
    assert break_windows(2025)["spring_break_season"][1] == date(2025, 4, 28)


def test_fl_spring_break_covers_second_and_third_weeks_of_march():
    # Orange County's 2026 spring break was March 16-20
    first, last = break_windows(2026)["fl_spring_break"]
    assert first == date(2026, 3, 7) and last == date(2026, 3, 22)


def test_holiday_weekend_around_monday_holiday():
    f = features_for(date(2024, 8, 29), date(2024, 9, 3))
    # Labor Day 2024 was Monday Sept 2
    assert f.loc[date(2024, 9, 2), "holiday_name"] == "Labor Day"
    assert f.loc[date(2024, 8, 31), "is_holiday_weekend"]
    assert not f.loc[date(2024, 8, 30), "is_holiday_weekend"]
    assert f.loc[date(2024, 8, 31), "days_to_next_holiday"] == 2


def test_event_flag_covers_inclusive_range():
    f = features_for(date(2024, 8, 29), date(2024, 11, 4))
    assert not f.loc[date(2024, 8, 29), "event_hhn"]
    assert f.loc[date(2024, 8, 30), "event_hhn"]
    assert f.loc[date(2024, 11, 3), "event_hhn"]
    assert not f.loc[date(2024, 11, 4), "event_hhn"]


def test_winter_break_flag_crosses_year_boundary():
    f = features_for(date(2025, 1, 1), date(2025, 1, 10))
    assert f.loc[date(2025, 1, 3), "winter_break"]
    assert not f.loc[date(2025, 1, 6), "winter_break"]
