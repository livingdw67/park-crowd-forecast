import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "src"))

from parks_lib.queue_times import parse_calendar_month

# Trimmed copies of the queue-times calendar markup: an outside-month pad, a measured day with
# early entry and a holiday, a forecast day, and a closed day.
PAGE = """
<div class='cell is-background is-marginless is-paddingless is-flex is-week-start is-outside-month'>
<article class='is-fullheight box has-background-grey-lighter is-hidden-touch calendar-day'>
<div class='calendar-day-date'><span class='calendar-day-weekday'>Mon</span>
<span class='calendar-day-number'>30</span></div>
</article>
</div>
<div class='cell is-background is-marginless is-paddingless is-flex'>
<a class="box is-fullheight calendar-day" href="/parks/334/calendar/2026/07/04"><div class='calendar-day-date'>
<span class='calendar-day-number'>4</span>
<span class='calendar-day-icons'>
<span aria-label='Sunny day' class='calendar-day-icon' role='img' title='Sunny day'>x</span>
<span aria-label='Public holiday' class='calendar-day-icon' role='img' title='Public holiday'>x</span>
</span></div>
<div class='calendar-day-crowd'>
<span class='calendar-day-percent'>87<small>%</small></span>
<span class='calendar-day-level'>
Very busy
</span></div>
<div class='calendar-day-hours'>
<div class='calendar-day-period'><span role="img" aria-label="Extra Hours">x</span> 08:00-09:00</div>
<div class='calendar-day-period'><span aria-hidden="true">x</span> 09:00-22:00</div>
</div>
</a></div>
<div class='cell is-background is-marginless is-paddingless is-flex'>
<a class="box is-fullheight calendar-day" href="/parks/334/calendar/2026/07/05"><div class='calendar-day-date'>
<span class='calendar-day-number'>5</span></div>
<div class='calendar-day-crowd'>
<span class='calendar-day-percent'>45<small>%</small></span>
<span class='calendar-day-level'>
Quiet
<span class='calendar-day-forecast-inline'>&middot; Forecast</span>
</span></div>
<div class='calendar-day-hours'>
<div class='calendar-day-period'><span aria-hidden="true">x</span> 09:00-21:00</div>
<div class='calendar-day-forecast'>Forecast</div>
</div>
</a></div>
<div class='cell is-background is-marginless is-paddingless is-flex'>
<article class='is-fullheight box calendar-day is-closed'>
<div class='calendar-day-date'><span class='calendar-day-number'>6</span></div>
</article>
</div>
"""


def test_skips_days_from_neighbouring_months():
    days = parse_calendar_month(PAGE, 334, 2026, 7)
    assert [d["date"] for d in days] == [date(2026, 7, 4), date(2026, 7, 5), date(2026, 7, 6)]


def test_measured_day_with_early_entry_and_holiday():
    day = parse_calendar_month(PAGE, 334, 2026, 7)[0]
    assert day["crowd_percent"] == 87
    assert day["crowd_level"] == "Very busy"
    assert not day["is_forecast"]
    assert day["is_public_holiday"]
    assert day["weather_icons"] == ["Sunny day"]
    assert (day["open_time"], day["close_time"]) == ("09:00", "22:00")
    assert (day["extra_hours_open"], day["extra_hours_close"]) == ("08:00", "09:00")


def test_forecast_day():
    day = parse_calendar_month(PAGE, 334, 2026, 7)[1]
    assert day["is_forecast"]
    assert day["crowd_level"] == "Quiet"
    assert day["extra_hours_open"] is None


def test_redirected_month_returns_nothing():
    # Asking for a month the park has no data for serves a different month's page
    assert parse_calendar_month(PAGE, 334, 2018, 9) == []


def test_closed_day():
    day = parse_calendar_month(PAGE, 334, 2026, 7)[2]
    assert day["is_closed"]
    assert day["crowd_percent"] is None
    assert day["open_time"] is None
