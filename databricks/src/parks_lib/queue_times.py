"""Parsers for queue-times.com pages (Powered by Queue-Times.com, https://queue-times.com/)."""

import html
import re
from datetime import date

USER_AGENT = "park-crowd-forecast (portfolio project)"

# Each in-month day is an <a> (has data) or an <article> (closed or no data) with class "calendar-day".
# Days from the neighbouring months are padded in with "is-outside-month" on the wrapping cell.
_DAY = re.compile(
    r"<div class='cell[^']*?(?P<outside>is-outside-month)?'>\s*"
    r"<(?P<tag>a|article) (?:class=\"(?P<cls_a>[^\"]*)\"|class='(?P<cls_b>[^']*)')(?P<body>.*?)</(?P=tag)>",
    re.S,
)
_NUMBER = re.compile(r"calendar-day-number'>(\d+)<")
_PERCENT = re.compile(r"calendar-day-percent'>(\d+)<")
_LEVEL = re.compile(r"calendar-day-level'>\s*([^<]+?)\s*<")
_ICON = re.compile(r"class='calendar-day-icon'[^>]*title='([^']+)'")
_DAY_LINK = re.compile(r"/calendar/(\d{4})/(\d\d)/\d\d\"")
_PERIOD = re.compile(r"calendar-day-period'>(?P<icon>.*?)</span>\s*(?P<open>\d\d:\d\d)-(?P<close>\d\d:\d\d)</div>", re.S)


def parse_calendar_month(page, park_id, year, month):
    """One dict per day of the month on a queue-times crowd calendar page.

    crowd_percent is queue-times' own daily crowd score (0-100). is_forecast marks days where the
    score is their prediction rather than measured, which covers future days and the weeks before
    they started tracking a park. Upcoming days can have park hours and no score at all (today, and
    future days for parks queue-times doesn't forecast, like Epic Universe). Park hours are split into the regular day and any extra hours
    (early entry), and weather_icons holds the icon titles, which include "Public holiday".

    A month with no data for the park redirects to another month's page, so the month shown is
    checked against the one asked for and an empty list comes back when they differ.
    """
    shown = {(int(y), int(m)) for y, m in _DAY_LINK.findall(page)}
    if shown and (year, month) not in shown:
        return []
    days = []
    for m in _DAY.finditer(page):
        if m.group("outside"):
            continue
        body = m.group("body")
        number = _NUMBER.search(body)
        if not number:
            continue
        classes = m.group("cls_a") or m.group("cls_b") or ""
        percent = _PERCENT.search(body)
        level = _LEVEL.search(body)
        icons = [html.unescape(t) for t in _ICON.findall(body)]

        regular, extra = [], []
        for p in _PERIOD.finditer(body):
            (extra if "Extra Hours" in p.group("icon") else regular).append((p.group("open"), p.group("close")))

        days.append({
            "park_id": park_id,
            "date": date(year, month, int(number.group(1))),
            "is_closed": "is-closed" in classes,
            "crowd_percent": int(percent.group(1)) if percent else None,
            "crowd_level": html.unescape(level.group(1)) if level else None,
            "is_forecast": "calendar-day-forecast" in body,
            "is_public_holiday": "Public holiday" in icons,
            "weather_icons": [i for i in icons if i != "Public holiday"],
            "open_time": regular[0][0] if regular else None,
            "close_time": regular[-1][1] if regular else None,
            "extra_hours_open": extra[0][0] if extra else None,
            "extra_hours_close": extra[-1][1] if extra else None,
        })
    return days
