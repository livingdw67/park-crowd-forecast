"""Date features for the crowd forecast: holidays, school-break windows and events.

Everything here is known in advance, so a feature row can be built for any future date. School
breaks are rule-based approximations (visitors come from many districts, so exact single-district
dates would add little); they are checked against TouringPlans' Florida in-session data in the
README.
"""

from datetime import date, timedelta

import pandas as pd
from dateutil.easter import easter
from pandas.tseries.holiday import USFederalHolidayCalendar


def _saturday_on_or_before(d):
    return d - timedelta(days=(d.weekday() - 5) % 7)


def _sunday_on_or_after(d):
    return d + timedelta(days=(6 - d.weekday()) % 7)


def _thanksgiving(year):
    nov1 = date(year, 11, 1)
    return nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21)


def _memorial_day(year):
    may31 = date(year, 5, 31)
    return may31 - timedelta(days=may31.weekday())


def _labor_day(year):
    sep1 = date(year, 9, 1)
    return sep1 + timedelta(days=(0 - sep1.weekday()) % 7)


def break_windows(year):
    """School-break windows that start in `year`, as {name: (first_day, last_day)}."""
    thanksgiving = _thanksgiving(year)
    easter_sunday = easter(year)
    mar1 = date(year, 3, 1)
    second_monday_march = mar1 + timedelta(days=(0 - mar1.weekday()) % 7 + 7)
    return {
        "thanksgiving_week": (_saturday_on_or_before(thanksgiving), thanksgiving + timedelta(days=3)),
        "winter_break": (_saturday_on_or_before(date(year, 12, 21)), _sunday_on_or_after(date(year + 1, 1, 2))),
        # Districts spread spring break from early March to the week after Easter
        "spring_break_season": (date(year, 3, 7), max(date(year, 4, 15), easter_sunday + timedelta(days=8))),
        "easter_week": (easter_sunday - timedelta(days=8), easter_sunday + timedelta(days=1)),
        # Florida districts take spring break in the second or third full week of March, not at Easter
        "fl_spring_break": (_saturday_on_or_before(second_monday_march), second_monday_march + timedelta(days=13)),
        # Florida schools let out around Memorial Day and go back around Aug 10
        "fl_summer": (_memorial_day(year), date(year, 8, 9)),
        # Most other states are out from mid-June to Labor Day
        "us_summer": (date(year, 6, 15), _labor_day(year)),
    }


def build_calendar_features(start, end, events):
    """One row per date from start to end with holiday, school-break and event flags.

    `events` is a DataFrame with columns event, start_date, end_date (dates, inclusive). Each
    distinct event becomes an `event_<name>` flag column.
    """
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": dates.date})
    df["day_of_week"] = dates.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5
    df["month"] = dates.month
    df["day_of_year"] = dates.dayofyear
    df["week_of_year"] = dates.isocalendar().week.to_numpy()

    holidays = USFederalHolidayCalendar().holidays(start - timedelta(days=60), end + timedelta(days=60), return_name=True)
    holiday_names = {d.date(): name for d, name in holidays.items()}
    for year in range(start.year - 1, end.year + 2):
        holiday_names[easter(year)] = "Easter Sunday"
    df["holiday_name"] = df["date"].map(holiday_names)
    df["is_holiday"] = df["holiday_name"].notna()

    # Long weekends: Saturday-Monday around a Monday holiday, Friday-Sunday around a Friday holiday
    long_weekend = set()
    for d in holiday_names:
        if d.weekday() == 0:
            long_weekend |= {d - timedelta(days=2), d - timedelta(days=1), d}
        elif d.weekday() == 4:
            long_weekend |= {d, d + timedelta(days=1), d + timedelta(days=2)}
    df["is_holiday_weekend"] = df["date"].isin(long_weekend)

    holiday_days = pd.Series(sorted(holiday_names))
    df["days_to_next_holiday"] = [min(((h - d).days for h in holiday_days if h >= d), default=None) for d in df["date"]]
    df["days_since_last_holiday"] = [min(((d - h).days for h in holiday_days if h <= d), default=None) for d in df["date"]]

    for year in range(start.year - 1, end.year + 1):
        for name, (first, last) in break_windows(year).items():
            if name not in df:
                df[name] = False
            df.loc[(df["date"] >= first) & (df["date"] <= last), name] = True

    for event, rows in events.groupby("event"):
        col = f"event_{event}"
        df[col] = False
        for first, last in zip(rows["start_date"], rows["end_date"]):
            df.loc[(df["date"] >= first) & (df["date"] <= last), col] = True
    return df
