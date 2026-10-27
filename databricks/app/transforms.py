"""Pandas helpers behind the pages, kept free of Streamlit so they can be unit-tested."""

import json

import pandas as pd

from parks_lib.staffing import Plan

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TREND_SERIES = ["Forecast", "Queue-times forecast", "Measured"]

# Epic Universe impact study: the choices the page offers, mapped to the study's stored keys
EPIC_OPENING = pd.Timestamp("2025-05-22")
IMPACT_SERIES = ["Actual", "Without Epic (estimated)"]
IMPACT_UNITS = {"Universal Studios Florida": 65, "Islands of Adventure": 64, "Both parks, averaged": 6564}
IMPACT_METHODS = {"Synthetic control": "synthetic_control", "Difference-in-differences": "diff_in_diff"}
IMPACT_GROUPS = {"Seven parks outside Florida": "main", "Without the Disneyland parks": "without_disneyland",
                 "Plus Busch Gardens Tampa": "with_busch_gardens_tampa"}


def hour_label(hour):
    return f"{hour % 12 or 12}{'am' if hour % 24 < 12 else 'pm'}"


def day_label(day):
    day = pd.Timestamp(day)
    return f"{day:%a, %b} {day.day}"  # no %-d: Windows strftime doesn't have it


def stamp(ts):
    """A run timestamp in park time, e.g. 'Oct 22, 2:11pm ET'."""
    t = pd.Timestamp(ts)
    t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert("America/New_York")
    return f"{t:%b} {t.day}, {hour_label(t.hour)[:-2]}:{t:%M}{hour_label(t.hour)[-2:]} ET"


def signed(delta, prefix="", suffix=""):
    """A metric delta: zero stays the number 0 (grey, no arrow), anything else gets an explicit sign."""
    if round(delta) == 0:
        return 0
    return f"{'+' if delta > 0 else '-'}{prefix}{abs(delta):,.0f}{suffix}"


def calendar_grid(forecast, park_id, start):
    """One park's forecast from `start` laid out as a calendar: a column per week (from Monday), a row per weekday."""
    df = forecast[(forecast["park_id"] == park_id) & (forecast["date"] >= pd.Timestamp(start))].copy()
    df["weekday"] = df["date"].dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
    df["week"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
    df["week_label"] = df["week"].dt.strftime("%b ") + df["week"].dt.day.astype(str)
    df["day"] = df["date"].dt.day
    return df.sort_values("date")


def trend_frame(forecast, measured, park_id):
    """Long frame for the trend chart: measured scores, our forecast and queue-times' forecast."""
    f = forecast[forecast["park_id"] == park_id]
    m = measured[measured["park_id"] == park_id]
    parts = [
        f.assign(series="Forecast", crowd=f["predicted_crowd"]),
        f.assign(series="Queue-times forecast", crowd=f["queue_times_forecast"]),
        m.assign(series="Measured", crowd=m["measured_crowd"]),
    ]
    parts = [p[["date", "series", "crowd"]].dropna() for p in parts]
    return pd.concat([p for p in parts if len(p)], ignore_index=True).sort_values("date")


def published_plan(shifts, park_id, day):
    """The stored plan for one park-day as a Plan, so evaluate_plan can replay its shifts."""
    rows = shifts[(shifts["park_id"] == park_id) & (shifts["date"] == pd.Timestamp(day))]
    return Plan(status="published", shifts=rows[["location", "start", "end", "workers"]].to_dict("records"))


def headline(plan):
    """The plan-level numbers the daily job stores in staffing_plan_summary."""
    return {
        "workers": plan.total_workers,
        "labor_cost": plan.labor_cost,
        "wait_guest_hours": plan.wait_guest_hours,
        "worst_wait_minutes": max((h["wait_minutes"] or 0) for h in plan.hourly),
        "skipped_guests": plan.skipped_guests,
    }


def by_hour(hourly):
    """Park-wide view of hourly plan rows: crew on duty and average wait across locations."""
    return (pd.DataFrame(hourly).groupby("hour", as_index=False)
            .agg(crew=("crew", "sum"), avg_wait=("wait_minutes", "mean")))


def at_max_crew(hourly, locations):
    """Locations a plan runs at their maximum crew for at least one hour: extra staff can't shorten those lines."""
    rows = pd.DataFrame(hourly).merge(locations[["location", "max_crew"]], on="location")
    return sorted(rows.loc[rows["crew"] >= rows["max_crew"], "location"].unique())


def location_order(locations):
    """Rides first, then food, busiest first within each: the order the heatmaps list them in."""
    return locations.sort_values(["kind", "daily_share"], ascending=[False, False])["location"].tolist()


def impact_row(summary, park_id, method, variant):
    """The study's summary row for one park, method and comparison group (None if it wasn't run)."""
    rows = summary[(summary["park_id"] == park_id) & (summary["method"] == method) & (summary["variant"] == variant)]
    return rows.iloc[0] if len(rows) else None


def impact_frame(gaps, park_id, method, variant):
    """Long frame for the impact chart: the park's actual monthly score and the no-Epic estimate."""
    g = gaps[(gaps["park_id"] == park_id) & (gaps["method"] == method) & (gaps["variant"] == variant)]
    parts = [g.assign(series=IMPACT_SERIES[0], value=g["actual"]), g.assign(series=IMPACT_SERIES[1], value=g["counterfactual"])]
    return pd.concat(parts, ignore_index=True)[["month", "series", "value"]].dropna().sort_values("month")


def placebo_frame(row):
    """The park's effect beside each comparison park's effect when it pretends to be the treated one."""
    placebos = [(name, effect, False) for name, effect in json.loads(row["placebo_effects"]).items()]
    frame = pd.DataFrame(placebos + [(row["park"], row["effect_points"], True)], columns=["park", "effect", "is_treated"])
    return frame.sort_values("effect", ignore_index=True)


def as_large_as(row):
    """How many placebo parks moved at least as much as the park: the p-value as a count, (1 + k) / (n + 1)."""
    n = int(row["n_donors"])
    return round(row["p_value_effect"] * (n + 1)) - 1, n
