"""The agents' tools: plain functions over ParkData, each returning a small JSON object.

Every number a tool returns is rounded the way an answer would quote it, so the validator can match
an answer's numbers against tool output. Staffing tools run the OR-Tools optimizer from parks_lib
in a worker thread, with the same park-hours rules as the daily planning job
(databricks/src/08_plan_staffing.py).
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from park_agent.data import PARKS, ParkData
from park_agent.types import Tool
from parks_lib.staffing import evaluate_plan, plan_staffing

PARK_IDS = {name: park_id for park_id, name in PARKS.items()}
ALIASES = {"usf": 65, "universal studios": 65, "ioa": 64, "islands": 64, "epic": 334, "volcano": 67}
SCHOOL_BREAKS = ["thanksgiving_week", "winter_break", "spring_break_season", "easter_week", "fl_spring_break",
                 "fl_summer", "us_summer"]
REGIME_FLAGS = {"epic_universe_open", "covid_closure"}    # in events.csv, but periods, not events
TYPICAL_ERROR_POINTS = 15      # the backtest's mean absolute error ran 12 to 19 points (README, Forecast)
FORECAST_MISS = 15             # the daily job replays each plan 15 points busier
EVENT_NIGHT_DAY_CLOSE, LATEST_CLOSE = 17, 23
MAX_DAYS = 31
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class ToolError(Exception):
    """A bad argument the model can fix (unknown park, date outside the forecast)."""


def park_id(name):
    if name in PARK_IDS:
        return PARK_IDS[name]
    key = (name or "").strip().lower()
    for alias, pid in ALIASES.items():
        if key == PARKS[pid].lower() or key.startswith(alias):
            return pid
    raise ToolError(f"unknown park {name!r}; use one of {list(PARK_IDS)}")


def parse_date(value):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ToolError(f"dates are YYYY-MM-DD, got {value!r}") from None


def park_hours(row):
    """Opening and closing hour as the planning job sets them: event nights listed past midnight
    close to day guests at 5pm, and nothing plans past 11pm."""
    def hour(t, default):
        return default if t is None or pd.isna(t) else int(str(t).split(":")[0])
    open_hour, close_hour = hour(row["open_time"], 9), hour(row["close_time"], 21)
    if close_hour <= open_hour:
        close_hour = EVENT_NIGHT_DAY_CLOSE
    return open_hour, min(close_hour, LATEST_CLOSE)


def plan_summary(plan):
    waits = [h["wait_minutes"] or 0 for h in plan.hourly]
    return {"status": plan.status, "workers": plan.total_workers, "labor_cost_usd": round(plan.labor_cost),
            "guest_hours_waiting": round(plan.wait_guest_hours), "worst_wait_minutes": round(max(waits, default=0)),
            "guests_who_skipped_a_ride": plan.skipped_guests}


@dataclass
class ParkTools:
    data: ParkData

    def __post_init__(self):
        self._plan = lru_cache(maxsize=256)(self._plan_uncached)

    # --- lookups ---------------------------------------------------------------------------------

    def _forecast_row(self, pid, day):
        rows = self.data.forecast[(self.data.forecast["park_id"] == pid) & (self.data.forecast["date"] == day)]
        if rows.empty:
            first, last = self.data.forecast_dates()[0], self.data.forecast_dates()[-1]
            raise ToolError(f"no forecast for {PARKS[pid]} on {day}; forecasts cover {first} to {last}")
        row = rows.iloc[0]
        if bool(row["is_closed"]):
            raise ToolError(f"{PARKS[pid]} is closed on {day}")
        return row

    def _drivers(self, day):
        cal = self.data.calendar.loc[day]
        events = [c.removeprefix("event_") for c in cal.index
                  if c.startswith("event_") and bool(cal[c]) and c.removeprefix("event_") not in REGIME_FLAGS]
        return {"weekday": WEEKDAYS[day.weekday()],
                "holiday": cal["holiday_name"] if isinstance(cal["holiday_name"], str) else None,
                "holiday_weekend": bool(cal["is_holiday_weekend"]),
                "school_breaks": [b for b in SCHOOL_BREAKS if b in cal.index and bool(cal[b])],
                "events": events}

    # --- forecast analyst ------------------------------------------------------------------------

    def get_forecast(self, park, start_date, end_date):
        start, end = parse_date(start_date), parse_date(end_date)
        if end < start or (end - start).days >= MAX_DAYS:
            raise ToolError(f"ask for 1 to {MAX_DAYS} days, start before end")
        ids = list(PARKS) if park == "all" else [park_id(park)]
        f = self.data.forecast
        rows = f[f["park_id"].isin(ids) & (f["date"] >= start) & (f["date"] <= end) & ~f["is_closed"].astype(bool)]
        if rows.empty:
            raise ToolError(f"no forecast in that range; forecasts cover {self.data.forecast_dates()[0]} "
                            f"to {self.data.forecast_dates()[-1]}")
        days = []
        for r in rows.sort_values(["date", "park_id"]).itertuples():
            qt = r.queue_times_forecast
            days.append({"park": PARKS[r.park_id], "date": str(r.date), "weekday": WEEKDAYS[r.date.weekday()],
                         "predicted_crowd": round(r.predicted_crowd),
                         "queue_times_forecast": None if pd.isna(qt) else round(qt),
                         **{k: v for k, v in self._drivers(r.date).items() if k != "weekday" and v}})
        return {"scale": "0-100 daily crowd score (queue-times)", "typical_error_points": TYPICAL_ERROR_POINTS,
                "days": days}

    def explain_day(self, park, date_):
        pid, day = park_id(park), parse_date(date_)
        row = self._forecast_row(pid, day)
        window = self.data.forecast[(self.data.forecast["park_id"] == pid)
                                    & (self.data.forecast["date"] >= self.data.today())
                                    & (self.data.forecast["date"] < self.data.today() + timedelta(days=30))
                                    & ~self.data.forecast["is_closed"].astype(bool)]
        ranked = window.sort_values("predicted_crowd", ascending=False)["date"].tolist()
        open_hour, close_hour = park_hours(row)
        return {"park": PARKS[pid], "date": str(day), "predicted_crowd": round(row["predicted_crowd"]),
                "typical_error_points": TYPICAL_ERROR_POINTS, "open_hour": open_hour, "close_hour": close_hour,
                **self._drivers(day),
                "rank_in_next_30_days": ranked.index(day) + 1 if day in ranked else None,
                "days_ranked": len(ranked), "average_next_30_days": round(window["predicted_crowd"].mean())}

    def get_epic_impact(self):
        s = self.data.impact_summary
        s = s[(s["role"] == "treated") | ((s["role"] == "orlando") & (s["variant"] == "main"))]
        rows = [{"park": r.park, "role": r.role, "variant": r.variant, "method": r.method,
                 "effect_points": round(r.effect_points, 1), "effect_pct": round(r.effect_pct, 1),
                 "placebo_p_value": round(r.p_value_effect, 2)} for r in s.itertuples()]
        return {"study": "Epic Universe opening (May 2025): difference-in-differences and synthetic control "
                         "against out-of-state parks, with placebo tests",
                "reading": "effect_points is the change in the 0-100 crowd score since the opening (negative = "
                           "fewer crowds); placebo_p_value is the share of placebo parks with an effect as large, "
                           "so with few comparison parks it can't go below about 0.12. Variant 'main' is the "
                           "headline; 'placebo_opening_2024_05' pretends Epic opened a year early and should "
                           "show no effect.",
                "results": rows}

    # --- staffing planner ------------------------------------------------------------------------

    def _plan_uncached(self, pid, crowd, open_hour, close_hour, headcount, fixed_from):
        locs = self.data.locations[self.data.locations["park_id"] == pid].to_dict("records")
        if fixed_from is not None:
            return evaluate_plan(self._plan(pid, *fixed_from), locs, crowd, open_hour, close_hour)
        return plan_staffing(locs, crowd, open_hour, close_hour, headcount=headcount)

    def get_staffing_plan(self, park, date_):
        pid, day = park_id(park), parse_date(date_)
        row = self._forecast_row(pid, day)
        crowd, (open_hour, close_hour) = round(row["predicted_crowd"]), park_hours(row)
        plan = self._plan(pid, crowd, open_hour, close_hour, None, None)
        busier = self._plan(pid, min(100, crowd + FORECAST_MISS), open_hour, close_hour, None,
                            (crowd, open_hour, close_hour, None, None))
        peaks = {}
        for h in plan.hourly:
            if h["wait_minutes"] is not None and h["wait_minutes"] > peaks.get(h["location"], (0, None))[0]:
                peaks[h["location"]] = (h["wait_minutes"], h["hour"])
        longest = sorted(peaks.items(), key=lambda kv: -kv[1][0])[:5]
        return {"park": PARKS[pid], "date": str(day), "predicted_crowd": crowd, "open_hour": open_hour,
                "close_hour": close_hour, "plan": plan_summary(plan),
                "longest_waits": [{"location": loc, "wait_minutes": round(w), "hour": hr} for loc, (w, hr) in longest],
                "if_15_points_busier": plan_summary(busier),
                "assumptions": "demand is modeled from the crowd score, not measured; staffing cost and wait "
                               "cost are planning assumptions"}

    def what_if(self, park, date_, crowd_score=None, crowd_change=None, open_hour=None, close_hour=None,
                max_workers=None):
        pid, day = park_id(park), parse_date(date_)
        row = self._forecast_row(pid, day)
        forecast = round(row["predicted_crowd"])
        planned_open, planned_close = park_hours(row)
        crowd = crowd_score if crowd_score is not None else forecast + (crowd_change or 0)
        crowd = max(0, min(100, round(crowd)))
        hours = (open_hour if open_hour is not None else planned_open,
                 close_hour if close_hour is not None else planned_close)
        if not 0 <= hours[0] < hours[1] <= 24:
            raise ToolError("open_hour must be before close_hour, both 0 to 24")
        published = (forecast, planned_open, planned_close, None, None)
        return {"park": PARKS[pid], "date": str(day), "forecast_crowd": forecast, "scenario_crowd": crowd,
                "scenario_hours": list(hours), "max_workers": max_workers,
                "published_plan_at_forecast": plan_summary(self._plan(pid, *published)),
                "published_plan_at_scenario": plan_summary(
                    self._plan(pid, crowd, planned_open, planned_close, None, published)),
                "replanned_for_scenario": plan_summary(self._plan(pid, crowd, *hours, max_workers, None))}

    # --- what the models see -----------------------------------------------------------------------

    def specs(self, names):
        park = {"type": "string", "enum": list(PARK_IDS), "description": "Park name"}
        day = {"type": "string", "description": "Date as YYYY-MM-DD"}
        all_specs = {
            "get_forecast": Tool("get_forecast", "Predicted daily crowd scores (0-100) for one park or all parks "
                                 "over 1 to 31 days, with holidays, school breaks and events on each day.",
                                 _schema({"park": {**park, "enum": [*PARK_IDS, "all"]}, "start_date": day,
                                          "end_date": day})),
            "explain_day": Tool("explain_day", "Why a park's day is forecast busy or quiet: weekday, holiday, "
                                "school breaks, events, park hours, and how the day ranks in the next 30 days.",
                                _schema({"park": park, "date": day})),
            "get_epic_impact": Tool("get_epic_impact", "Results of the study of how Epic Universe's opening "
                                    "changed crowds at the other parks.", _schema({})),
            "get_staffing_plan": Tool("get_staffing_plan", "The optimizer's crew plan for one park-day at the "
                                      "forecast crowd: workers, labor cost, waits, and the same plan if the day "
                                      "runs 15 points busier.", _schema({"park": park, "date": day})),
            "what_if": Tool("what_if", "Re-run staffing for a scenario: a different crowd score or change, park "
                            "hours, or a cap on workers. Compares the published plan with a re-plan.",
                            _schema({"park": park, "date": day,
                                     "crowd_score": {"type": ["integer", "null"], "description": "Scenario crowd score 0-100"},
                                     "crowd_change": {"type": ["integer", "null"], "description": "Points above (+) or below (-) the forecast"},
                                     "open_hour": {"type": ["integer", "null"]}, "close_hour": {"type": ["integer", "null"]},
                                     "max_workers": {"type": ["integer", "null"], "description": "Cap on total workers"}},
                                    required=["park", "date"])),
        }
        return [all_specs[n] for n in names]

    async def execute(self, name, args):
        """Run a tool for a model's tool loop: (JSON result, is_error)."""
        functions = {"get_forecast": self.get_forecast, "explain_day": self.explain_day,
                     "get_epic_impact": self.get_epic_impact, "get_staffing_plan": self.get_staffing_plan,
                     "what_if": self.what_if}
        if name not in functions:
            return json.dumps({"error": f"unknown tool {name!r}"}), True
        kwargs = {("date_" if k == "date" else k): v for k, v in (args or {}).items()}
        try:
            result = await asyncio.to_thread(functions[name], **kwargs)
        except ToolError as e:
            return json.dumps({"error": str(e)}), True
        except TypeError as e:
            return json.dumps({"error": f"bad arguments: {e}"}), True
        return json.dumps(result, default=str), False


def _schema(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or list(properties),
            "additionalProperties": False}
