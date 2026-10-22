"""Staffing optimizer: turns a day's crowd forecast into crew counts per location and shift.

A mixed-integer program (OR-Tools, SCIP). For each location (ride or food outlet) it chooses how
many workers start each shift, trading wages against time guests spend in line:

- crew on duty each hour = workers whose shift covers that hour, between min_crew and max_crew
- guests served per hour <= crew x guests_per_crew_hour
- the line carries over: line[h] = line[h-1] + joined[h] - served[h]
- guests skip long lines: joined[h] = arrivals[h] x (1 - line[h-1] / full_skip_line), where the
  full-skip line is `full_skip_wait_hours` of the location's top capacity. Linear, since arrivals
  are fixed, and it keeps lines from growing without limit on days demand exceeds capacity
- a soft wait target: line[h] <= target wait x hourly capacity, with a large penalty per guest-hour over
- optional: total workers across the park <= headcount

Demand comes from assumptions, not data (see ASSUMPTIONS and seeds/staffing_locations.csv):
the crowd score maps to attendance per open hour, each location sees a share of the day's guests,
and arrivals peak early afternoon. evaluate_plan replays a fixed plan against a different crowd level.
"""

import math
from dataclasses import dataclass, field

ASSUMPTIONS = {
    "attendance_at_0": 25000,      # guests on a crowd score of 0
    "attendance_at_100": 50000,    # guests on a crowd score of 100
    "reference_hours": 12,         # attendance figures are for a 12-hour day; the crowd score tracks
                                   # how busy each open hour is, so guests scale with hours open
    "peak_position": 0.45,         # arrivals peak this far into the operating day
    "peak_spread": 0.30,           # width of the arrival curve, as a share of the day
    "wage_per_hour": 25.0,         # loaded labor cost
    "wait_cost_per_guest_hour": 8.0,
    "target_wait_minutes": 60,
    "over_target_cost_per_guest_hour": 50.0,
    "full_skip_wait_hours": 4,     # nobody joins a line this long; half skip at half of it
    "full_shift_hours": 8,
    "part_shift_hours": 4,
    "shift_start_step_hours": 2,
}


@dataclass
class Plan:
    status: str
    shifts: list = field(default_factory=list)   # {location, start, end, workers}
    hourly: list = field(default_factory=list)   # {location, hour, crew, arrivals, served, line, wait_minutes}
    labor_cost: float = 0.0
    wait_guest_hours: float = 0.0
    skipped_guests: int = 0
    over_target_guest_hours: float = 0.0
    total_workers: int = 0


def attendance(crowd_score, a=ASSUMPTIONS):
    return a["attendance_at_0"] + (a["attendance_at_100"] - a["attendance_at_0"]) * crowd_score / 100


def arrivals_by_hour(daily_guests, open_hour, close_hour, a=ASSUMPTIONS):
    hours = list(range(open_hour, close_hour))
    n = len(hours)
    weights = [math.exp(-0.5 * (((i + 0.5) / n - a["peak_position"]) / a["peak_spread"]) ** 2) for i in range(n)]
    total = sum(weights)
    return {h: daily_guests * w / total for h, w in zip(hours, weights)}


def shift_templates(open_hour, close_hour, a=ASSUMPTIONS):
    """(start, end) pairs: full shifts every `shift_start_step_hours`, part shifts every hour."""
    shifts = []
    for length, step in ((a["full_shift_hours"], a["shift_start_step_hours"]), (a["part_shift_hours"], 1)):
        starts = list(range(open_hour, close_hour - length + 1, step))
        if starts and starts[-1] != close_hour - length:
            starts.append(close_hour - length)
        shifts += [(s, s + length) for s in starts]
    return sorted(set(shifts))


def _solve(locations, crowd_score, open_hour, close_hour, a, headcount, fixed_shifts):
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("SCIP")
    hours = list(range(open_hour, close_hour))
    shifts = shift_templates(open_hour, close_hour, a)
    guests = attendance(crowd_score, a) * len(hours) / a["reference_hours"]
    target = a["target_wait_minutes"] / 60

    x, crew, served, line, over, demand, joined = {}, {}, {}, {}, {}, {}, {}
    for loc in locations:
        name = loc["location"]
        demand[name] = arrivals_by_hour(guests * loc["daily_share"], open_hour, close_hour, a)
        for s in shifts:
            if fixed_shifts is None:
                x[name, s] = solver.IntVar(0, loc["max_crew"], f"x[{name},{s}]")
            else:
                x[name, s] = fixed_shifts.get((name, s), 0)
        full_skip_line = a["full_skip_wait_hours"] * loc["guests_per_crew_hour"] * loc["max_crew"]
        previous = 0
        for h in hours:
            crew[name, h] = sum(x[name, s] for s in shifts if s[0] <= h < s[1])
            if fixed_shifts is None:
                solver.Add(crew[name, h] >= loc["min_crew"])
                solver.Add(crew[name, h] <= loc["max_crew"])
            capacity = loc["guests_per_crew_hour"] * crew[name, h]
            served[name, h] = solver.NumVar(0, solver.infinity(), "")
            line[name, h] = solver.NumVar(0, solver.infinity(), "")
            over[name, h] = solver.NumVar(0, solver.infinity(), "")
            solver.Add(served[name, h] <= capacity)
            joined[name, h] = demand[name][h] * (1 - previous / full_skip_line) if h > open_hour else demand[name][h]
            solver.Add(served[name, h] <= previous + joined[name, h])
            solver.Add(line[name, h] == previous + joined[name, h] - served[name, h])
            solver.Add(line[name, h] <= target * capacity + over[name, h])
            previous = line[name, h]

    if headcount is not None and fixed_shifts is None:
        solver.Add(sum(x.values()) <= headcount)

    labor = a["wage_per_hour"] * sum(x[n, s] * (s[1] - s[0]) for (n, s) in x)
    waiting = sum(line.values())
    overflow = sum(over.values())
    solver.Minimize(labor + a["wait_cost_per_guest_hour"] * waiting + a["over_target_cost_per_guest_hour"] * overflow)
    solver.SetTimeLimit(30_000)
    status = solver.Solve()
    names = {pywraplp.Solver.OPTIMAL: "optimal", pywraplp.Solver.FEASIBLE: "feasible"}
    if status not in names:
        return Plan(status="infeasible")

    val = lambda v: v.solution_value() if hasattr(v, "solution_value") else float(v)
    plan = Plan(status=names[status])
    for (n, s), v in x.items():
        workers = round(val(v))
        if workers:
            plan.shifts.append({"location": n, "start": s[0], "end": s[1], "workers": workers})
    for loc in locations:
        n = loc["location"]
        for h in hours:
            crew_h = round(sum(val(x[n, s]) for s in shifts if s[0] <= h < s[1]))
            capacity = loc["guests_per_crew_hour"] * crew_h
            plan.hourly.append({
                "location": n, "hour": h, "crew": crew_h, "arrivals": round(demand[n][h]),
                "skipped": round(demand[n][h] - val(joined[n, h])),
                "served": round(val(served[n, h])), "line": round(val(line[n, h])),
                "wait_minutes": round(60 * val(line[n, h]) / capacity) if capacity else None,
            })
    plan.labor_cost = round(sum(sh["workers"] * (sh["end"] - sh["start"]) for sh in plan.shifts) * a["wage_per_hour"], 2)
    plan.wait_guest_hours = round(sum(val(v) for v in line.values()))
    plan.skipped_guests = sum(h["skipped"] for h in plan.hourly)
    plan.over_target_guest_hours = round(sum(val(v) for v in over.values()))
    plan.total_workers = sum(sh["workers"] for sh in plan.shifts)
    return plan


def plan_staffing(locations, crowd_score, open_hour=9, close_hour=21, assumptions=None, headcount=None):
    """Lowest-cost crew plan for one park-day. `locations` are rows of staffing_locations.csv."""
    return _solve(locations, crowd_score, open_hour, close_hour, {**ASSUMPTIONS, **(assumptions or {})}, headcount, None)


def evaluate_plan(plan, locations, crowd_score, open_hour=9, close_hour=21, assumptions=None):
    """Replay a fixed plan's shifts against a different crowd score (e.g. the forecast was off)."""
    fixed = {(sh["location"], (sh["start"], sh["end"])): sh["workers"] for sh in plan.shifts}
    return _solve(locations, crowd_score, open_hour, close_hour, {**ASSUMPTIONS, **(assumptions or {})}, None, fixed)


def normalize_name(name):
    """Ride names without trademark symbols or punctuation, for matching queue-times names."""
    return " ".join("".join(ch for ch in name.lower() if ch.isalnum() or ch.isspace()).split())


def calibrate_share(location, observed_wait_minutes, crowd_score, open_hour=9, close_hour=21,
                    window=(11, 17), assumptions=None, low=0.02, high=3.0, steps=25):
    """Demand share at which the plan's average wait over `window` hours matches an observed wait.

    Bisection: more demand means longer lines once crew is maxed out, so the planned wait rises
    with the share. Posted waits run higher than actual ones, so this calibrates to what guests see.
    """
    for _ in range(steps):
        mid = (low + high) / 2
        plan = plan_staffing([{**location, "daily_share": mid}], crowd_score, open_hour, close_hour, assumptions)
        waits = [h["wait_minutes"] or 0 for h in plan.hourly if window[0] <= h["hour"] < window[1]]
        if sum(waits) / len(waits) < observed_wait_minutes:
            low = mid
        else:
            high = mid
    return round((low + high) / 2, 3)
