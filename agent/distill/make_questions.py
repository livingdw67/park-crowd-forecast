"""Template questions for distilling the writer, with their tool calls run directly (no model).

    .venv\\Scripts\\python agent\\distill\\make_questions.py

Each example is what the writer sees in a real turn: the question, today's context and the tool
results, built with the harness's own prompt code. The specialists' notes are left empty, because no
model runs here; the tools are called the way a specialist would call them for that question type.

Held-out split by week: three of the forecast's weeks are held out, and every question about a date
in them (and ranges never cross a week) goes to `heldout`, so a held-out answer never shares tool
results with a training example. Epic-study and out-of-range questions have no week and are split at
random. Writes agent/distill/data/questions.jsonl (git-ignored: the numbers come from queue-times data).
"""

import asyncio
import json
import random
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from park_agent.data import PARKS, ParkData  # noqa: E402
from park_agent.harness import Harness, Turn, _context  # noqa: E402
from park_agent.tools import PARK_IDS, ParkTools  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "questions.jsonl"
SEED = 7
HELDOUT_WEEKS = 3
ALL = list(PARKS.values())
STAFFED = ["Universal Studios Florida", "Islands of Adventure", "Epic Universe"]


def say(d, today):
    """A date the way people ask about it."""
    if d == today + timedelta(days=1):
        return "tomorrow"
    return random.choice([f"on {d:%A}, {d:%B} {d.day}", f"on {d:%B} {d.day}", f"on {d.isoformat()}"])


def fc(park, start, end=None):
    return ("forecast", "get_forecast", {"park": park, "start_date": str(start), "end_date": str(end or start)})


def ex(park, d):
    return ("forecast", "explain_day", {"park": park, "date": str(d)})


def sp(park, d):
    return ("staffing", "get_staffing_plan", {"park": park, "date": str(d)})


def wi(park, d, **kw):
    return ("staffing", "what_if", {"park": park, "date": str(d), **kw})


def templates(tools, today, week):
    """Every dated template for one week: name -> (route, question, tool calls)."""
    days = week
    d = random.choice(days)
    f = tools.data.forecast
    # A park without a forecast that day stays possible for `p` (the tool error is a real case),
    # but the staffing templates need a plan, so `s` is drawn from parks that have one.
    s = random.choice([x for x in STAFFED if ((f["park_id"] == PARK_IDS[x]) & (f["date"] == d)).any()])
    p = random.choice(ALL)
    when = say(d, today)
    span = sorted(random.sample(days, k=min(len(days), random.randint(2, 4))))
    a, b = span[0], span[-1]
    p2 = random.choice([x for x in ALL if x != p])
    plan = tools.get_staffing_plan(s, str(d))
    hours = tools.explain_day(s, str(d))
    return {
        "forecast_day": ("forecast", random.choice([
            f"How busy will {p} be {when}?", f"What crowds should we expect at {p} {when}?",
            f"Is {p} going to be crowded {when}?"]), [fc(p, d), ex(p, d)]),
        "quietest_in_range": ("forecast", random.choice([
            f"Which day from {a:%B} {a.day} to {b:%B} {b.day} is quietest at {p}?",
            f"I can visit {p} any day between {a:%A} and {b:%A} ({a.isoformat()} to {b.isoformat()}). Which is least crowded?"]),
            [fc(p, a, b)]),
        "busiest_in_range": ("forecast", f"What's the busiest day at {p} between {a:%B} {a.day} and {b:%B} {b.day}?",
                             [fc(p, a, b)]),
        "compare_parks": ("forecast", random.choice([
            f"Compare {p} and {p2} {when}.", f"Will {p} or {p2} be busier {when}?"]), [fc("all", d)]),
        "least_crowded_park": ("forecast", random.choice([
            f"Which park is least crowded {when}?", f"Which Universal Orlando park should we pick {when} to avoid crowds?"]),
            [fc("all", d)]),
        "why_day": ("forecast", random.choice([
            f"Why is {p} forecast the way it is {when}?", f"What's driving the crowd forecast for {p} {when}?"]),
            [ex(p, d)]),
        "reliability": ("forecast", random.choice([
            f"How much can I trust the crowd forecast for {p} {when}?", f"How accurate is the forecast for {p} {when}?"]),
            [ex(p, d), fc(p, d)]),
        "staffing_plan": ("staffing", random.choice([
            f"What's the staffing plan for {s} {when}?", f"How many workers does {s} need {when}?"]), [sp(s, d)]),
        "labor_cost": ("staffing", f"What will labor cost at {s} {when} under the staffing plan?", [sp(s, d)]),
        "longest_waits": ("staffing", f"Where are the longest waits in the plan for {s} {when}?", [sp(s, d)]),
        "busier_than_forecast": ("staffing", *(lambda n: (
            f"If {s} runs {n} points busier than forecast {when}, what happens with the published plan?",
            [wi(s, d, crowd_change=n)]))(random.choice([5, 10, 15, 20, 25]))),
        "quieter_than_forecast": ("staffing", *(lambda n: (
            f"If {s} is {n} points quieter than forecast {when}, could we staff fewer people?",
            [wi(s, d, crowd_change=-n)]))(random.choice([10, 15, 20]))),
        "crowd_score": ("staffing", *(lambda c: (
            f"What would it cost to staff {s} for a crowd score of {c} {when}?", [wi(s, d, crowd_score=c)]))(
            random.choice([30, 45, 60, 70, 80, 90]))),
        "worker_cap": ("staffing", *(lambda w: (
            f"What happens to waits at {s} {when} if we can only staff {w} workers?", [wi(s, d, max_workers=w)]))(
            max(10, round(plan["plan"]["workers"] * random.choice([0.7, 0.8, 0.9]))))),
        "later_open": ("staffing", *(lambda h: (
            f"If {s} opens {h} hour{'s' if h > 1 else ''} later {when}, how does the staffing plan change?",
            [wi(s, d, open_hour=hours["open_hour"] + h)]))(random.choice([1, 2]))),
        "both": ("both", random.choice([
            f"How busy is {s} {when}, and how many staff will it need?",
            f"Give me the crowd forecast and the staffing plan for {s} {when}."]), [fc(s, d), ex(s, d), sp(s, d)]),
        "no_staffing_model": ("staffing", f"What's the staffing plan for Volcano Bay {when}?",
                              [sp("Volcano Bay", d)]),
    }


def undated(tools, last):
    """Questions with no week: the Epic study, and dates past the forecast."""
    p = random.choice(ALL)
    far = last + timedelta(days=random.randint(10, 200))
    return {
        "epic_impact": ("forecast", random.choice([
            "Did Epic Universe's opening reduce crowds at Universal Studios Florida?",
            "How did Epic Universe change crowds at Islands of Adventure?",
            "What did the Epic Universe impact study find?",
            "Has Epic Universe pulled visitors away from the older Universal parks?",
            "How confident is the study that Epic Universe made Universal Studios Florida quieter?"]),
            [("forecast", "get_epic_impact", {})]),
        "out_of_range": ("forecast", f"How busy will {p} be on {far:%B} {far.day}, {far.year}?",
                         [fc(p, far), ex(p, far)]),
    }


async def build(tools, question, calls, route):
    turn = Turn("distill", question, route=route)
    for agent, name, args in calls:
        result, is_error = await tools.execute(name, args)
        turn.tool_calls.append({"agent": agent, "name": name, "args": args, "result": result, "is_error": is_error})
    return turn


async def main(per_week=40, undated_count=60):
    random.seed(SEED)
    data = ParkData.load()
    tools = ParkTools(data)
    today, dates = data.today(), data.forecast_dates()
    context = _context(data)
    weeks = {}
    for d in dates:
        weeks.setdefault(d - timedelta(days=d.weekday()), []).append(d)
    week_starts = sorted(weeks)
    heldout_weeks = set(random.sample(week_starts[1:-1], HELDOUT_WEEKS))   # full weeks only

    rows, seen = [], set()
    jobs = [(w, None) for w in week_starts for _ in range(per_week)] + [(None, None)] * undated_count
    for week_start, _ in jobs:
        if week_start is None:
            options, split = undated(tools, dates[-1]), ("heldout" if random.random() < 0.2 else "train")
        else:
            options, split = templates(tools, today, weeks[week_start]), (
                "heldout" if week_start in heldout_weeks else "train")
        template = random.choice(list(options))
        route, question, calls = options[template]
        if question in seen:
            continue
        seen.add(question)
        turn = await build(tools, question, calls, route)
        rows.append({"id": f"q{len(rows):04d}", "split": split, "template": template, "route": route,
                     "question": question, "week": str(week_start) if week_start else None,
                     "tool_calls": [{k: c[k] for k in ("agent", "name", "args", "is_error")} for c in turn.tool_calls],
                     "writer_prompt": Harness._writer_prompt(question, context, turn)})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    by_split = {s: sum(r["split"] == s for r in rows) for s in ("train", "heldout")}
    print(f"{len(rows)} questions {by_split}; held-out weeks {sorted(map(str, heldout_weeks))}")
    print({t: sum(r["template"] == t for r in rows) for t in sorted({r["template"] for r in rows})})


if __name__ == "__main__":
    asyncio.run(main())
