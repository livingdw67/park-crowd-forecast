import json
import pytest

from park_agent.tools import ToolError

def test_forecast_rows_carry_calendar_drivers(tools):
    out = tools.get_forecast("Universal Studios Florida", "2026-11-26", "2026-11-28")
    thanksgiving = out["days"][0]
    assert thanksgiving["holiday"] == "Thanksgiving Day" and "thanksgiving_week" in thanksgiving["school_breaks"]
    assert [d["predicted_crowd"] for d in out["days"]] == [50, 71, 52]
    assert out["typical_error_points"] == 15


def test_forecast_for_all_parks_and_bad_ranges(tools):
    assert {d["park"] for d in tools.get_forecast("all", "2026-11-21", "2026-11-21")["days"]} == {
        "Universal Studios Florida", "Islands of Adventure"}
    with pytest.raises(ToolError, match="forecasts cover"):
        tools.get_forecast("Islands of Adventure", "2027-03-01", "2027-03-02")
    with pytest.raises(ToolError, match="1 to 31 days"):
        tools.get_forecast("Islands of Adventure", "2026-11-25", "2026-11-20")


def test_explain_day_ranks_the_busiest_day(tools):
    out = tools.explain_day("universal studios", "2026-11-27")
    assert out["predicted_crowd"] == 71 and out["rank_in_next_30_days"] == 1 and out["weekday"] == "Friday"
    assert out["holiday_weekend"] is False and "thanksgiving_week" in out["school_breaks"]


def test_staffing_plan_and_what_if_run_the_optimizer(tools):
    plan = tools.get_staffing_plan("Islands of Adventure", "2026-11-23")
    assert plan["plan"]["status"] in ("optimal", "feasible") and plan["plan"]["workers"] > 0
    assert len(plan["longest_waits"]) <= 5
    busier = tools.what_if("Islands of Adventure", "2026-11-23", crowd_change=30)
    assert busier["scenario_crowd"] == busier["forecast_crowd"] + 30
    # the published plan, unchanged, waits longer on the busier day than a plan built for it
    assert (busier["published_plan_at_scenario"]["guest_hours_waiting"]
            >= busier["replanned_for_scenario"]["guest_hours_waiting"])


@pytest.mark.anyio
async def test_execute_returns_tool_errors_to_the_model(tools):
    result, is_error = await tools.execute("explain_day", {"park": "Disneyland", "date": "2026-11-27"})
    assert is_error and "unknown park" in json.loads(result)["error"]
    result, is_error = await tools.execute("get_epic_impact", {})
    assert not is_error and json.loads(result)["results"][0]["effect_points"] == -14.2
    result, is_error = await tools.execute("what_if", {"park": "Volcano Bay", "date": "2026-11-27", "crowd_change": 10})
    assert is_error and "no staffing model for Volcano Bay" in json.loads(result)["error"]
