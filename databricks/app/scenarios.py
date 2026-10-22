"""What-if runs of the staffing optimizer, cached so revisiting a scenario doesn't re-solve it."""

import streamlit as st

from parks_lib.staffing import Plan, evaluate_plan, plan_staffing

SCENARIOS = ["Published plan", "Published plan, scenario crowd", "Re-planned"]


@st.cache_data(max_entries=64, show_spinner=False)
def run(locations, published_shifts, forecast_crowd, planned_hours, crowd, hours, assumptions, headcount):
    """Three views of one park-day, all costed with the same assumptions.

    - Published plan: the daily job's shifts at the forecast crowd
    - Published plan, scenario crowd: the same shifts if the day turns out like the scenario
      (hours stay as published, since the shifts were built for them)
    - Re-planned: the optimizer's shifts for the scenario's crowd, hours, costs and worker cap
    """
    locs = locations.to_dict("records")
    published = Plan(status="published", shifts=published_shifts.to_dict("records"))
    return dict(zip(SCENARIOS, [
        evaluate_plan(published, locs, forecast_crowd, *planned_hours, assumptions),
        evaluate_plan(published, locs, crowd, *planned_hours, assumptions),
        plan_staffing(locs, crowd, *hours, assumptions, headcount),
    ]))
