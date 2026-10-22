import pandas as pd
import streamlit as st

from parks_lib.staffing import ASSUMPTIONS

import charts
import controls
import park_data
import scenarios
import transforms

data = park_data.load()
locations = park_data.load_locations()
if data["summary"].empty:
    st.warning("No staffing plan yet. The daily-features job writes one at 6:30am Eastern.", icon=":material/schedule:")
    st.stop()

name, park_id, day, plan = controls.park_and_day(data["summary"], locations)
if plan is None:
    st.info(f"No plan for {name} on {transforms.day_label(day)}: the park is closed or has no forecast.",
            icon=":material/event_busy:")
    st.stop()

park_locations = locations[locations["park_id"] == park_id]
forecast_crowd = float(plan.predicted_crowd)
planned_hours = (int(plan.open_hour), int(plan.close_hour))
st.caption(f"Change the day and the costs, then compare the crew the daily job published (planned for a forecast "
           f"crowd of {forecast_crowd:.0f}) with what the optimizer would schedule for your scenario.")

controls_col, results = st.columns([1, 2.4], gap="large")
with controls_col, st.form("scenario"):
    # Keyed by park-day so the crowd and hours start from that day's plan whenever the selection changes
    suffix = f"{park_id}_{day}"
    crowd = st.slider("Crowd score", 0, 100, min(100, round(forecast_crowd) + 15), key=f"crowd_{suffix}",
                      help="Starts at the forecast plus 15 points, about one typical forecast error.")
    hours = st.select_slider("Hours open (re-plan only)", list(range(6, 24)), value=planned_hours,
                             format_func=transforms.hour_label, key=f"hours_{suffix}")
    cap = st.number_input("Worker cap", min_value=1, value=None, step=5, placeholder="No cap", key="cap",
                          help="Most workers the re-plan may schedule across the park.")
    wage = st.slider("Wage, $ per hour", 15, 45, int(ASSUMPTIONS["wage_per_hour"]), key="wage")
    wait_cost = st.slider("Cost of a guest-hour in line, $", 0, 30, int(ASSUMPTIONS["wait_cost_per_guest_hour"]),
                          key="wait_cost", help="What an hour of a guest's time in line is worth to the park.")
    target = st.slider("Wait target, minutes", 15, 120, ASSUMPTIONS["target_wait_minutes"], step=5, key="target",
                       help="Lines longer than this carry a steep extra cost in the plan.")
    st.form_submit_button("Run scenario", type="primary", icon=":material/play_arrow:", width="stretch")

with results:
    if hours[1] - hours[0] < ASSUMPTIONS["part_shift_hours"]:
        st.warning(f"Keep the park open at least {ASSUMPTIONS['part_shift_hours']} hours: the shortest shift is that long.",
                   icon=":material/warning:")
        st.stop()
    assumptions = {"wage_per_hour": float(wage), "wait_cost_per_guest_hour": float(wait_cost),
                   "target_wait_minutes": int(target)}
    published = data["shifts"][(data["shifts"]["park_id"] == park_id) & (data["shifts"]["date"] == pd.Timestamp(day))]
    with st.spinner("Solving staffing plans..."):
        plans = scenarios.run(park_locations, published[["location", "start", "end", "workers"]], forecast_crowd,
                              planned_hours, int(crowd), tuple(hours), assumptions, int(cap) if cap else None)

    replan, keep = plans["Re-planned"], plans["Published plan, scenario crowd"]
    if replan.status == "infeasible":
        st.warning("No plan fits a worker cap that low: every location needs its minimum crew all day. "
                   "Raise the cap or clear it.", icon=":material/warning:")
        st.stop()

    st.markdown(f"**Re-planned for a crowd of {crowd}**, compared with keeping the published crew")
    new, old = transforms.headline(replan), transforms.headline(keep)
    vs = "vs keeping the plan"
    with st.container(horizontal=True):
        st.metric("Labor cost", f"${new['labor_cost']:,.0f}",
                  transforms.signed(new["labor_cost"] - old["labor_cost"], prefix="$"),
                  delta_color="inverse", delta_description=vs, border=True)
        st.metric("Guest-hours in line", f"{new['wait_guest_hours']:,.0f}",
                  transforms.signed(new["wait_guest_hours"] - old["wait_guest_hours"]),
                  delta_color="inverse", delta_description=vs, border=True)
        st.metric("Longest wait", f"{new['worst_wait_minutes']} min",
                  transforms.signed(new["worst_wait_minutes"] - old["worst_wait_minutes"], suffix=" min"),
                  delta_color="inverse", delta_description=vs, border=True)
        st.metric("Guests who skipped a line", f"{new['skipped_guests']:,}",
                  transforms.signed(new["skipped_guests"] - old["skipped_guests"]),
                  delta_color="inverse", delta_description=vs, border=True)
    maxed = transforms.at_max_crew(replan.hourly, park_locations)
    if maxed:
        st.caption(f"{len(maxed)} of {len(park_locations)} locations reach their maximum crew in the re-plan: "
                   f"{', '.join(maxed)}. More staff can't shorten those lines, so the re-plan can only do so much.")

    comparison =pd.DataFrame({label: transforms.headline(p) for label, p in plans.items()})
    comparison.index = ["Workers", "Labor cost, $", "Guest-hours in line", "Longest wait, minutes",
                        "Guests who skipped a line"]
    st.table(comparison.map(lambda v: f"{v:,.0f}"), alt="The three plans side by side")

    wait_by_hour = pd.concat([transforms.by_hour(p.hourly).assign(series=label) for label, p in plans.items()])
    with st.container(border=True):
        st.subheader("Average wait by hour")
        st.altair_chart(charts.lines(wait_by_hour.rename(columns={"avg_wait": "value"}), "hour:Q", "Minutes",
                                     scenarios.SCENARIOS, hourly=True),
                        alt="Average wait across locations by hour for the three plans")
    crew_by_hour = pd.concat([transforms.by_hour(plans[label].hourly).assign(series=label)
                              for label in ("Published plan", "Re-planned")])
    with st.container(border=True):
        st.subheader("Crew on duty by hour")
        st.altair_chart(charts.lines(crew_by_hour.rename(columns={"crew": "value"}), "hour:Q", "Workers",
                                     ["Published plan", "Re-planned"], hourly=True),
                        alt="Workers on duty across the park by hour, published plan and re-plan")
