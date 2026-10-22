import pandas as pd
import streamlit as st

import charts
import controls
import park_data
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

st.caption(f"Planned for a forecast crowd of {plan.predicted_crowd:.0f}, open "
           f"{transforms.hour_label(plan.open_hour)} to {transforms.hour_label(plan.close_hour)}. The arrows show what "
           "the same crew faces on a day 15 points busier, about one typical forecast error. "
           f"Planned {transforms.stamp(plan.planned_at)}.")

busier = "if 15 points busier"
cols = st.columns(5)
cols[0].metric("Workers", f"{plan.workers:,}", border=True, height="stretch")
cols[1].metric("Labor cost", f"${plan.labor_cost:,.0f}", border=True, height="stretch")
cols[2].metric("Guest-hours in line", f"{plan.wait_guest_hours:,.0f}",
               transforms.signed(plan.wait_guest_hours_if_15_busier - plan.wait_guest_hours),
               delta_color="inverse", delta_description=busier, border=True, height="stretch")
cols[3].metric("Longest wait", f"{plan.worst_wait_minutes} min",
               transforms.signed(plan.worst_wait_minutes_if_15_busier - plan.worst_wait_minutes, suffix=" min"),
               delta_color="inverse", delta_description=busier, border=True, height="stretch")
cols[4].metric("Guests who skipped a line", f"{plan.skipped_guests:,}",
               transforms.signed(plan.skipped_guests_if_15_busier - plan.skipped_guests),
               delta_color="inverse", delta_description=busier, border=True, height="stretch")

def this_day(frame):
    return frame[(frame["park_id"] == park_id) & (frame["date"] == pd.Timestamp(day))]


hourly = this_day(data["hourly"])
order = transforms.location_order(locations[locations["park_id"] == park_id])

with st.container(border=True):
    st.subheader("Crew on duty")
    st.altair_chart(charts.location_heatmap(hourly, "crew", "Workers", order),
                    alt=f"Crew on duty by location and hour at {name}")

with st.container(border=True):
    st.subheader("Planned wait")
    st.altair_chart(charts.location_heatmap(hourly, "wait_minutes", "Minutes", order),
                    alt=f"Planned wait by location and hour at {name}")

with st.container(border=True):
    st.subheader("Shifts")
    shifts = this_day(data["shifts"]).assign(
        location=lambda d: pd.Categorical(d["location"], order, ordered=True),
        shift=lambda d: d["start"].map(transforms.hour_label) + " to " + d["end"].map(transforms.hour_label),
        hours=lambda d: d["end"] - d["start"],
    ).sort_values(["location", "start"])
    st.dataframe(shifts[["location", "shift", "hours", "workers"]], hide_index=True, alt=f"Shifts at {name}",
                 column_config={"location": "Location", "shift": "Shift",
                                "hours": st.column_config.NumberColumn("Hours"),
                                "workers": st.column_config.NumberColumn("Workers")})

with st.expander("Hourly detail", icon=":material/table:"):
    st.dataframe(hourly.assign(hour=hourly["hour"].map(transforms.hour_label))
                 [["location", "hour", "crew", "arrivals", "skipped", "served", "line", "wait_minutes"]],
                 hide_index=True, alt=f"Hourly plan detail at {name}",
                 column_config={"location": "Location", "hour": "Hour", "crew": "Crew",
                                "arrivals": "Guests arriving", "skipped": "Skipped the line", "served": "Served",
                                "line": "In line", "wait_minutes": "Wait, minutes"})
