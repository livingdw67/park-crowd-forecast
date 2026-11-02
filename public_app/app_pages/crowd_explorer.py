import pandas as pd
import streamlit as st

import cached
import lab
import viz
from transforms import trend_frame

daily = cached.daily()
days = daily.assign(park_id=daily["park_id"].astype(int))

st.caption("Every measured day since 2018, one park and year at a time. The crowd score is queue-times' daily 0-100 "
           "score, relative to each park's own busy days, so compare days within a park rather than across parks. "
           "Click a day to see why the model expects it to be busy or quiet.")

with st.container(horizontal=True):
    park = st.segmented_control("Park", list(lab.PARKS.values()), default=lab.PARKS[65], required=True,
                                key="explore_park")
    park_id = {v: k for k, v in lab.PARKS.items()}[park]
    park_days = days[days["park_id"] == park_id]
    years = sorted(park_days["date"].dt.year.unique().tolist())
    year = st.segmented_control("Year", years, default=years[-1], required=True,
                                key=f"explore_year_{park_id}")

year_days = park_days[park_days["date"].dt.year == year].assign(
    day_key=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
busiest = year_days.loc[year_days["crowd_percent"].idxmax()]
quietest = year_days.loc[year_days["crowd_percent"].idxmin()]

with st.container(horizontal=True):
    st.metric(f"Average, {year}", f"{year_days['crowd_percent'].mean():.0f}", border=True,
              help=f"{len(year_days)} measured days")
    st.metric(f"Busiest: {lab.day_label(busiest['date'])}", f"{busiest['crowd_percent']:.0f}", border=True)
    st.metric(f"Quietest: {lab.day_label(quietest['date'])}", f"{quietest['crowd_percent']:.0f}", border=True)

key = f"explore_cal_{park_id}_{year}"
picked = (st.session_state.get(key) or {}).get("selection", {}).get("day") or []
selected = picked[0]["day_key"] if picked else busiest["day_key"]
st.altair_chart(viz.year_calendar(year_days, selected), on_select="rerun", selection_mode="day", key=key,
                alt=f"Daily crowd score calendar for {park}, {year}; click a day to explain it")

# ------------------------------------------------------------------------------------- why this day
row = year_days[year_days["day_key"] == selected]
day = row.iloc[0]
model, features = cached.champion_model()
contrib = lab.contributions(model, daily.loc[row.index], features).iloc[0]
baseline = float(contrib.pop("baseline"))
estimate = min(max(baseline + float(contrib.sum()), 0), 100)
groups = (contrib.rename("points").rename_axis("group").reset_index()
          .assign(size=lambda f: f["points"].abs()).sort_values("size", ascending=False)
          .query("size >= 0.05").drop(columns="size"))

holiday = f" ({day['holiday_name']})" if isinstance(day["holiday_name"], str) and day["holiday_name"] else ""
st.subheader(f"Why {lab.day_label(day['date'], '%A,', '%B', year=True)}{holiday}", anchor=False)
left, right = st.columns([1, 2], gap="large")
with left:
    st.metric("Measured crowd score", f"{day['crowd_percent']:.0f}", border=True)
    st.metric("Model's estimate", f"{estimate:.0f}", border=True,
              delta=f"{estimate - day['crowd_percent']:+.0f} vs. measured", delta_color="off",
              help="The champion model trained on every measured day, so this day was in its training data. "
                   "On days it hasn't seen, its typical miss is about 15 points (see the model lab).")
    st.metric("Model's average day", f"{baseline:.0f}", border=True,
              help="Where the model starts before looking at anything about the day.")
with right:
    st.caption("Each bar is how far one group of features moved the estimate from the model's average day "
               "(SHAP values, which add up exactly to the estimate).")
    st.altair_chart(viz.contributions(groups), alt="Contribution of each feature group to the day's estimate")

# ------------------------------------------------------------------------------------- next 90 days
st.subheader("The next 90 days", anchor=False)
forecast = cached.forecast()
snapshot = pd.Timestamp(cached.meta()["exported_at"][:10])
measured = (days[days["date"] >= snapshot - pd.Timedelta(days=90)]
            .rename(columns={"crowd_percent": "measured_crowd"})[["park_id", "date", "measured_crowd"]])
frame = trend_frame(forecast[forecast["date"] >= snapshot], measured, park_id)
if frame.empty:
    st.info("No forecast for this park in the snapshot.", icon=":material/event_busy:")
else:
    st.caption(f"The production forecast as of the snapshot ({lab.day_label(snapshot, '', '%B', year=True).strip()}), next to queue-times' own "
               "forecast and the last 90 measured days.")
    st.altair_chart(viz.trend(frame, snapshot), alt=f"Measured crowd, forecast and queue-times forecast for {park}")
