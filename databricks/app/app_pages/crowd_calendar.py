import pandas as pd
import streamlit as st

import charts
import park_data
import transforms

data = park_data.load()
today = park_data.today()
forecast = data["forecast"]

name = st.segmented_control("Park", list(park_data.PARK_IDS), default="Universal Studios Florida", required=True,
                            key="park", bind="query-params")
park_id = park_data.PARK_IDS[name]

grid = transforms.calendar_grid(forecast, park_id, today)
if grid.empty:
    st.warning("No forecast from today on yet. The daily-features job writes one at 6:30am Eastern.",
               icon=":material/schedule:")
    st.stop()

st.caption("The crowd score is queue-times' daily 0-100 score. It is relative to each park's own busy days, so "
           "compare days within a park rather than parks with each other. The forecast is a LightGBM model on "
           "calendar features (holidays, school breaks, events); a single day's forecast is typically off by about "
           f"15 points. Scored {transforms.stamp(forecast['scored_at'].max())}.")

month = grid[grid["date"] < pd.Timestamp(today) + pd.Timedelta(days=30)]
busiest = month.loc[month["predicted_crowd"].idxmax()]
quietest = month.loc[month["predicted_crowd"].idxmin()]
both = grid.dropna(subset=["queue_times_forecast"])
with st.container(horizontal=True):
    st.metric("Next 7 days, average", f"{grid.head(7)['predicted_crowd'].mean():.0f}", border=True)
    st.metric(f"Busiest in the next 30 days: {transforms.day_label(busiest['date'])}",
              f"{busiest['predicted_crowd']:.0f}", border=True)
    st.metric(f"Quietest in the next 30 days: {transforms.day_label(quietest['date'])}",
              f"{quietest['predicted_crowd']:.0f}", border=True)
    st.metric("Gap to queue-times' forecast",
              f"{(both['predicted_crowd'] - both['queue_times_forecast']).abs().mean():.0f} points" if len(both)
              else "Not published", border=True,
              help="Average absolute difference on the days queue-times publishes its own forecast.")

with st.container(border=True):
    st.subheader("Next 90 days")
    st.altair_chart(charts.calendar(grid, today), alt=f"Forecast crowd score by day for {name}, next 90 days")

with st.container(border=True):
    st.subheader("Measured and forecast")
    trend = transforms.trend_frame(forecast, data["measured"], park_id)
    st.altair_chart(charts.trend(trend, today),
                    alt=f"Measured crowd score for the last 90 days and the forecasts ahead for {name}")

with st.expander("Forecast table", icon=":material/table:"):
    st.dataframe(
        grid[["date", "weekday", "predicted_crowd", "queue_times_forecast", "holiday_name"]],
        hide_index=True, alt=f"Daily forecast for {name}",
        column_config={
            "date": st.column_config.DateColumn("Date", format="ddd MMM D, YYYY"),
            "weekday": None,
            "predicted_crowd": st.column_config.NumberColumn("Forecast", format="%.0f"),
            "queue_times_forecast": st.column_config.NumberColumn("Queue-times forecast", format="%.0f"),
            "holiday_name": st.column_config.TextColumn("Holiday"),
        })
