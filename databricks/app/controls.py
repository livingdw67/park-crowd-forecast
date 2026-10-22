"""Pickers shared by the staffing and what-if pages, bound to the URL so a park-day carries across pages."""

import pandas as pd
import streamlit as st

import park_data
import transforms


def park_and_day(summary, locations):
    """Park (those with staffed locations) and plan day; returns (name, park_id, day, the day's summary row or None)."""
    staffed = [name for name, park_id in park_data.PARK_IDS.items() if park_id in set(locations["park_id"])]
    dates = sorted(summary["date"].dt.strftime("%Y-%m-%d").unique())
    upcoming = [d for d in dates if d >= str(park_data.today())] or dates
    with st.container(horizontal=True, vertical_alignment="bottom"):
        name = st.segmented_control("Park", staffed, default=staffed[0], required=True, key="park",
                                    bind="query-params")
        day = st.selectbox("Day", upcoming, format_func=transforms.day_label, key="day", bind="query-params",
                           width=220)
    park_id = park_data.PARK_IDS[name]
    rows = summary[(summary["park_id"] == park_id) & (summary["date"] == pd.Timestamp(day))]
    return name, park_id, day, rows.iloc[0] if len(rows) else None
