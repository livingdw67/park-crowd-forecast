import pandas as pd
import streamlit as st

import charts
import park_data
import transforms

impact = park_data.load_impact()
summary, gaps = impact["summary"], impact["gaps"]
if summary.empty:
    st.warning("No study results yet. The comparison-calendar-ingest job runs the study on Mondays at 5am Eastern.",
               icon=":material/schedule:")
    st.stop()

st.caption("Did Epic Universe (opened May 22, 2025) pull crowds from Universal's older parks? Each park is compared "
           "with parks Epic couldn't have affected, outside Florida, to estimate what its crowds would have been "
           "without Epic. Monthly average crowd scores, each park's usual seasonal pattern taken into account. "
           f"Results through {gaps['month'].max():%B %Y}; rerun weekly.")

with st.container(horizontal=True):
    name = st.segmented_control("Park", list(transforms.IMPACT_UNITS), default="Universal Studios Florida",
                                required=True, key="impact_park", bind="query-params")
    method_label = st.segmented_control(
        "Method", list(transforms.IMPACT_METHODS), default="Synthetic control", required=True, key="impact_method",
        help="Synthetic control: the mix of comparison parks that best tracked this park before the opening. "
             "Difference-in-differences: the comparison parks' simple average.")
    group_label = st.selectbox("Comparison parks", list(transforms.IMPACT_GROUPS), key="impact_group",
                               help="Disneyland's 70th anniversary began the same month as Epic opened. Busch "
                                    "Gardens Tampa is close enough that Orlando visitors day-trip there.")
park_id = transforms.IMPACT_UNITS[name]
method = transforms.IMPACT_METHODS[method_label]
variant = transforms.IMPACT_GROUPS[group_label]

row = transforms.impact_row(summary, park_id, method, variant)
fake = transforms.impact_row(summary, park_id, method, "placebo_opening_2024_05")
if row is None:
    st.warning("The study has no result for this combination yet.", icon=":material/info:")
    st.stop()

as_large, placebos = transforms.as_large_as(row)
with st.container(horizontal=True):
    st.metric("Change since Epic opened", f"{row['effect_points']:+.0f} points", border=True,
              help="Average gap between the park and its no-Epic estimate after the opening, minus the gap before it.")
    st.metric("Relative to its pre-Epic level", f"{row['effect_pct']:+.0f}%", border=True,
              help="The crowd score is built from wait times, which shrink faster than crowds; the drop in visitors "
                   "is likely smaller than this.")
    st.metric("Placebo parks that moved as much", f"{as_large} of {placebos}", border=True,
              help="Each comparison park, rerun as if it had its own Epic. Fewer is stronger evidence; with "
                   f"{placebos} parks, 0 of {placebos} is the strongest result the test can give.")
    if fake is not None:
        st.metric("Same method, fake opening in May 2024", f"{fake['effect_points']:+.0f} points", border=True,
                  help="A check: nothing happened in May 2024, so this should be near zero.")

with st.container(border=True):
    st.subheader("Actual crowds vs. without Epic")
    frame = transforms.impact_frame(gaps, park_id, method, variant)
    st.altair_chart(charts.impact(frame, transforms.EPIC_OPENING),
                    alt=f"Monthly crowd score for {name} and its estimated level without Epic Universe, 2022 on")
    st.caption("Before the opening the two lines should track each other; the space between them afterwards is the "
               "estimated effect.")

with st.container(border=True):
    st.subheader("How unusual is this?")
    st.altair_chart(charts.placebo_bars(transforms.placebo_frame(row)),
                    alt=f"Estimated change for {name} beside each comparison park rerun as a placebo")
    st.caption("Grey bars: each comparison park rerun as if Epic had opened next to it. Real effects should stand "
               "out from them; parks also move for reasons of their own.")

with st.expander("What this can and can't say", icon=":material/help:"):
    st.markdown(
        "- **Suggestive, not proven.** With so few comparison parks, no result can be statistically conclusive, "
        "and Disneyland moved almost as much as Universal Studios on its own.\n"
        "- **Crowd score isn't headcount.** It comes from wait times, and lines shrink faster than crowds when rides "
        "run near capacity, so the drop in visitors is likely smaller than the drop in the score. Comcast said Epic "
        "took fewer visitors from the older parks than it expected.\n"
        "- **Islands of Adventure was already slipping** in 2024, before Epic; that is why its fake-2024 check "
        "isn't zero.\n"
        "- **Synthetic control leans on Legoland California.** Difference-in-differences weighs every comparison "
        "park equally and gives a similar answer for Universal Studios.")

with st.expander("Other Orlando parks (inconclusive)", icon=":material/table:"):
    others = summary[summary["role"].isin(["orlando", "nearby"]) & (summary["variant"] == "main")]
    table = others.pivot_table(index="park", columns="method", values=["effect_points", "pre_rmspe"]).round(1)
    table.columns = [f"{m}_{v}" for v, m in table.columns]
    st.dataframe(
        table.reset_index(), hide_index=True, alt="Estimated change at other Orlando-area parks",
        column_config={
            "park": st.column_config.TextColumn("Park"),
            "synthetic_control_effect_points": st.column_config.NumberColumn("Synthetic control, points", format="%+.1f"),
            "diff_in_diff_effect_points": st.column_config.NumberColumn("Diff-in-diff, points", format="%+.1f"),
            "synthetic_control_pre_rmspe": st.column_config.NumberColumn(
                "Pre-opening miss, points", format="%.1f",
                help="How far the synthetic control missed each month before the opening. Universal Studios: about 3."),
            "diff_in_diff_pre_rmspe": None,
        })
    st.caption("The comparison parks can't reproduce Disney's history well (large pre-opening misses), and the two "
               "methods disagree, so these don't support a conclusion either way.")

with st.expander("Monthly numbers", icon=":material/table:"):
    wide = frame.pivot_table(index="month", columns="series", values="value").reset_index()
    st.dataframe(wide, hide_index=True, alt=f"Monthly actual and no-Epic estimate for {name}",
                 column_config={"month": st.column_config.DateColumn("Month", format="MMM YYYY"),
                                **{s: st.column_config.NumberColumn(s, format="%.0f") for s in transforms.IMPACT_SERIES}})
