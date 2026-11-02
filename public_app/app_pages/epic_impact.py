import pandas as pd
import streamlit as st

import cached
import lab
import viz

panel = cached.monthly()
PRE_STARTS = pd.date_range("2021-07-01", "2023-06-01", freq="MS")
METHODS = {"Synthetic control": "synthetic_control", "Difference-in-differences": "diff_in_diff"}

st.caption("Did Epic Universe (opened May 22, 2025) pull crowds from Universal's older parks? Each park is compared "
           "with parks Epic couldn't have affected, outside Florida, to estimate its crowds without Epic. Monthly "
           "average crowd scores, each park's usual seasonal pattern removed. Change the setup and the study reruns "
           "here, the same code the weekly Databricks job runs.")

with st.container(horizontal=True):
    name = st.segmented_control("Park", list(lab.IMPACT_UNITS), default="Universal Studios Florida", required=True,
                                key="epic_park")
    method_label = st.segmented_control(
        "Method", list(METHODS), default="Synthetic control", required=True, key="epic_method",
        help="Synthetic control: the weighted mix of comparison parks that best tracked this park before the "
             "opening. Difference-in-differences: the comparison parks' simple average.")
    pre_start = st.select_slider("Before period starts", list(PRE_STARTS), value=pd.Timestamp("2022-01-01"),
                                 format_func=lambda m: f"{m:%b %Y}", key="epic_pre",
                                 help="The months used to learn each park's seasonal pattern and, for synthetic "
                                      "control, the weights. Comparison history starts July 2021.")
donors = st.pills("Comparison parks", list(lab.DONORS), default=lab.DEFAULT_DONORS, selection_mode="multi",
                  format_func=lab.DONORS.get, key="epic_donors",
                  help="Disneyland's 70th anniversary began the same month Epic opened. Busch Gardens Tampa is near "
                       "enough to Orlando that Epic could have affected it, so it is out by default.")
fake = st.toggle("Fake opening in May 2024 (a year before Epic existed)", key="epic_fake",
                 help="A placebo in time: run the same study as if Epic had opened a year early. A sound setup should "
                      "find nothing.")

if len(donors) < 3:
    st.warning("Pick at least three comparison parks: the placebo test needs parks to compare with.",
               icon=":material/warning:")
    st.stop()

unit = lab.IMPACT_UNITS[name]
r = cached.study(unit, tuple(sorted(donors)), pre_start.strftime("%Y-%m-%d"), METHODS[method_label], fake)
k = round(r["p_effect"] * (r["n_placebos"] + 1)) - 1

with st.container(horizontal=True):
    st.metric("Change after the opening", f"{r['effect']:+.1f} points", border=True,
              help="Average gap between actual and estimated crowds after the opening, minus the same gap before.")
    st.metric("Relative to before", f"{r['effect_pct']:+.0f}%", border=True,
              help=f"Against the park's average monthly score before the opening ({r['pre_mean']:.0f}).")
    st.metric("Placebo parks that moved as much", f"{k} of {r['n_placebos']}", border=True,
              help=f"p = {r['p_effect']:.2f}, with {r['n_placebos'] + 1} parks the smallest possible p is "
                   f"{1 / (r['n_placebos'] + 1):.2f}.")
    st.metric("Fit before the opening", f"{r['pre_rmspe']:.1f} points", border=True,
              help="Root mean squared gap before the opening. Small means the estimate tracked the park well, so "
                   "the gap after means more.")

pre_months, post_months = r["months"]
st.caption(f"{pre_months} months before, {post_months} after, through {r['through']:%B %Y}.")
st.altair_chart(viz.impact(r["series"], r["opening"]), alt=f"Actual monthly crowd score and the no-Epic estimate for {name}")

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Against the placebo parks", anchor=False)
    st.caption("The same study run with each comparison park pretending to be the one Epic affected. If this park's "
               "change is no bigger than theirs, it could be ordinary noise.")
    st.altair_chart(viz.placebo_bars(r["placebos"]), alt="Estimated change for this park and each placebo park")
with right:
    if METHODS[method_label] == "synthetic_control":
        st.subheader("What the estimate is made of", anchor=False)
        st.caption("The mix of comparison parks that tracked this park best before the opening.")
        w = pd.DataFrame(sorted(r["weights"].items(), key=lambda kv: -kv[1]), columns=["park", "weight"])
        st.altair_chart(viz.weights(w), alt="Synthetic control weights per comparison park")
    else:
        st.subheader("What the estimate is made of", anchor=False)
        st.caption("Difference-in-differences weighs every comparison park equally.")

with st.expander("About the placebo test", icon=":material/science:"):
    st.markdown(f"""
**Why it's needed.** There is one Epic Universe and one opening, so there is no sample of openings to
compute a standard error from. Instead the test asks: how unusual is this park's change compared with parks
that had no opening at all?

**What it does.** It reruns the study {r['n_placebos']} times, each time pretending one comparison park was
the one affected, and counts how many of those fake effects are at least as large as the real one. The
p-value is (1 + that count) / ({r['n_placebos']} + 1); with this few parks it can't go below
{1 / (r['n_placebos'] + 1):.2f}, so "not significant" here partly reflects how few parks there are.

**In plain terms.** {k} of {r['n_placebos']} placebo parks moved at least as much as {name}. The fewer, the
less likely the change is ordinary park-to-park noise.

**Assumptions.** Without Epic, the park would have kept tracking its comparison parks the way it did before
(check the fit before the opening and the chart's left side); the comparison parks weren't affected by Epic
themselves (why Florida parks are left out); and nothing else changed for this park at the same time. Two checks
are on this page: the fake 2024 opening, which should find nothing, and dropping the Disneyland parks, whose
70th anniversary began the same month.
""")
