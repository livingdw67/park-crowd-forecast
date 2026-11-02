import pandas as pd
import streamlit as st

import cached
import lab
import viz

LEARNING_RATES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
DEFAULTS = {"n_estimators": lab.CHAMPION_PARAMS["n_estimators"], "learning_rate": lab.CHAMPION_PARAMS["learning_rate"],
            "num_leaves": lab.CHAMPION_PARAMS["num_leaves"], "min_child_samples": lab.CHAMPION_PARAMS["min_child_samples"],
            "subsample": lab.CHAMPION_PARAMS["subsample"], "colsample_bytree": lab.CHAMPION_PARAMS["colsample_bytree"],
            "objective": "Absolute error (L1)", "groups": lab.CHAMPION_GROUPS}


def reset():
    for key, value in DEFAULTS.items():
        st.session_state[f"lab_{key}"] = value


if "lab_n_estimators" not in st.session_state:
    reset()

st.caption("Retrain the crowd forecast with your own settings and see how it does on years it never saw. Each test "
           "year (2023 to 2026) is predicted by a model trained only on the days before it, exactly like the weekly "
           "Databricks job. The champion is the production model: LightGBM on calendar features.")

with st.form("lab"):
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Settings", anchor=False)
        a, b = st.columns(2)
        a.slider("Trees", 50, 1500, step=50, key="lab_n_estimators",
                 help="How many small trees the model adds up. More trees fit more detail, and take longer.")
        b.select_slider("Learning rate", LEARNING_RATES, key="lab_learning_rate",
                        help="How much each tree corrects the ones before it. Lower needs more trees.")
        a.slider("Leaves per tree", 4, 128, key="lab_num_leaves",
                 help="How many groups each tree can split days into. More leaves = more complex trees.")
        b.slider("Minimum days per leaf", 5, 200, step=5, key="lab_min_child_samples",
                 help="A leaf needs at least this many training days. Higher = smoother, less overfitting.")
        a.slider("Days sampled per tree", 0.5, 1.0, step=0.05, key="lab_subsample",
                 help="Share of training days each tree sees (bagging).")
        b.slider("Features sampled per tree", 0.3, 1.0, step=0.05, key="lab_colsample_bytree",
                 help="Share of features each tree may use.")
    with right:
        st.subheader("Features", anchor=False)
        st.pills("Feature groups", list(lab.FEATURE_GROUPS), selection_mode="multi", key="lab_groups",
                 help="Park, season and weekday is always on. Park hours and weather are measured, not known "
                      "ahead: hours are published about two months out and weather about two weeks out.")
        st.segmented_control("Loss", list(lab.OBJECTIVES), key="lab_objective", required=True,
                             help="What training minimizes. The champion uses absolute error, the same measure the "
                                  "backtest reports.")
    st.form_submit_button("Retrain and backtest", type="primary", icon=":material/model_training:")
st.button("Reset to the champion", icon=":material/restart_alt:", on_click=reset)

groups = tuple(g for g in lab.FEATURE_GROUPS
               if g == "Park, season and weekday" or g in (st.session_state["lab_groups"] or []))
params = {k: st.session_state[f"lab_{k}"] for k in lab.CHAMPION_PARAMS}
params["objective"] = lab.OBJECTIVES[st.session_state["lab_objective"]]

with st.spinner("Training four models, one per test year..."):
    champ_preds, _, _ = cached.backtest(tuple(lab.CHAMPION_GROUPS), tuple(sorted(
        {**lab.CHAMPION_PARAMS, "objective": "regression_l1"}.items())))
    preds, importance, seconds = cached.backtest(groups, tuple(sorted(params.items())))
base = cached.baselines()

yours_by_year = lab.mae_by_year(preds).assign(method=viz.YOURS)
champ_by_year = lab.mae_by_year(champ_preds).assign(method=viz.CHAMPION)
yours_mae = float((preds["actual"] - preds["pred"]).abs().mean())
champ_mae = float((champ_preds["actual"] - champ_preds["pred"]).abs().mean())
base_mae = base.groupby("method")["mae"].mean()
boot = lab.block_bootstrap(preds, champ_preds)
outcome, sentence = lab.verdict(boot)

st.subheader("How your model did", anchor=False)
with st.container(horizontal=True):
    st.metric("Your model, average error", f"{yours_mae:.2f} points", border=True,
              delta=f"{yours_mae - champ_mae:+.2f} vs. champion", delta_color="inverse",
              help="Mean absolute error over every park-day in 2023-2026: how far a day's prediction typically lands "
                   "from the measured crowd score (0-100).")
    st.metric("Champion", f"{champ_mae:.2f} points", border=True)
    st.metric(f"Best baseline: {base_mae.idxmin().lower()}", f"{base_mae.min():.2f} points", border=True,
              help="Simple rules any forecast should beat.")
    st.metric("Training time", f"{seconds:.1f} s", border=True, help="Four models, one per test year, in this app.")

alert = {"better": st.success, "worse": st.warning, "tie": st.info}[outcome]
alert(f"{sentence} 95% interval for the difference: {boot['lo']:+.2f} to {boot['hi']:+.2f} points a day.",
      title={"better": "Better than the champion", "worse": "Worse than the champion",
             "tie": "No real difference"}[outcome],
      icon={"better": ":material/trending_down:", "worse": ":material/trending_up:", "tie": ":material/balance:"}[outcome])

with st.expander("About this test: a paired block bootstrap", icon=":material/science:"):
    st.markdown(f"""
**Why it's needed.** Two models' average errors almost never match exactly, so "yours is 0.3 points lower"
on its own says nothing: a different set of test days could flip it. The question is whether the
difference is bigger than the luck of the draw.

**What it does.** Both models predicted the same {boot['days']:,} park-days. For each day it takes your
model's error minus the champion's, then reshuffles whole weeks of those differences
({boot['blocks']} weeks, drawn with replacement, 4,000 times) to see how much the average difference moves
from one plausible set of test weeks to another. The middle 95% of those averages is the interval above.

**In plain terms.** If the interval sits entirely below zero, your model's errors are reliably smaller; entirely
above zero, reliably larger; across zero, the two can't be told apart on this data.

**Assumptions, and a check.** Days aren't independent: a holiday week or a hurricane moves every day near it,
so days are resampled in week-long blocks that keep those runs together, and the test assumes weeks are roughly
independent of one another. That is the shakiest part, so here is the same interval with longer blocks. If the
verdict holds as the blocks grow, it doesn't hinge on that assumption.
""")
    rows = []
    for days in (7, 14, 28):
        b = lab.block_bootstrap(preds, champ_preds, block_days=days)
        rows.append({"Block length": f"{days} days", "Blocks": b["blocks"], "Difference": b["diff"],
                     "95% interval, low": b["lo"], "95% interval, high": b["hi"], "Verdict": lab.verdict(b)[0]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Bootstrap interval for three block lengths",
                 column_config={c: st.column_config.NumberColumn(format="%+.2f")
                                for c in ("Difference", "95% interval, low", "95% interval, high")})

st.subheader("Error by test year", anchor=False)
st.caption("Lower is better. The two grey lines are the baselines. Every year is a model trained only on earlier days; "
           "2026 runs through the snapshot date.")
frame = pd.concat([yours_by_year, champ_by_year, base], ignore_index=True)
st.altair_chart(viz.mae_by_year(frame), alt="Backtest error by test year for your model, the champion and two baselines")

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Predictions against measured crowds", anchor=False)
    with st.container(horizontal=True):
        park = st.segmented_control("Park", [lab.PARKS[p] for p in (65, 64, 334)], default=lab.PARKS[65],
                                    required=True, key="lab_park")
        year = st.segmented_control("Year", lab.TEST_YEARS, default=2025, required=True, key="lab_year")
    park_id = {v: k for k, v in lab.PARKS.items()}[park]

    def one(p):
        return p[(p["park_id"] == park_id) & (p["test_year"] == year)]

    mine, champ = one(preds), one(champ_preds)
    if mine.empty:
        st.info("No measured days for this park in that year (Epic Universe opened in May 2025).",
                icon=":material/event_busy:")
    else:
        lines = pd.concat([
            mine.assign(series=viz.MEASURED, value=mine["actual"]),
            champ.assign(series=viz.CHAMPION, value=champ["pred"]),
            mine.assign(series=viz.YOURS, value=mine["pred"]),
        ])[["date", "series", "value"]]
        st.altair_chart(viz.prediction_lines(lines), alt=f"Daily measured crowd score and predictions, {park}, {year}")
with right:
    st.subheader("What your model leans on", anchor=False)
    st.caption("Share of the model's total gain (how much each feature's splits cut the error), averaged over the "
               "four test-year models.")
    st.altair_chart(viz.importance(importance), alt="Feature importance for your model")
