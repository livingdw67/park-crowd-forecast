import pandas as pd
import streamlit as st

import cached
import lab
import viz

SCORE_ON = {"All test years, 2023-2026": lab.TEST_YEARS, "2025-2026 only": [2025, 2026]}
# Sliders hold the exact preset values (they don't snap to their step); depth 0 on the slider means no limit
SLIDERS = ["n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree",
           "max_depth", "reg_alpha", "reg_lambda"]


def load(values, groups, score_on):
    for key in SLIDERS:
        value = values[key]
        st.session_state[f"lab_{key}"] = 0 if key == "max_depth" and value == -1 else value
    st.session_state["lab_objective"] = "Absolute error (L1)"
    st.session_state["lab_groups"] = list(groups)
    st.session_state["lab_score_on"] = score_on


def reset():
    load(lab.CHAMPION_PARAMS, lab.CHAMPION_GROUPS, "All test years, 2023-2026")


def load_optuna():
    # Optuna chose these on 2023-2024, so only 2025-2026 is a fair test of them
    load(lab.OPTUNA_PARAMS, lab.CHAMPION_GROUPS, "2025-2026 only")


if "lab_n_estimators" not in st.session_state:
    reset()

st.caption("Retrain the crowd forecast with your own settings and see how it does on years it never saw. Each test "
           "year (2023 to 2026) is predicted by a model trained only on the days before it, exactly like the weekly "
           "Databricks job. The champion is the production model: LightGBM on calendar features.")

with st.form("lab"):
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Settings", anchor=False)
        a, b, c = st.columns(3)
        a.slider("Trees", 50, 1500, step=10, key="lab_n_estimators",
                 help="How many small trees the model adds up. More trees fit more detail, and take longer.")
        b.slider("Learning rate", 0.005, 0.2, step=0.005, format="%.4f", key="lab_learning_rate",
                 help="How much each tree corrects the ones before it. Lower needs more trees.")
        c.slider("Leaves per tree", 4, 128, key="lab_num_leaves",
                 help="How many groups each tree can split days into. More leaves = more complex trees.")
        a.slider("Maximum depth", 0, 12, key="lab_max_depth",
                 help="How many splits deep a tree may go; 0 means no limit (the champion's setting).")
        b.slider("Minimum days per leaf", 5, 200, key="lab_min_child_samples",
                 help="A leaf needs at least this many training days. Higher = smoother, less overfitting.")
        c.slider("Days sampled per tree", 0.5, 1.0, step=0.01, key="lab_subsample",
                 help="Share of training days each tree sees (bagging).")
        a.slider("Features sampled per tree", 0.3, 1.0, step=0.01, key="lab_colsample_bytree",
                 help="Share of features each tree may use.")
        b.slider("L1 penalty", 0.0, 5.0, step=0.1, format="%.2f", key="lab_reg_alpha",
                 help="Pushes small leaf values to zero. 0 for the champion.")
        c.slider("L2 penalty", 0.0, 5.0, step=0.1, format="%.2f", key="lab_reg_lambda",
                 help="Shrinks every leaf value. 0 for the champion.")
    with right:
        st.subheader("Features", anchor=False)
        st.pills("Feature groups", list(lab.FEATURE_GROUPS), selection_mode="multi", key="lab_groups",
                 help="Park, season and weekday is always on. Park hours and weather are measured, not known "
                      "ahead: hours are published about two months out and weather about two weeks out.")
        st.segmented_control("Loss", list(lab.OBJECTIVES), key="lab_objective", required=True,
                             help="What training minimizes. The champion uses absolute error, the same measure the "
                                  "backtest reports.")
    st.form_submit_button("Retrain and backtest", type="primary", icon=":material/model_training:")
with st.container(horizontal=True):
    st.button("Reset to the champion", icon=":material/restart_alt:", on_click=reset)
    st.button("Load Optuna's best", icon=":material/auto_awesome:", on_click=load_optuna,
              help="The best of 100 Optuna trials run in Databricks, tuned on 2023-2024. Loads its settings and "
                   "scores on 2025-2026 only; click Retrain and backtest.")

groups = tuple(g for g in lab.FEATURE_GROUPS
               if g == "Park, season and weekday" or g in (st.session_state["lab_groups"] or []))
params = {k: st.session_state[f"lab_{k}"] for k in SLIDERS}
params["max_depth"] = params["max_depth"] or -1
params["objective"] = lab.OBJECTIVES[st.session_state["lab_objective"]]

champion = (tuple(lab.CHAMPION_GROUPS), tuple(sorted({**lab.CHAMPION_PARAMS, "objective": "regression_l1"}.items())))
mine_key = (groups, tuple(sorted(params.items())))
is_champion = mine_key == champion   # then "your model" is the champion, line for line
is_optuna = mine_key == (tuple(lab.CHAMPION_GROUPS), tuple(sorted({**lab.OPTUNA_PARAMS, "objective": "regression_l1"}.items())))

with st.spinner("Training four models, one per test year..."):
    champ_preds, _, _ = cached.backtest(*champion)
    preds, importance, seconds = cached.backtest(*mine_key)
base = cached.baselines()

yours_by_year = lab.mae_by_year(preds).assign(method=viz.YOURS)
champ_by_year = lab.mae_by_year(champ_preds).assign(method=viz.CHAMPION)

score_on = st.segmented_control("Score on", list(SCORE_ON), key="lab_score_on", required=True,
                                help="Settings tuned on some test years look better on those years. Score a tuned "
                                     "model only on years its tuning never saw.")
years = SCORE_ON[score_on]
scored, champ_scored = preds[preds["test_year"].isin(years)], champ_preds[champ_preds["test_year"].isin(years)]
yours_mae = float((scored["actual"] - scored["pred"]).abs().mean())
champ_mae = float((champ_scored["actual"] - champ_scored["pred"]).abs().mean())
base_mae = base[base["test_year"].isin(years)].groupby("method")["mae"].mean()
boot = lab.block_bootstrap(scored, champ_scored)
outcome, sentence = lab.verdict(boot)
if is_optuna:
    st.caption(f"Optuna's best: the top trial of a 100-trial study in Databricks (MLflow run `{lab.OPTUNA_RUN[:8]}`), "
               "tuned on 2023-2024. Scored on all years it would look better than it is.")

st.subheader("How your model did", anchor=False)
with st.container(horizontal=True):
    st.metric("Your model, average error", f"{yours_mae:.2f} points", border=True,
              delta=f"{yours_mae - champ_mae:+.2f} vs. champion", delta_color="inverse",
              help="Mean absolute error over every scored park-day: how far a day's prediction typically lands "
                   "from the measured crowd score (0-100).")
    st.metric("Champion", f"{champ_mae:.2f} points", border=True)
    st.metric(f"Best baseline: {base_mae.idxmin().lower()}", f"{base_mae.min():.2f} points", border=True,
              help="Simple rules any forecast should beat.")
    st.metric("Training time", f"{seconds:.1f} s", border=True, help="Four models, one per test year, in this app.")

if is_champion:
    st.info("Change a setting or a feature group and click Retrain and backtest to compare your model with it.",
            title="These are the champion's settings", icon=":material/tune:")
else:
    alert = {"better": st.success, "worse": st.warning, "tie": st.info}[outcome]
    alert(f"{sentence} 95% interval for the difference: {boot['lo']:+.2f} to {boot['hi']:+.2f} points a day.",
          title={"better": "Better than the champion", "worse": "Worse than the champion",
                 "tie": "No real difference"}[outcome],
          icon={"better": ":material/trending_down:", "worse": ":material/trending_up:",
                "tie": ":material/balance:"}[outcome])

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
        b = lab.block_bootstrap(scored, champ_scored, block_days=days)
        rows.append({"Block length": f"{days} days", "Blocks": b["blocks"], "Difference": b["diff"],
                     "95% interval, low": b["lo"], "95% interval, high": b["hi"], "Verdict": lab.verdict(b)[0]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Bootstrap interval for three block lengths",
                 column_config={c: st.column_config.NumberColumn(format="%+.2f")
                                for c in ("Difference", "95% interval, low", "95% interval, high")})

st.subheader("Error by test year", anchor=False)
st.caption("Lower is better. The two grey lines are the baselines. Every year is a model trained only on earlier days; "
           "2026 runs through the snapshot date.")
frame = pd.concat(([] if is_champion else [yours_by_year]) + [champ_by_year, base], ignore_index=True)
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
            *([] if is_champion else [mine.assign(series=viz.YOURS, value=mine["pred"])]),
        ])[["date", "series", "value"]]
        st.altair_chart(viz.prediction_lines(lines), alt=f"Daily measured crowd score and predictions, {park}, {year}")
with right:
    st.subheader("What the champion leans on" if is_champion else "What your model leans on", anchor=False)
    st.caption("Share of the model's total gain (how much each feature's splits cut the error), averaged over the "
               "four test-year models.")
    st.altair_chart(viz.importance(importance, champion=is_champion), alt="Feature importance for your model")
