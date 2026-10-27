# Databricks notebook source
# MAGIC %md
# MAGIC # 09 Epic Universe impact study
# MAGIC
# MAGIC Did Epic Universe's opening (2025-05-22) change crowds at Universal's older Orlando parks, and
# MAGIC at Disney World and the other Orlando parks? Compares each park with parks Epic could not have
# MAGIC affected (California parks; see `seeds/parks.csv`), using monthly crowd scores with each park's
# MAGIC own seasonal pattern removed (`parks_lib/causal.py`).
# MAGIC
# MAGIC - Before: January 2022 (after the California parks' COVID limits ended) to April 2025.
# MAGIC   May 2025 is left out as the opening month. After: June 2025 to the last full month.
# MAGIC - Methods: difference-in-differences (comparison parks' simple average) and synthetic control
# MAGIC   (the weighted mix of comparison parks that best tracked the park before the opening).
# MAGIC - Checks: each comparison park as a placebo, a fake opening in May 2024, the study without the
# MAGIC   Disneyland parks (their 70th anniversary began the same month), and with Busch Gardens Tampa.
# MAGIC
# MAGIC Writes `epic_impact_summary` (one row per park, method and variant) and `epic_impact_gaps`
# MAGIC (actual vs counterfactual by month).

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "parks")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

import parks_lib
from parks_lib.causal import deseasonalize, monthly_panel, months_between, placebo_test

parks = pd.read_csv(Path(parks_lib.__file__).parents[2] / "seeds" / "parks.csv")
names = dict(zip(parks["park_id"], parks["name"]))
role = dict(zip(parks["park_id"], parks["causal_role"]))
treated = [p for p, r in role.items() if r == "treated"]
donors = [p for p, r in role.items() if r == "donor"]
nearby = [p for p, r in role.items() if r == "nearby"]
orlando = [p for p, r in role.items() if r == "orlando"]

crowd = pd.concat([
    spark.table(f"{catalog}.{schema}.raw_crowd_calendar").toPandas(),
    spark.table(f"{catalog}.{schema}.raw_crowd_calendar_comparison").toPandas(),
])
panel = monthly_panel(crowd)

# Universal's two older parks together, as one more unit
BOTH = 6564
names[BOTH] = "Universal Studios + Islands of Adventure (average)"
role[BOTH] = "treated"
panel[BOTH] = panel[treated].mean(axis=1, skipna=False)

this_month = pd.Timestamp(date.today().replace(day=1))
pre = months_between("2022-01-01", "2025-04-01")
post = months_between("2025-06-01", this_month - pd.DateOffset(months=1))
placebo_pre = months_between("2022-01-01", "2024-04-01")
placebo_post = months_between("2024-06-01", "2025-04-01")
print(f"Before: {len(pre)} months; after: {len(post)} months, through {post[-1]:%Y-%m}")

# COMMAND ----------

runs = []  # (unit, variant, donor pool, before months, after months)
for unit in treated + [BOTH]:
    runs += [
        (unit, "main", donors, pre, post),
        (unit, "without_disneyland", [d for d in donors if d not in (16, 17)], pre, post),
        (unit, "with_busch_gardens_tampa", donors + nearby, pre, post),
        (unit, "placebo_opening_2024_05", donors, placebo_pre, placebo_post),
    ]
runs += [(unit, "main", donors, pre, post) for unit in orlando + nearby]

summary, gaps = [], []
for unit, variant, pool, before, after in runs:
    adj = deseasonalize(panel, before)
    for method in ("diff_in_diff", "synthetic_control"):
        result = placebo_test(adj, unit, pool, before, after, method)
        est = result["treated"]
        pre_mean = float(panel.loc[before, unit].mean())
        summary.append({
            "park_id": unit, "park": names[unit], "role": role[unit], "variant": variant, "method": method,
            "effect_points": round(est.effect, 2), "pre_mean_crowd": round(pre_mean, 1),
            "effect_pct": round(100 * est.effect / pre_mean, 1),
            "pre_rmspe": round(est.pre_rmspe, 2), "rmspe_ratio": round(est.rmspe_ratio, 2),
            "p_value_effect": round(result["p_effect"], 3), "p_value_ratio": round(result["p_ratio"], 3),
            "placebo_effects": json.dumps({names[p.unit]: round(p.effect, 1) for p in result["placebos"]}),
            "n_donors": len(est.weights),
            "weights": json.dumps({names[d]: w for d, w in est.weights.items() if w >= 0.01}),
        })
        offset = est.gap.loc[before].mean()
        for month, gap in est.gap.items():
            actual = panel.loc[month, unit]
            gaps.append({"park_id": unit, "park": names[unit], "variant": variant, "method": method,
                         "month": month.date(), "period": "before" if month in before else "after",
                         "actual": actual, "counterfactual": actual - (gap - offset), "gap": gap - offset})

run_at = datetime.now(timezone.utc)
for name, rows in [("epic_impact_summary", summary), ("epic_impact_gaps", gaps)]:
    (spark.createDataFrame(pd.DataFrame(rows).assign(run_at=run_at))
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{name}"))

display(spark.table(f"{catalog}.{schema}.epic_impact_summary")
        .select("park", "variant", "method", "effect_points", "effect_pct", "p_value_effect", "rmspe_ratio", "weights")
        .orderBy("role", "park", "variant", "method"))
