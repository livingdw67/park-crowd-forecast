"""Comparing two forecasts on the same days: a paired block bootstrap of the difference in error.

Why: two models' average errors almost never match, so "0.3 points lower" says nothing on its own; a
different set of test days could flip it. The question is whether the difference is bigger than the luck
of which days were in the test.

Plain meaning: if the 95% interval for (A's error minus B's) sits entirely below zero, A is reliably more
accurate; entirely above, reliably less; across zero, the data can't tell them apart.

Assumptions: days aren't independent (a holiday week or a hurricane moves every day near it), so days are
resampled in blocks of consecutive days, all parks together, and blocks are assumed roughly independent of
one another. That is the shakiest part, so callers check longer blocks too (`block_days` 14 and 28): a
verdict that holds as blocks grow doesn't hinge on it.
"""

import numpy as np
import pandas as pd


def block_bootstrap(a, b, block_days=7, n_boot=4000, seed=0):
    """Paired block bootstrap of the MAE difference, model `a` minus model `b`, over the days both scored.

    `a` and `b`: park_id, date, actual, pred. Returns the observed difference, its 95% percentile interval,
    and how many park-days and blocks it rests on.
    """
    keys = ["park_id", "date"]
    both = a[keys + ["actual", "pred"]].merge(b[keys + ["pred"]], on=keys, suffixes=("", "_b"))
    d = (both["actual"] - both["pred"]).abs() - (both["actual"] - both["pred_b"]).abs()
    day = (pd.to_datetime(both["date"]) - pd.Timestamp("2000-01-03")).dt.days
    blocks = pd.DataFrame({"block": day // block_days, "d": d}).groupby("block")["d"].agg(["sum", "count"])
    sums, counts = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(blocks), size=(n_boot, len(blocks)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "days": int(len(d)), "blocks": int(len(blocks))}


def robust_to_block_length(a, b, block_days=(7, 14, 28)):
    """The bootstrap at several block lengths: one row each, for the sensitivity check."""
    return pd.DataFrame([{"block_days": n, **block_bootstrap(a, b, block_days=n)} for n in block_days])
