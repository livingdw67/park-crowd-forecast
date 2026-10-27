"""Epic Universe impact study: did its opening (2025-05-22) change crowds at other parks?

Works on monthly average crowd scores per park. Each park's own seasonal pattern is removed first
(its pre-opening average for that calendar month), so Orlando and California parks can be compared
even though their busy seasons differ. Two estimators then build a "no Epic" counterfactual from
comparison parks that Epic could not have affected:

- difference-in-differences: the comparison parks' simple average
- synthetic control: the weighted mix of comparison parks (weights >= 0, summing to 1) that best
  tracked the treated park before the opening

Both report the effect as the average post-opening gap minus the average pre-opening gap, in crowd
score points. Inference comes from placebos: rerunning the method with each comparison park
pretending to be treated (in space) and with a fake opening date (in time).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def monthly_panel(crowd, min_days=10):
    """Average measured crowd score per park and month; months with fewer than `min_days` measured
    open days are left empty. Index: first day of each month; columns: park_id."""
    d = crowd[~crowd["is_closed"].astype(bool) & ~crowd["is_forecast"].astype(bool) & crowd["crowd_percent"].notna()].copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    g = d.groupby(["park_id", "month"])["crowd_percent"].agg(["mean", "count"])
    g.loc[g["count"] < min_days, "mean"] = np.nan
    panel = g["mean"].unstack("park_id").sort_index()
    # Months with no measured days at all still get a (blank) row
    return panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="MS"))


def months_between(first, last):
    return list(pd.date_range(first, last, freq="MS"))


def deseasonalize(panel, pre):
    """Subtract each park's pre-period average for the same calendar month."""
    profile = panel.loc[pre].groupby(panel.loc[pre].index.month).mean()
    return panel - profile.reindex(panel.index.month).to_numpy()


def fit_weights(y, X):
    """Non-negative weights summing to 1 that minimize squared error between y and X @ w."""
    J = X.shape[1]
    result = minimize(
        lambda w: np.sum((y - X @ w) ** 2), np.full(J, 1 / J), method="SLSQP",
        bounds=[(0, 1)] * J, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.clip(result.x, 0, None)
    return w / w.sum()


@dataclass
class Estimate:
    unit: int
    method: str
    weights: dict          # donor park_id -> weight
    gap: pd.Series         # treated minus counterfactual, deseasonalized, every month in pre + post
    pre_rmspe: float
    post_rmspe: float
    effect: float          # mean post gap minus mean pre gap

    @property
    def rmspe_ratio(self):
        return self.post_rmspe / self.pre_rmspe if self.pre_rmspe > 0 else np.inf


def _combine(adj, donors, weights):
    """Weighted donor average per month, re-normalizing over donors present that month."""
    X = adj[donors]
    w = pd.Series(weights, index=donors)
    present = X.notna()
    return (X.fillna(0) * w).sum(axis=1) / (present * w).sum(axis=1).replace(0, np.nan)


def estimate(adj, unit, donors, pre, post, method="synthetic_control", max_missing=0.1):
    """Effect on `unit` of the event between `pre` and `post` (lists of month timestamps)."""
    months = pre + post
    donors = [d for d in donors if d != unit and d in adj and adj.loc[months, d].isna().mean() <= max_missing]
    if method == "synthetic_control":
        fit_months = [m for m in pre if adj.loc[m, [unit] + donors].notna().all()]
        weights = fit_weights(adj.loc[fit_months, unit].to_numpy(), adj.loc[fit_months, donors].to_numpy())
    elif method == "diff_in_diff":
        weights = np.full(len(donors), 1 / len(donors))
    else:
        raise ValueError(method)
    gap = (adj.loc[months, unit] - _combine(adj.loc[months], donors, weights)).rename(unit)
    pre_gap, post_gap = gap.loc[pre].dropna(), gap.loc[post].dropna()
    return Estimate(
        unit=unit, method=method, weights=dict(zip(donors, np.round(weights, 4))), gap=gap,
        pre_rmspe=float(np.sqrt((pre_gap ** 2).mean())), post_rmspe=float(np.sqrt((post_gap ** 2).mean())),
        effect=float(post_gap.mean() - pre_gap.mean()),
    )


def placebo_test(adj, unit, donors, pre, post, method="synthetic_control"):
    """Rerun `method` with each donor pretending to be treated (using the other donors).

    Returns the treated estimate, the placebo estimates, and two permutation p-values: the share of
    all units (treated included) whose post/pre RMSPE ratio is at least the treated one (Abadie's
    test), and the share whose absolute effect is at least the treated one.
    """
    treated = estimate(adj, unit, donors, pre, post, method)
    placebos = [estimate(adj, d, [x for x in donors if x != d], pre, post, method) for d in treated.weights]
    ratios = [p.rmspe_ratio for p in placebos]
    effects = [abs(p.effect) for p in placebos]
    n = len(placebos) + 1
    return {
        "treated": treated,
        "placebos": placebos,
        "p_ratio": (1 + sum(r >= treated.rmspe_ratio for r in ratios)) / n,
        "p_effect": (1 + sum(e >= abs(treated.effect) for e in effects)) / n,
    }
