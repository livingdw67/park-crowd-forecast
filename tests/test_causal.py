import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "src"))

from parks_lib.causal import deseasonalize, estimate, fit_weights, monthly_panel, months_between, placebo_test

PRE = months_between("2022-01-01", "2025-04-01")
POST = months_between("2025-06-01", "2026-09-01")


def simulated_panel(effect=-10.0, seed=0):
    """Six donor parks with their own trend and seasonality; park 1 is half donor 101 and half
    donor 102, plus `effect` from June 2025."""
    rng = np.random.default_rng(seed)
    months = months_between("2022-01-01", "2026-09-01")
    season = np.sin(2 * np.pi * np.array([m.month for m in months]) / 12)
    donors = {}
    for i, d in enumerate(range(101, 107)):
        trend = np.cumsum(rng.normal(0, 1.5, len(months)))
        donors[d] = 50 + 8 * season * (1 + 0.2 * i) + trend + rng.normal(0, 0.5, len(months))
    panel = pd.DataFrame(donors, index=months)
    treated = 0.5 * panel[101] + 0.5 * panel[102] + 5 + rng.normal(0, 0.5, len(months))
    treated[panel.index >= pd.Timestamp("2025-06-01")] += effect
    panel[1] = treated
    return panel


def test_fit_weights_recovers_a_known_mix():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 4))
    w = fit_weights(X @ np.array([0.7, 0.3, 0, 0]), X)
    assert np.allclose(w, [0.7, 0.3, 0, 0], atol=1e-3)


def test_synthetic_control_recovers_effect_and_weights():
    panel = simulated_panel(effect=-10)
    adj = deseasonalize(panel, PRE)
    est = estimate(adj, 1, list(range(101, 107)), PRE, POST)
    assert abs(est.effect - (-10)) < 1.5
    assert est.weights[101] + est.weights[102] > 0.8


def test_no_effect_means_estimate_near_zero_and_placebos_not_significant():
    panel = simulated_panel(effect=0, seed=3)
    result = placebo_test(deseasonalize(panel, PRE), 1, list(range(101, 107)), PRE, POST)
    assert abs(result["treated"].effect) < 1.5
    assert result["p_effect"] > 0.3


def test_large_effect_ranks_first_among_placebos():
    panel = simulated_panel(effect=-25, seed=4)
    result = placebo_test(deseasonalize(panel, PRE), 1, list(range(101, 107)), PRE, POST)
    assert result["p_effect"] == 1 / 7


def test_monthly_panel_skips_thin_months_and_closed_days():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(70)]
    crowd = pd.DataFrame({
        "park_id": 65, "date": days, "crowd_percent": [40.0] * 70,
        "is_closed": [False] * 70, "is_forecast": [False] * 31 + [True] * 28 + [False] * 11,
    })
    crowd.loc[0, "is_closed"] = True
    crowd.loc[0, "crowd_percent"] = 99.0
    panel = monthly_panel(crowd)
    assert panel.loc[pd.Timestamp("2025-01-01"), 65] == 40.0
    assert np.isnan(panel.loc[pd.Timestamp("2025-02-01"), 65])  # only forecast days in February
    assert panel.loc[pd.Timestamp("2025-03-01"), 65] == 40.0
