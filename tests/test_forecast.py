import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "src"))

from parks_lib.forecast import _hours_between, score, seasonal_naive


def test_hours_handle_closing_after_midnight():
    hours = _hours_between(pd.Series(["09:00", "18:00", None]), pd.Series(["21:00", "01:00", None]))
    assert hours[0] == 12 and hours[1] == 7 and np.isnan(hours[2])


def test_seasonal_naive_uses_same_weekday_52_weeks_back():
    history = pd.DataFrame({"park_id": [65, 64], "date": [date(2024, 7, 3), date(2024, 7, 3)], "crowd_percent": [70.0, 40.0]})
    target = pd.DataFrame({"park_id": [65, 65], "date": [date(2025, 7, 2), date(2025, 7, 3)]})
    pred = seasonal_naive(history, target)
    assert pred[0] == 70.0 and np.isnan(pred[1])


def test_score_ignores_missing_predictions():
    s = score(np.array([50.0, 60.0, 70.0]), np.array([55.0, np.nan, 90.0]))
    assert s == {"mae": 12.5, "within_10": 0.5, "n": 2}
