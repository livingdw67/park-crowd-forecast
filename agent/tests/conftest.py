import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from park_agent.data import ParkData  # noqa: E402
from park_agent.tools import ParkTools  # noqa: E402

START = date(2026, 11, 20)      # covers Thanksgiving (Nov 26) and its long weekend


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def tools():
    """Two parks, 14 days of synthetic forecasts (Nov 27 runs 20 points busier), real seed files."""
    days = [START + timedelta(days=i) for i in range(14)]
    forecast = pd.DataFrame([
        {"park_id": pid, "date": d, "predicted_crowd": base + (20 if d == date(2026, 11, 27) else 0) + i % 3,
         "queue_times_forecast": None, "scored_at": "2026-11-20", "open_time": "09:00", "close_time": "21:00",
         "is_closed": False}
        for pid, base in [(65, 50.4), (64, 44.6)] for i, d in enumerate(days)])
    measured = pd.DataFrame({"park_id": [65], "date": [START - timedelta(days=1)], "measured_crowd": [48]})
    impact = pd.DataFrame({"park": ["Universal Studios Florida"], "method": ["synthetic_control"],
                           "effect_points": [-14.237], "placebo_p": [0.25]})
    return ParkTools(ParkData(forecast, measured, impact, pd.DataFrame()))
