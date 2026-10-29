"""What the agent's tools read: a snapshot of the Databricks tables plus the repo's seed files.

The snapshot (agent/snapshot/*.parquet, git-ignored because it holds queue-times data) is written by
scripts/export_snapshot.py. Staffing plans aren't in it: they are a pure function of the forecast and
seeds/staffing_locations.csv, so the tools recompute them the way the daily job does. Calendar
features are rebuilt from seeds/events.csv with parks_lib, the same code the job runs.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SEEDS = REPO / "databricks" / "seeds"
SNAPSHOT = REPO / "agent" / "snapshot"
sys.path.insert(0, str(REPO / "databricks" / "src"))

from parks_lib.calendar import build_calendar_features  # noqa: E402

PARKS = {65: "Universal Studios Florida", 64: "Islands of Adventure", 334: "Epic Universe", 67: "Volcano Bay"}
TABLES = ["forecast", "measured", "impact_summary", "impact_gaps"]


@dataclass
class ParkData:
    forecast: pd.DataFrame        # park_id, date, predicted_crowd, queue_times_forecast, scored_at, open_time, close_time, is_closed
    measured: pd.DataFrame        # park_id, date, measured_crowd
    impact_summary: pd.DataFrame  # the Epic Universe study, one row per park and method
    impact_gaps: pd.DataFrame     # monthly actual vs. estimate, per park
    locations: pd.DataFrame = field(default_factory=lambda: pd.read_csv(SEEDS / "staffing_locations.csv"))
    events: pd.DataFrame = field(default_factory=lambda: pd.read_csv(SEEDS / "events.csv", parse_dates=["start_date", "end_date"]))
    exported_at: str = ""

    def __post_init__(self):
        for frame in (self.forecast, self.measured):
            frame["date"] = pd.to_datetime(frame["date"]).dt.date
        for col in ("start_date", "end_date"):
            self.events[col] = pd.to_datetime(self.events[col]).dt.date

    @classmethod
    def load(cls, folder=SNAPSHOT):
        folder = Path(folder)
        missing = [t for t in TABLES if not (folder / f"{t}.parquet").exists()]
        if missing:
            raise FileNotFoundError(f"snapshot is missing {missing}; run agent/scripts/export_snapshot.py")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8")) if (folder / "meta.json").exists() else {}
        return cls(**{t: pd.read_parquet(folder / f"{t}.parquet") for t in TABLES}, exported_at=meta.get("exported_at", ""))

    @cached_property
    def calendar(self):
        """Calendar features for every date the forecast or recent history covers."""
        dates = list(self.forecast["date"]) + list(self.measured["date"])
        return build_calendar_features(min(dates), max(dates), self.events).set_index("date")

    def forecast_dates(self):
        return sorted(self.forecast["date"].unique())

    def today(self):
        """The first forecast date: the snapshot's 'today', so answers stay fixed for a given snapshot."""
        return self.forecast_dates()[0] if len(self.forecast) else date.today()
