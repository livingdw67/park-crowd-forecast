"""Export the tables the agent reads from Databricks into agent/snapshot/ (git-ignored).

    databricks auth login --host <your workspace URL> --profile free
    .venv\\Scripts\\python agent\\scripts\\export_snapshot.py

Reads through the first SQL warehouse in the workspace (on Free Edition, the Serverless Starter
Warehouse) with the CLI profile's credentials. One query per table, about a minute of warehouse time.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from databricks import sql
from databricks.sdk import WorkspaceClient

SCHEMA = os.getenv("PARKS_SCHEMA", "workspace.parks")
OUT = Path(__file__).resolve().parents[1] / "snapshot"
QUERIES = {
    "forecast": f"""SELECT f.park_id, f.date, f.predicted_crowd, f.queue_times_forecast, f.scored_at,
                           c.open_time, c.close_time, coalesce(c.is_closed, false) AS is_closed
                    FROM {SCHEMA}.crowd_forecast f
                    LEFT JOIN {SCHEMA}.raw_crowd_calendar c USING (park_id, date)""",
    "measured": f"""SELECT park_id, date, crowd_percent AS measured_crowd FROM {SCHEMA}.raw_crowd_calendar
                    WHERE NOT is_forecast AND crowd_percent IS NOT NULL AND date >= date_sub(current_date(), 365)""",
    "impact_summary": f"SELECT * FROM {SCHEMA}.epic_impact_summary",
    "impact_gaps": f"SELECT * FROM {SCHEMA}.epic_impact_gaps",
}


def main():
    w = WorkspaceClient(profile=os.getenv("DATABRICKS_CONFIG_PROFILE", "free"))
    warehouse = next(iter(w.warehouses.list()))
    OUT.mkdir(exist_ok=True)
    rows = {}
    with sql.connect(server_hostname=w.config.host.removeprefix("https://"),
                     http_path=f"/sql/1.0/warehouses/{warehouse.id}",
                     credentials_provider=lambda: w.config.authenticate) as conn, conn.cursor() as cur:
        for name, statement in QUERIES.items():
            cur.execute(statement)
            frame = cur.fetchall_arrow().to_pandas()
            frame.to_parquet(OUT / f"{name}.parquet", index=False)
            rows[name] = len(frame)
            print(f"{name}: {len(frame)} rows")
    meta = {"exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "warehouse": warehouse.name,
            "rows": rows}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
