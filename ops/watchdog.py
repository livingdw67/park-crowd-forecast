"""Pipeline watchdog: checks the Databricks jobs and tables from outside Databricks.

A check that runs inside Databricks can't report a scheduler that has stopped, because it stops with it.
That happened in November 2026: no job ran for five days while every schedule still read UNPAUSED. So
this runs once a day on GitHub Actions (.github/workflows/watchdog.yml) and asks the Databricks REST API:

1. Jobs. Has each scheduled job succeeded recently? Has the scheduler itself started a run recently?
   The second question separates "the scheduler stopped" from "the runs are failing".
2. Tables. Freshness (the newest load stamp in each table) and volume (wait-time rows in the last
   24 hours against a normal day), because a job can succeed and still write nothing.

Any problem exits 1; the workflow then opens or comments on a GitHub issue. The issue is public, so the
report leaves out the workspace URL and development mode's "[dev user]" name prefix.

Needs DATABRICKS_HOST and DATABRICKS_TOKEN (DATABRICKS_WAREHOUSE_ID is optional). Standard library only:
    python ops/watchdog.py [--report report.md]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

CATALOG, SCHEMA = "workspace", "parks"

# Scheduled jobs and the most hours allowed since their last success, and since the scheduler last
# started one. Each limit is the schedule's gap plus a little slack, judged at the workflow's check time
# (about 10:30am Eastern): a daily job that missed this morning is 28+ hours old by then, and the hourly
# ingest is 12+ hours old only if every run since 10pm missed.
JOB_LIMITS_H = {
    "wait-time-ingest": 12,                     # hourly 8am-10pm ET; the overnight gap is 10 hours
    "crowd-calendar-ingest": 26,                # daily 6:00 ET
    "daily-features": 26,                       # daily 6:30 ET
    "train-forecast": 7 * 24 + 2,               # Mondays 7:00 ET
    "comparison-calendar-ingest": 7 * 24 + 2,   # Mondays 5:00 ET
}

# Tables, the column that stamps each load, and the most hours allowed since the newest load.
TABLE_LIMITS_H = {
    "raw_wait_times": ("fetched_at", 12),
    "raw_crowd_calendar": ("scraped_at", 26),
    "raw_weather_daily": ("fetched_at", 26),
    "crowd_forecast": ("scored_at", 26),
}

# Wait-time rows fetched in the last 24 hours must reach this share of a normal day: the median of the
# previous 14 days that had any rows. Days with none are left out, so an outage doesn't lower the bar.
MIN_VOLUME_SHARE = 0.5

RUNS_TO_READ = 25


@dataclass
class Check:
    name: str
    status: str   # "ok", or a short problem label
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def short_name(name: str) -> str:
    """The job name without development mode's "[dev user] " prefix."""
    return re.sub(r"^\[[^\]]*\]\s*", "", name)


def hours_since(ms: int, now: datetime) -> float:
    return (now - datetime.fromtimestamp(ms / 1000, timezone.utc)).total_seconds() / 3600


def fmt_age(hours: float) -> str:
    return f"{hours:.0f}h" if hours < 48 else f"{hours / 24:.1f} days"


def check_job(name: str, limit_h: float, job: dict | None, runs: list[dict], now: datetime) -> Check:
    """One scheduled job, from its settings and its completed runs (newest first)."""
    label = f"job `{name}`"
    if job is None:
        return Check(label, "missing", "not deployed in the workspace")
    if job.get("settings", {}).get("schedule", {}).get("pause_status") == "PAUSED":
        return Check(label, "paused", "the schedule is paused")

    success = next((r for r in runs if r.get("state", {}).get("result_state") == "SUCCESS"), None)
    scheduled = next((r for r in runs if r.get("trigger") == "PERIODIC"), None)
    success_age = hours_since(success["start_time"], now) if success else None
    since_success = f"last success {fmt_age(success_age)} ago" if success else f"no success in the last {len(runs)} runs"

    if scheduled is None or hours_since(scheduled["start_time"], now) > limit_h:
        when = f"in {fmt_age(hours_since(scheduled['start_time'], now))}" if scheduled else "in its recent runs"
        return Check(label, "not firing", f"the scheduler hasn't started a run {when} (limit {limit_h}h); {since_success}")
    if success is None or success_age > limit_h:
        latest = runs[0].get("state", {})
        message = (latest.get("state_message") or "").strip()[:150]
        return Check(label, "failing", f"{since_success} (limit {limit_h}h); latest run "
                                       f"{latest.get('result_state', 'unknown')}" + (f": {message}" if message else ""))
    return Check(label, "ok", since_success)


def check_freshness(rows: list[list]) -> list[Check]:
    """Rows of [table, latest load stamp, hours since it] from freshness_sql."""
    checks = []
    for table, latest, age_h in rows:
        label, limit_h = f"table `{table}`", TABLE_LIMITS_H[table][1]
        if latest is None:
            checks.append(Check(label, "empty", "no rows"))
        elif float(age_h) > limit_h:
            checks.append(Check(label, "stale", f"newest load {fmt_age(float(age_h))} ago (limit {limit_h}h)"))
        else:
            checks.append(Check(label, "ok", f"newest load {fmt_age(float(age_h))} ago"))
    return checks


def check_volume(last_24h, normal_day) -> Check:
    """Wait-time rows in the last 24 hours against the median of the previous 14 days."""
    label = "volume `raw_wait_times`"
    if normal_day is None or float(normal_day) == 0:
        return Check(label, "no history", "no rows in the previous 14 days to compare with")
    share = int(last_24h or 0) / float(normal_day)
    detail = f"{int(last_24h or 0):,} rows in the last 24h, {share:.0%} of a normal day ({float(normal_day):,.0f})"
    return Check(label, "ok" if share >= MIN_VOLUME_SHARE else "low", detail)


def freshness_sql(catalog: str = CATALOG, schema: str = SCHEMA) -> str:
    return " UNION ALL ".join(
        f"SELECT '{table}' AS tbl, CAST(max({column}) AS STRING) AS latest, "
        f"timestampdiff(MINUTE, max({column}), current_timestamp()) / 60.0 AS age_h "
        f"FROM {catalog}.{schema}.{table}"
        for table, (column, _) in TABLE_LIMITS_H.items())


def volume_sql(catalog: str = CATALOG, schema: str = SCHEMA) -> str:
    return (
        "WITH days AS ("
        " SELECT floor(timestampdiff(MINUTE, fetched_at, current_timestamp()) / 1440) AS days_ago, count(*) AS n"
        f" FROM {catalog}.{schema}.raw_wait_times"
        " WHERE fetched_at >= current_timestamp() - INTERVAL 15 DAYS GROUP BY 1) "
        "SELECT coalesce(sum(n) FILTER (WHERE days_ago = 0), 0) AS last_24h,"
        " percentile(n, 0.5) FILTER (WHERE days_ago BETWEEN 1 AND 14) AS normal_day FROM days")


class Databricks:
    """The few REST calls the watchdog needs. `opener` is urllib's urlopen, swappable in tests."""

    def __init__(self, host: str, token: str, opener=urllib.request.urlopen, sleep=time.sleep):
        self.host = (host if host.startswith("http") else f"https://{host}").rstrip("/")
        self.token, self.opener, self.sleep = token, opener, sleep

    def call(self, method: str, path: str, params: dict | None = None, body: dict | None = None) -> dict:
        url = self.host + path + (f"?{urllib.parse.urlencode(params)}" if params else "")
        request = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                         headers={"Authorization": f"Bearer {self.token}",
                                                  "Content-Type": "application/json"})
        with self.opener(request, timeout=60) as response:
            return json.load(response)

    def jobs(self) -> list[dict]:
        jobs, params = [], {"limit": 100}
        while True:
            page = self.call("GET", "/api/2.2/jobs/list", params)
            jobs += page.get("jobs", [])
            if not page.get("next_page_token"):
                return jobs
            params = {"limit": 100, "page_token": page["next_page_token"]}

    def runs(self, job_id: int) -> list[dict]:
        page = self.call("GET", "/api/2.2/jobs/runs/list",
                         {"job_id": job_id, "completed_only": "true", "limit": RUNS_TO_READ})
        return page.get("runs", [])

    def warehouse_id(self) -> str:
        if os.environ.get("DATABRICKS_WAREHOUSE_ID"):
            return os.environ["DATABRICKS_WAREHOUSE_ID"]
        warehouses = self.call("GET", "/api/2.0/sql/warehouses").get("warehouses", [])
        if not warehouses:
            raise RuntimeError("no SQL warehouse in the workspace")
        return warehouses[0]["id"]

    def sql(self, statement: str, warehouse_id: str, poll_s: float = 5, max_polls: int = 36) -> list[list]:
        """Run one statement on the warehouse (it may need to start) and return its rows."""
        result = self.call("POST", "/api/2.0/sql/statements", body={
            "warehouse_id": warehouse_id, "statement": statement, "wait_timeout": "50s",
            "format": "JSON_ARRAY", "disposition": "INLINE", "on_wait_timeout": "CONTINUE"})
        for _ in range(max_polls):
            state = result.get("status", {}).get("state")
            if state == "SUCCEEDED":
                return result.get("result", {}).get("data_array", [])
            if state not in ("PENDING", "RUNNING"):
                error = result.get("status", {}).get("error", {}).get("message", "")
                raise RuntimeError(f"statement {state}: {error[:200]}")
            self.sleep(poll_s)
            result = self.call("GET", f"/api/2.0/sql/statements/{result['statement_id']}")
        raise RuntimeError("statement still running after the polling limit")


def job_checks(db: Databricks, now: datetime) -> list[Check]:
    by_name = {}
    for job in db.jobs():
        by_name.setdefault(short_name(job.get("settings", {}).get("name", "")), job)
    return [check_job(name, limit, by_name.get(name), db.runs(by_name[name]["job_id"]) if name in by_name else [], now)
            for name, limit in JOB_LIMITS_H.items()]


def table_checks(db: Databricks) -> list[Check]:
    warehouse = db.warehouse_id()
    checks = check_freshness(db.sql(freshness_sql(), warehouse))
    rows = db.sql(volume_sql(), warehouse)
    return checks + [check_volume(*rows[0]) if rows else check_volume(0, None)]


def safely(area: str, run) -> list[Check]:
    """A check that can't run is itself a problem (an expired token, say), not a crash."""
    try:
        return run()
    except urllib.error.HTTPError as e:
        hint = ": the token may have expired or been revoked" if e.code in (401, 403) else ""
        return [Check(f"watchdog ({area})", "error", f"HTTP {e.code} {e.reason}{hint}")]
    except Exception as e:  # noqa: BLE001, report anything rather than die without a report
        return [Check(f"watchdog ({area})", "error", f"{type(e).__name__}: {e}"[:300])]


def run_checks(db: Databricks, now: datetime) -> list[Check]:
    return safely("jobs", lambda: job_checks(db, now)) + safely("tables", lambda: table_checks(db))


def render(checks: list[Check], now: datetime, host: str = "") -> str:
    problems = [c for c in checks if not c.ok]
    headline = (f"**{len(problems)} problem{'s' if len(problems) != 1 else ''}** found." if problems
                else "All checks passed.")
    lines = [f"## Pipeline watchdog, {now:%Y-%m-%d %H:%M} UTC", "", headline, "",
             "| Check | Status | Detail |", "|---|---|---|"]
    lines += [f"| {c.name} | {'ok' if c.ok else f'**{c.status}**'} | {c.detail.replace('|', '/')} |"
              for c in sorted(checks, key=lambda c: c.ok)]
    report = "\n".join(lines) + "\n"
    bare_host = re.sub(r"^https?://", "", host).rstrip("/")
    return report.replace(bare_host, "<workspace>") if bare_host else report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", help="also write the markdown report to this file")
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    host, token = os.environ.get("DATABRICKS_HOST", ""), os.environ.get("DATABRICKS_TOKEN", "")
    if host and token:
        checks = run_checks(Databricks(host, token), now)
    else:
        checks = [Check("watchdog", "error", "DATABRICKS_HOST and DATABRICKS_TOKEN must both be set")]

    report = render(checks, now, host)
    print(report)
    for path, mode in [(args.report, "w"), (os.environ.get("GITHUB_STEP_SUMMARY"), "a")]:
        if path:
            with open(path, mode, encoding="utf-8") as out:
                out.write(report)
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
