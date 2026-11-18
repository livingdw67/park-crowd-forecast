import io
import json
import sys
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

import watchdog
from watchdog import Check, Databricks, check_freshness, check_job, check_volume, render, short_name

NOW = datetime(2026, 11, 18, 15, 30, tzinfo=timezone.utc)
SCHEDULED = {"job_id": 1, "settings": {"name": "[dev someone] daily-features",
                                       "schedule": {"pause_status": "UNPAUSED"}}}


def run(hours_ago, result="SUCCESS", trigger="PERIODIC", message=""):
    start = NOW - timedelta(hours=hours_ago)
    return {"start_time": int(start.timestamp() * 1000), "trigger": trigger,
            "state": {"result_state": result, "state_message": message}}


def test_short_name_drops_the_development_prefix():
    assert short_name("[dev someone] daily-features") == "daily-features"
    assert short_name("daily-features") == "daily-features"


def test_a_recent_scheduled_success_is_ok():
    check = check_job("daily-features", 26, SCHEDULED, [run(4)], NOW)
    assert check.ok and "4h ago" in check.detail


def test_failing_runs_report_the_latest_error():
    runs = [run(4, "FAILED", message="HTTP Error 429"), run(28, "FAILED"), run(40)]
    check = check_job("daily-features", 26, SCHEDULED, runs, NOW)
    assert check.status == "failing"
    assert "last success 40h ago" in check.detail and "FAILED: HTTP Error 429" in check.detail


def test_a_stopped_scheduler_is_caught_even_after_a_manual_success():
    # November 2026: no scheduled run for five days, then a manual run that succeeded.
    runs = [run(0.5, trigger="ONE_TIME"), run(5 * 24)]
    check = check_job("daily-features", 26, SCHEDULED, runs, NOW)
    assert check.status == "not firing"
    assert "5.0 days" in check.detail and "last success 0h ago" in check.detail


def test_no_runs_at_all_is_not_firing():
    assert check_job("daily-features", 26, SCHEDULED, [], NOW).status == "not firing"


def test_paused_and_missing_jobs():
    paused = {"job_id": 1, "settings": {"name": "x", "schedule": {"pause_status": "PAUSED"}}}
    assert check_job("daily-features", 26, paused, [run(1)], NOW).status == "paused"
    assert check_job("daily-features", 26, None, [], NOW).status == "missing"


def test_freshness_flags_stale_and_empty_tables():
    checks = check_freshness([["raw_wait_times", "2026-11-18 15:00", "0.5"],
                              ["raw_crowd_calendar", "2026-11-13 11:00", "124.5"],
                              ["crowd_forecast", None, None]])
    assert [c.status for c in checks] == ["ok", "stale", "empty"]
    assert "5.2 days ago (limit 26h)" in checks[1].detail


def test_volume_compares_with_a_normal_day():
    assert check_volume("700", "800").ok
    low = check_volume("59", "778.0")
    assert low.status == "low" and "8% of a normal day" in low.detail
    assert check_volume("0", None).status == "no history"


def test_render_puts_problems_first_and_hides_the_workspace_host():
    checks = [Check("job `a`", "ok", "fine"),
              Check("job `b`", "failing", "see https://dbc-123.cloud.databricks.com/jobs/1")]
    report = render(checks, NOW, "https://dbc-123.cloud.databricks.com")
    assert "**1 problem** found." in report
    assert report.index("job `b`") < report.index("job `a`")
    assert "dbc-123" not in report and "<workspace>/jobs/1" in report
    assert "All checks passed." in render([checks[0]], NOW)


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_for(*answers):
    """An opener that answers (or raises) in turn, recording each request's method and URL."""
    calls = []

    def opener(request, timeout):
        calls.append((request.get_method(), request.full_url))
        answer = answers[len(calls) - 1]
        if isinstance(answer, Exception):
            raise answer
        return Response(json.dumps(answer).encode())

    return opener, calls


def test_jobs_follow_page_tokens():
    opener, calls = opener_for({"jobs": [{"job_id": 1}], "next_page_token": "p2"}, {"jobs": [{"job_id": 2}]})
    jobs = Databricks("dbc-1.example.com", "t", opener=opener).jobs()
    assert [j["job_id"] for j in jobs] == [1, 2]
    assert calls[0][1].startswith("https://dbc-1.example.com/api/2.2/jobs/list?") and "page_token=p2" in calls[1][1]


def test_sql_polls_until_the_statement_finishes():
    opener, calls = opener_for({"statement_id": "s1", "status": {"state": "PENDING"}},
                               {"statement_id": "s1", "status": {"state": "SUCCEEDED"},
                                "result": {"data_array": [["1"]]}})
    rows = Databricks("https://h", "t", opener=opener, sleep=lambda s: None).sql("SELECT 1", "w")
    assert rows == [["1"]]
    assert calls == [("POST", "https://h/api/2.0/sql/statements"), ("GET", "https://h/api/2.0/sql/statements/s1")]


def test_a_failed_statement_raises_with_its_message():
    opener, _ = opener_for({"statement_id": "s1", "status": {"state": "FAILED", "error": {"message": "no table"}}})
    try:
        Databricks("https://h", "t", opener=opener).sql("SELECT 1", "w")
    except RuntimeError as e:
        assert "FAILED: no table" in str(e)
    else:
        raise AssertionError("expected a RuntimeError")


def test_an_expired_token_becomes_a_reported_problem():
    denied = urllib.error.HTTPError("https://h", 403, "Forbidden", {}, io.BytesIO(b""))
    opener, _ = opener_for(denied, denied)
    checks = watchdog.run_checks(Databricks("https://h", "t", opener=opener), NOW)
    assert [c.status for c in checks] == ["error", "error"]
    assert "token may have expired" in checks[0].detail


def test_main_exits_1_and_writes_the_report_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    assert watchdog.main(["--report", str(tmp_path / "report.md")]) == 1
    assert "must both be set" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert (tmp_path / "summary.md").exists()


def test_the_monitor_asking_for_a_person_is_a_problem():
    assert watchdog.check_retune([["none", "no trigger fired", "3.0"]]).ok
    asks = watchdog.check_retune([["needs a person", "the last automatic retune found no reliable gain", "3.0"]])
    assert asks.status == "needs a person" and "no reliable gain" in asks.detail
    assert watchdog.check_retune([["none", "", "30.0"]]).status == "stale"
    assert watchdog.check_retune([]).status == "empty"


class FakeDatabricks:
    """Answers each statement by the table it reads; one table is missing."""

    def warehouse_id(self):
        return "w"

    def sql(self, statement, warehouse):
        if "forecast_replay" in statement:
            raise RuntimeError("statement FAILED: [TABLE_OR_VIEW_NOT_FOUND] forecast_replay")
        if "retune_log" in statement:
            return [["start", "error: 28-day error 24.0 against a band up to 21.3", "2.0"]]
        if "WITH days" in statement:
            return [["700", "800"]]
        table = statement.split("'")[1]
        return [[table, "2026-11-18 15:00", "1.0"]]


def test_a_missing_table_is_reported_on_its_own_line():
    checks = {c.name: c for c in watchdog.table_checks(FakeDatabricks())}
    assert checks["watchdog (table forecast_replay)"].status == "error"
    assert checks["table `crowd_forecast`"].ok and checks["volume `raw_wait_times`"].ok
    assert checks["monitor `retune_log`"].ok and "start" in checks["monitor `retune_log`"].detail
