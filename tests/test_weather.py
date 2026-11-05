import io
import json
import sys
import urllib.error
import urllib.parse
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "src"))

from parks_lib.weather import ARCHIVE_START, ARCHIVE_URL, DAILY, archive_start, fetch_daily

TODAY = date(2026, 11, 5)
BODY = {"daily": {"time": ["2026-11-03", "2026-11-04"], **{v: [1.0, 2.0] for v in DAILY}}}


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, reason=None):
    body = json.dumps({"error": True, "reason": reason}).encode() if reason else b""
    return urllib.error.HTTPError(ARCHIVE_URL, code, "error", {}, io.BytesIO(body))


def opener_for(*outcomes):
    """An opener that raises or answers in turn, recording the URLs it was asked for."""
    calls = []

    def opener(url, timeout):
        calls.append(url)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return Response(json.dumps(outcome).encode())

    return opener, calls


def test_archive_start_first_run_loads_all_history():
    assert archive_start(None, TODAY) == ARCHIVE_START


def test_archive_start_refetches_a_recent_window():
    assert archive_start(date(2026, 11, 3), TODAY, refetch_days=92) == date(2026, 8, 3)


def test_archive_start_never_goes_before_history_or_after_today():
    assert archive_start(date(2018, 2, 1), TODAY) == ARCHIVE_START
    assert archive_start(date(2027, 1, 1), TODAY, refetch_days=10) == date(2026, 10, 26)


def test_fetch_daily_builds_the_query_and_returns_a_frame():
    opener, calls = opener_for(BODY)
    df = fetch_daily(ARCHIVE_URL, opener=opener, start_date="2026-11-03", end_date="2026-11-04")
    assert list(df["time"]) == ["2026-11-03", "2026-11-04"]
    query = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0]).query)
    assert query["start_date"] == ["2026-11-03"] and query["daily"] == [",".join(DAILY)]
    assert query["temperature_unit"] == ["fahrenheit"]


def test_fetch_daily_retries_a_per_minute_rate_limit():
    waits = []
    opener, calls = opener_for(http_error(429, "Minutely API request limit exceeded. Please try again in one minute."), BODY)
    df = fetch_daily(ARCHIVE_URL, opener=opener, sleep=waits.append, waits=(60, 120))
    assert len(calls) == 2 and waits == [60] and len(df) == 2


def test_fetch_daily_retries_server_errors_then_gives_up_with_the_code():
    waits = []
    opener, calls = opener_for(http_error(502), http_error(503), http_error(503))
    with pytest.raises(RuntimeError, match="HTTP 503.*attempt 3"):
        fetch_daily(ARCHIVE_URL, opener=opener, sleep=waits.append, waits=(60, 120))
    assert waits == [60, 120] and len(calls) == 3


@pytest.mark.parametrize("reason", ["Hourly API request limit exceeded. Please try again in the next hour.",
                                    "Daily API request limit exceeded. Please try again tomorrow."])
def test_fetch_daily_does_not_wait_out_hourly_or_daily_limits(reason):
    waits = []
    opener, calls = opener_for(http_error(429, reason))
    with pytest.raises(RuntimeError, match="HTTP 429: " + reason.split(".")[0]):
        fetch_daily(ARCHIVE_URL, opener=opener, sleep=waits.append)
    assert waits == [] and len(calls) == 1


def test_fetch_daily_does_not_retry_a_bad_request():
    opener, calls = opener_for(http_error(400, "Parameter 'start_date' is out of range"))
    with pytest.raises(RuntimeError, match="HTTP 400: Parameter 'start_date' is out of range"):
        fetch_daily(ARCHIVE_URL, opener=opener, sleep=lambda s: None)
    assert len(calls) == 1
