"""Daily Orlando weather from Open-Meteo (CC BY 4.0): observed history and the 16-day forecast."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

import pandas as pd

# Universal Orlando Resort
LATITUDE, LONGITUDE = 28.47, -81.47
DAILY = ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
         "precipitation_sum", "precipitation_hours", "weather_code"]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_START = date(2018, 1, 1)
# Open-Meteo counts a long date range as many calls (one per two weeks of data), so refetching all
# history since 2018 every day cost ~230 calls a run and drew HTTP 429s from the shared serverless IPs.
# Later runs refetch only the last three months: the archive's newest days are preliminary and get
# revised, and three months covers that with room to spare.
REFETCH_DAYS = 92
# Waits before each retry of a rate-limited or failed request. Open-Meteo's per-minute limit clears
# within a minute; hourly and daily limits don't, so those fail at once.
RETRY_WAITS = (60, 120)


def archive_start(last_observed, today, refetch_days=REFETCH_DAYS):
    """First day to request from the archive: all history on the first run, then a recent window."""
    if last_observed is None:
        return ARCHIVE_START
    return max(ARCHIVE_START, min(last_observed, today) - timedelta(days=refetch_days))


def _reason(error):
    try:
        return json.loads(error.read()).get("reason", "")
    except Exception:
        return ""


def fetch_daily(base_url, opener=urllib.request.urlopen, sleep=time.sleep, waits=RETRY_WAITS, **params):
    """One Open-Meteo daily request as a DataFrame, retrying per-minute rate limits and server errors."""
    query = urllib.parse.urlencode({
        "latitude": LATITUDE, "longitude": LONGITUDE, "daily": ",".join(DAILY),
        "temperature_unit": "fahrenheit", "precipitation_unit": "inch", "timezone": "America/New_York", **params,
    })
    url = f"{base_url}?{query}"
    for attempt in range(len(waits) + 1):
        try:
            with opener(url, timeout=60) as response:
                return pd.DataFrame(json.loads(response.read())["daily"])
        except urllib.error.HTTPError as e:
            reason = _reason(e)
            retryable = e.code >= 500 or (e.code == 429 and "hour" not in reason.lower() and "daily" not in reason.lower())
            if not retryable or attempt == len(waits):
                raise RuntimeError(f"Open-Meteo {base_url} returned HTTP {e.code}"
                                   f"{': ' + reason if reason else ''} (attempt {attempt + 1})") from e
            print(f"Open-Meteo HTTP {e.code}{': ' + reason if reason else ''}; retrying in {waits[attempt]} s")
            sleep(waits[attempt])
