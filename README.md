# Park Crowd Forecast

Forecasts daily crowd levels at the Universal Orlando parks and turns the forecast into a staffing
plan. Built on Databricks (Delta tables, Unity Catalog, MLflow) with a Streamlit front end.

Daily park attendance isn't published (only annual totals), so ride wait times stand in for it.
The crowd index is built from them, and the forecast should be read as "how busy relative to
other days", not a headcount.

## Status

| Phase | What | State |
|---|---|---|
| 1a | Hourly wait-time ingest into `workspace.parks.raw_wait_times` | Live since 2026-10-22 |
| 1b | Crowd calendar history into `workspace.parks.raw_crowd_calendar` (daily, Jul 2018 on) | Live since 2026-10-22 |
| 1c | Weather, holidays, school calendars, Halloween Horror Nights and other event dates | Next |
| 1d | Daily crowd index + LightGBM forecast, tracked in MLflow | Planned |
| 1e | Staffing optimizer (OR-Tools) driven by the forecast | Planned |
| 1f | Streamlit app: crowd calendar, staffing plan, what-if scenarios | Planned |
| 2 | Agent that explains forecast changes and re-plans staffing, with an eval suite | Later |

Forecast, scenario and optimizer code is written as plain functions with clear inputs and outputs
so phase 2 can call them as agent tools.

## Data

Wait times: Powered by [Queue-Times.com](https://queue-times.com/). Parks (queue-times ids):
Universal Studios Florida (65), Islands of Adventure (64), Epic Universe (334), Volcano Bay (67).

Crowd history: the queue-times crowd calendar, scraped one month page at a time (1.5 seconds apart).
It gives a daily 0-100 crowd score, park hours, weather and holiday flags. Measured history starts
mid-July 2018 for Universal Studios and Islands of Adventure, May 2025 for Epic Universe and May 2026
for Volcano Bay. The score is queue-times' own and its formula isn't published, so the crowd index
built from our ride-level data runs alongside it. Scraped pages and data stay out of this repo.

Caveats found so far:
- The score looks relative to each park's own busy days: Epic Universe reads 1-6% through
  September 2026 while its live waits were the longest in the resort.
- Halloween Horror Nights haunted houses show up as rides with some of the longest waits of the
  year, so event attractions must be left out of any crowd index.
- Days before tracking started and future days carry queue-times' forecast (`is_forecast`); days
  they don't forecast have park hours and no score.

## Databricks

The bundle is in [databricks/](databricks/). Free Edition, serverless only.

```
cd databricks
databricks bundle deploy --profile free
databricks bundle run wait_time_ingest --profile free
```

Run the crowd calendar backfill once (later daily runs refresh two months back and two ahead):

```
databricks bundle run crowd_calendar_ingest --profile free --params backfill=true
```

The `wait-time-ingest` job runs hourly 8am-10pm Eastern. Each run saves the raw JSON to the
`raw_files` volume and merges parsed rows into `raw_wait_times`, keyed on ride and `last_updated`.
