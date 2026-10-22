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
| 1c | `raw_weather_daily` (Open-Meteo) and `calendar_features` (holidays, school breaks, events) | Live since 2026-10-22 |
| 1d | LightGBM forecast in MLflow / Unity Catalog (`crowd_forecast`), ride-based `daily_crowd_index` | Live since 2026-10-22 |
| 1e | Staffing optimizer (OR-Tools MILP) planning 14 days from the forecast | Live since 2026-10-22 |
| 1f | Streamlit app on Databricks Apps: crowd calendar, staffing plan, what-if scenarios | Live since 2026-10-22 |
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

## Features

Everything in `calendar_features` is known ahead of time, so it can be built for any future date.

- **Holidays:** US federal holidays plus Easter, long weekends, days to the next and since the last.
- **School breaks:** rule-based windows ([calendar.py](databricks/src/parks_lib/calendar.py)) rather
  than one district's calendar, since visitors come from everywhere. Checked against TouringPlans'
  Florida in-session rates for 2015-2021: on weekdays flagged Florida summer, winter break or
  Thanksgiving week, 6-20% of Florida students were in school, against 92.5% on unflagged weekdays.
  Florida spring break is mid-March, not Easter (71% still in school during Easter week), so it has
  its own window.
- **Events:** [seeds/events.csv](databricks/seeds/events.csv) lists Halloween Horror Nights runs,
  the COVID closure and Epic Universe's opening, with a source for each row.
- **Weather:** Open-Meteo (CC BY 4.0) observed history from 2018 plus a 16-day forecast. Beyond
  16 days there is no weather row.

First look, Universal Studios Florida, average daily crowd score (measured days, 2018-2026):

| Days | Avg score | Days |
|---|---|---|
| All | 50.8 | 2,932 |
| Weekends | 58.4 | 839 |
| Easter week | 84.3 | 70 |
| Winter break | 81.8 | 148 |
| Thanksgiving week | 74.8 | 72 |
| Holiday weekends | 73.1 | 188 |
| Florida spring break window | 67.7 | 121 |
| Halloween Horror Nights season | 40.0 | 476 |
| September-October, no HHN | 42.9 | 82 |
| 1+ inch of rain | 43.4 | 90 |
| Since Epic Universe opened | 42.7 | 518 |

Easter week beats Florida's own spring break: the crowd is national, not local. HHN doesn't lift
the daytime score.

## Forecast

One LightGBM model for all parks predicts the daily crowd score. Backtest: for each test year,
train on every earlier day and score that year (mean absolute error, 0-100 scale, all parks):

| Method | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| Baseline: park x month x weekday mean | 20.4 | 17.0 | 19.2 | 24.2 |
| Baseline: same weekday last year | 16.7 | 18.1 | 18.4 | 21.9 |
| LightGBM: calendar | **15.9** | **12.4** | 16.2 | **18.8** |
| LightGBM: calendar + park hours | 17.2 | 13.7 | 16.1 | 19.3 |
| LightGBM: calendar + hours + weather | 16.5 | 13.4 | **15.9** | 18.9 |

What this says:

- **The target is noisy.** The score moves 14-17 points day to day at Universal Studios and Islands
  of Adventure. With each park-year's average bias removed, the error is 11-14, close to that floor.
  Correlation with actuals is about 0.77, and weekly averages are off by 8-13 points.
- **Most of the rest is level shifts.** 2025 overpredicted Universal Studios by 11 points: models
  trained before 2025 had never seen Epic Universe open. Epic itself is overpredicted by 20 in 2026,
  as its score fell sharply after its opening year.
- **Hours and weather add nothing measurable**, so production uses calendar features only, which
  also works at any horizon. Weighting recent days more (half-lives of 180 to 730 days) didn't help.

`train-forecast` (Mondays) reruns the backtest into MLflow, retrains on all days, and registers
`workspace.parks.crowd_forecast` with the `champion` alias. `daily-features` scores the next 90 days
into `crowd_forecast` (latest) and `crowd_forecast_history` (every run, for tracking accuracy later),
next to queue-times' own forecast where they publish one.

`daily_crowd_index` is the transparent alternative: average posted wait across open rides between
11am and 5pm, without single-rider lines and Halloween Horror Nights houses. It needs a few weeks of
hourly data before it can be compared with the queue-times score.

## Staffing optimizer

[staffing.py](databricks/src/parks_lib/staffing.py) is a mixed-integer program (OR-Tools, SCIP;
each park-day solves in well under a second). For every ride and food outlet it picks how many
workers start each shift (8-hour shifts every 2 hours, 4-hour shifts every hour), minimizing wages
plus a cost per guest-hour in line plus a penalty per guest-hour over a 60-minute target. Rules:
each location's crew stays between its minimum and maximum, guests served per hour are capped by
crew, lines carry from hour to hour, an optional park-wide headcount cap forces trade-offs, and
guests skip long lines (the share who skip rises with the line, everyone at a 4-hour line), which
keeps lines bounded and gives a "guests who skipped a ride" measure.

**Demand is assumed, not measured.** The crowd score maps to attendance per open hour (25,000 to
50,000 guests on a 12-hour day), each location sees a share of guests
([staffing_locations.csv](databricks/seeds/staffing_locations.csv)), and arrivals peak early
afternoon. Ride shares were calibrated by bisection so the plan's 11am-5pm waits match observed
posted waits on 2026-10-22, one day of data. Recalibrating over a few weeks of `wait_times_clean`
is the next step for this part. Known gaps: posted waits run higher than actual ones; long
operating days spread the arrival peak, which overstates waits at Islands of Adventure; on
Halloween Horror Nights nights the day plan ends at 5pm.

`daily-features` plans the next 14 days into `staffing_plan_summary`, `staffing_plan_shifts` and
`staffing_plan_hourly`. Each day's plan is also replayed at the forecast plus 15 points (about one
typical forecast error) to show what an underestimate would cost in waits and skipped guests.

## App

A Streamlit app on Databricks Apps ([databricks/app/](databricks/app/)), deployed by the bundle as
`park-crowds`. Three pages:

- **Crowd calendar:** the 90-day forecast per park as a calendar, next to the last 90 days of measured
  scores and queue-times' own forecast.
- **Staffing plan:** the published plan for a park-day: crew and planned waits by location and hour,
  the shifts, and what the same crew faces if the day runs 15 points busier.
- **What if:** set the crowd, hours, a worker cap, wages and wait costs, then compare three plans
  costed the same way: the published plan, that plan on the scenario's day, and a re-plan
  (`evaluate_plan` and `plan_staffing`). On busy days most rides already run at their maximum crew,
  so re-planning helps less than you'd expect; the page says when that happens.

The app reads the Delta tables through the SQL warehouse as its own service principal and caches
them for an hour; nothing it shows is computed at request time except the what-if plans. Its source
root is the bundle folder, so it ships with the same `parks_lib` and `seeds/staffing_locations.csv`
the daily job plans from. Colours follow the brand guide's dark palettes, checked with a palette
validator for contrast and colour-vision deficiency.

Deploy and start (Free Edition stops an app 24 hours after it starts; rerun to restart):

```
cd databricks
databricks bundle deploy --profile free
databricks bundle run park_app --profile free
```

After the first deploy, give the app's service principal read access (its client id is in
`databricks apps get park-crowds`):

```sql
GRANT USE CATALOG ON CATALOG workspace TO `<client id>`;
GRANT USE SCHEMA, SELECT ON SCHEMA workspace.parks TO `<client id>`;
```

Run it locally against the workspace, from `databricks/` so the theme in `.streamlit/` applies:

```
set DATABRICKS_CONFIG_PROFILE=free
set DATABRICKS_WAREHOUSE_ID=<warehouse id>
..\.venv\Scripts\streamlit run app/streamlit_app.py
```

`ortools==9.11` needs protobuf 5.26, so the app pins `databricks-sdk==0.72.0`, the newest release
that allows it ([databricks/requirements.txt](databricks/requirements.txt)).

## Local development

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q tests
```

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

The `daily-features` job (6:30am Eastern) refreshes weather and rebuilds `calendar_features`.

The `wait-time-ingest` job runs hourly 8am-10pm Eastern. Each run saves the raw JSON to the
`raw_files` volume and merges parsed rows into `raw_wait_times`, keyed on ride and `last_updated`.
