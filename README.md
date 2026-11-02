# Park Crowd Forecast

Forecasts daily crowd levels at the Universal Orlando parks and turns the forecast into a staffing
plan. Built on Databricks (Delta tables, Unity Catalog, MLflow) with a Streamlit front end, plus an
agent that answers questions about the forecast and staffing plans from the same data and optimizer.

**Try it:** [park-crowd-forecast.streamlit.app](https://park-crowd-forecast.streamlit.app/), a public demo with a
model lab, a crowd explorer and the Epic Universe study (no sign-in).

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
| 1g | Epic Universe impact study: difference-in-differences and synthetic control with placebo tests | Live since 2026-10-27 |
| 2 | Agent that explains forecasts and re-plans staffing, with an AI gateway, streaming API and eval suite ([agent/](agent/)) | Built 2026-10-29; distilling a local writer model in progress |
| 3 | Public demo app on a frozen snapshot: model lab, crowd explorer, interactive Epic study ([public_app/](public_app/)) | Built 2026-11-02 |

Forecast, scenario and optimizer code is written as plain functions with clear inputs and outputs,
so the agent calls the same code as agent tools.

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

**Tuning.** `tune-forecast` (on demand, [10_tune_forecast.py](databricks/src/10_tune_forecast.py)) runs an
Optuna search over LightGBM's settings and the calendar feature groups, scored on 2023-2024 only, with each
trial a nested MLflow run and weak trials stopped after their first year. It then looks once at 2025 onward,
which the search never saw, against the settings production uses, and registers the refit as
`crowd_forecast@challenger`. The challenger becomes `@champion` only if a paired block bootstrap puts the
whole 95% interval for its error difference below zero. Every run is logged to `forecast_settings`, and
`train-forecast` retrains with the latest promoted settings, so a promotion survives the weekly retrain.

| Study | Trials | MAE 2023-2024 (tuned on) | MAE 2025-2026 (unseen) vs production | 95% interval | Promoted |
|---|---|---|---|---|---|
| 2026-10-22 (first study, Assistant notebook) | 100 | 13.81 vs 14.14 | 17.39 vs 17.49 | -0.42 to +0.18 | no |
| 2026-11-02 (`tune-forecast`) | 60 (21 stopped early) | 13.92 vs 14.14 | 17.50 vs 17.52 | -0.30 to +0.24 | no |

Tuning buys about 0.3 points on the years it tunes on and nothing reliable on years it hasn't seen: the
error sits near the score's own day-to-day noise, so better features or a better target are the way forward,
not settings. The public app's model lab can load the first study's best settings to check this live.

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

## Epic Universe impact study

**Question:** Epic Universe opened on 2025-05-22. Did it pull crowds from Universal's two older
Orlando parks, and from Disney World and the other Orlando parks?

**Design.** A before/after comparison alone can't separate Epic from everything else that changed
(the economy, travel trends, weather). So each park is compared with parks Epic could not have
affected: seven year-round parks outside Florida (Disneyland, Disney California Adventure, Universal
Studios Hollywood, Knott's Berry Farm, Legoland California, SeaWorld San Diego, Six Flags Magic
Mountain), with their crowd calendars scraped the same way ([seeds/parks.csv](databricks/seeds/parks.csv)).

- Monthly average crowd score per park. Each park's own seasonal pattern (its pre-opening average for
  that calendar month) is removed, since Orlando and California parks peak at different times.
- Before: January 2022 (the California parks' COVID limits had ended) to April 2025. May 2025 is
  left out as the opening month. After: June 2025 to September 2026.
- **Difference-in-differences:** the change at the Orlando park minus the change at the comparison
  parks' simple average.
- **Synthetic control:** the weighted mix of comparison parks (weights >= 0, summing to 1) that best
  tracked the park before the opening stands in for "the park without Epic" afterwards.
- **Checks:** every comparison park rerun as a placebo (an effect as large as the real one should be
  rare among them), a fake opening date in May 2024 (should find nothing), the study without the
  Disneyland parks (their 70th anniversary began the same month), and with Busch Gardens Tampa added.

Code: [causal.py](databricks/src/parks_lib/causal.py) (tested on simulated data with a known effect)
and [09_epic_impact_study.py](databricks/src/09_epic_impact_study.py). The weekly
`comparison-calendar-ingest` job refreshes the comparison parks and reruns the study into
`epic_impact_summary` and `epic_impact_gaps`.

**Results** (crowd score points, range across both methods and three comparison groups):

| Park | Effect after opening | % of pre-Epic level | Fake 2024 opening | Placebo p-value |
|---|---|---|---|---|
| Universal Studios Florida | **-12 to -18** | -20% to -30% | +0.2 / +1.0 | 0.17 to 0.38 |
| Islands of Adventure | -2 to -9 | -4% to -16% | -3.2 / -3.1 | 0.33 to 0.88 |
| Both, averaged | -7 to -14 | -12% to -24% | -1.5 / -0.7 | 0.17 to 0.67 |

What this says:

- **Universal Studios Florida dropped when Epic opened and stayed down.** It tracked its comparison
  within 2 points a year from 2022 to 2024, fell about 7 points below it for the rest of 2025, and
  15-20 points below in 2026. No dip before the opening, so this isn't people delaying trips.
- **Islands of Adventure shows no clear effect.** It was already slipping before Epic (summer 2024,
  matching its 5.5% attendance drop that year), and its post-opening gaps swing both ways.
- **Not statistically conclusive.** With seven comparison parks, the smallest possible placebo
  p-value is 1/8. Disneyland alone moved -21 points as a placebo, for reasons of its own. Without the
  Disneyland parks, Universal Studios' drop is larger than every placebo's.
- **Disney World and the other Orlando parks: inconclusive.** The comparison parks can't reproduce
  Disney's pre-opening path (monthly misses of 5-12 points against 3 for Universal Studios), and the
  two methods disagree (Magic Kingdom -17 vs -7). SeaWorld Orlando, Legoland Florida and Epcot come
  out near zero.
- **A crowd score is not headcount.** It is built from wait times, and lines shrink faster than
  crowds when rides run near capacity, so a 25% lower score likely means a smaller drop in visitors.
  That fits Comcast's comment that Epic took fewer visitors from the older parks than it expected.
  The 2025 TEA/AECOM attendance report, when published, is the check.
- The synthetic control leans on Legoland California (60-90% weight). The simple-average
  difference-in-differences doesn't, and gives a similar answer.

## App

A Streamlit app on Databricks Apps ([databricks/app/](databricks/app/)), deployed by the bundle as
`park-crowds`. Four pages:

- **Crowd calendar:** the 90-day forecast per park as a calendar, next to the last 90 days of measured
  scores and queue-times' own forecast.
- **Staffing plan:** the published plan for a park-day: crew and planned waits by location and hour,
  the shifts, and what the same crew faces if the day runs 15 points busier.
- **What if:** set the crowd, hours, a worker cap, wages and wait costs, then compare three plans
  costed the same way: the published plan, that plan on the scenario's day, and a re-plan
  (`evaluate_plan` and `plan_staffing`). On busy days most rides already run at their maximum crew,
  so re-planning helps less than you'd expect; the page says when that happens.
- **Epic Universe impact:** the impact study for Universal Studios, Islands of Adventure or both:
  actual monthly crowds against the no-Epic estimate, the park's effect beside every placebo park's,
  the fake-2024 check, and the other Orlando parks' (inconclusive) results. Method and comparison
  group are selectable. Cached for six hours; the study reruns weekly.

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

## Agent

[agent/](agent/) answers questions such as "Why is next Friday so busy?" or "What would it cost to
staff Islands of Adventure for a crowd score of 80?". A supervisor routes each question to a forecast
analyst, a staffing planner or both; they call five tools over a snapshot of the Databricks tables
(the staffing tools run the same OR-Tools optimizer as the daily job); a writer streams the answer;
and a validator checks that every number in it came from tool output. All model calls go through an
AI gateway with local-first routing, failover, a circuit breaker and spending caps. A FastAPI service
streams answers as Server-Sent Events.

A 30-question eval set, graded without a model judge (route, tools, grounded numbers, no errors),
passes 30/30 on gpt-5.4-mini at about $0.17 a run. Details, load test results and the distillation
pipeline are in [agent/README.md](agent/README.md).

## Public app

[public_app/](public_app/) is a Streamlit app anyone can open without a Databricks sign-in, live at
[park-crowd-forecast.streamlit.app](https://park-crowd-forecast.streamlit.app/). It runs on a
small snapshot of derived data in [public_app/data/](public_app/data/) (daily crowd scores with the model
features, monthly averages per park, and the 90-day forecast; no scraped pages or ride waits) and the same
`parks_lib` code as the jobs. Every model is trained in the app, so clicking costs nothing.

- **Model lab:** change LightGBM's settings, the loss and the feature groups, retrain, and compare the
  backtest with the champion and the baselines. A paired block bootstrap says whether your model is really
  better or worse, with a check across block lengths. "Load Optuna's best" loads the tuning study's best
  settings and scores them only on 2025-2026, the years the tuning never saw.
- **Crowd explorer:** every measured day since 2018 as a calendar per park and year. Click a day to see
  which feature groups pushed its estimate up or down (SHAP values), plus the next 90 days' forecast.
- **Epic Universe impact:** rerun the impact study live: pick the park, method, comparison parks and
  before period, or fake the opening a year early, and the estimate, placebo test and weights update.

```
.venv\Scripts\python -m pip install -r public_app
equirements.txt
.venv\Scripts\streamlit run public_app\streamlit_app.py
```

Refresh the snapshot with `.venv\Scripts\python public_app\scripts\export_data.py` (needs the
Databricks CLI login); pushing it to `main` redeploys the hosted app. It runs on Streamlit Community Cloud
from this repo with `public_app/streamlit_app.py` as the main file and Python 3.11.

## Local development

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q tests
```

The agent has its own requirements and tests; see [agent/README.md](agent/README.md#run-it).

## Databricks

The bundle is in [databricks/](databricks/). Free Edition, serverless only. The workspace host comes
from the CLI profile, so log in once first:

```
databricks auth login --host <your workspace URL> --profile free
cd databricks
databricks bundle deploy --profile free
databricks bundle run wait_time_ingest --profile free
```

Run the crowd calendar backfill once (later daily runs refresh two months back and two ahead):

```
databricks bundle run crowd_calendar_ingest --profile free --params backfill=true
```

The `daily-features` job (6:30am Eastern) refreshes weather and rebuilds `calendar_features`.

The `comparison-calendar-ingest` job (Mondays 5am Eastern) scrapes the comparison parks in
`seeds/parks.csv` into `raw_crowd_calendar_comparison`, kept apart so the forecast and app only see
Universal, then reruns the Epic Universe impact study. Its first run backfills from July 2021 on its
own; that history was fetched from a laptop and copied into the volume to save Free Edition compute.

The `wait-time-ingest` job runs hourly 8am-10pm Eastern. Each run saves the raw JSON to the
`raw_files` volume and merges parsed rows into `raw_wait_times`, keyed on ride and `last_updated`.

## License

The code is MIT licensed ([LICENSE](LICENSE)); the data isn't. Crowd scores come from Queue-Times.com; only the
derived snapshot in [public_app/data/](public_app/data/) is included, with no scraped pages or ride waits.
