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
| 1b | Historical training data, weather, holidays and school calendars | Next |
| 1c | Daily crowd index + LightGBM forecast, tracked in MLflow | Planned |
| 1d | Staffing optimizer (OR-Tools) driven by the forecast | Planned |
| 1e | Streamlit app: crowd calendar, staffing plan, what-if scenarios | Planned |
| 2 | Agent that explains forecast changes and re-plans staffing, with an eval suite | Later |

Forecast, scenario and optimizer code is written as plain functions with clear inputs and outputs
so phase 2 can call them as agent tools.

## Data

Wait times: Powered by [Queue-Times.com](https://queue-times.com/). Parks (queue-times ids):
Universal Studios Florida (65), Islands of Adventure (64), Epic Universe (334), Volcano Bay (67).

## Databricks

The bundle is in [databricks/](databricks/). Free Edition, serverless only.

```
cd databricks
databricks bundle deploy --profile free
databricks bundle run wait_time_ingest --profile free
```

The `wait-time-ingest` job runs hourly 8am-10pm Eastern. Each run saves the raw JSON to the
`raw_files` volume and merges parsed rows into `raw_wait_times`, keyed on ride and `last_updated`.
