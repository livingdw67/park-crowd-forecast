"""Daily crowd forecast: training frame, baselines, backtest, fit and predict.

The target is the queue-times daily crowd score (0-100) per park. One LightGBM model covers all
parks, with park_id as a categorical feature. Feature sets are nested so the backtest shows what
each one adds:

- calendar: known for any date, so it works at any horizon
- hours: park hours, which Universal publishes a couple of months ahead (and extends on busy days)
- weather: observed weather in training; only usable within the 16-day weather forecast
"""

import numpy as np
import pandas as pd

CALENDAR_FEATURES = [
    "park_id", "year", "month", "day_of_year", "week_of_year", "day_of_week", "is_weekend",
    "is_holiday", "is_holiday_weekend", "days_to_next_holiday", "days_since_last_holiday",
    "thanksgiving_week", "winter_break", "spring_break_season", "easter_week", "fl_spring_break",
    "fl_summer", "us_summer", "event_hhn", "event_epic_universe_open",
]
HOURS_FEATURES = ["open_hours", "has_extra_hours"]
WEATHER_FEATURES = ["temperature_2m_max", "apparent_temperature_max", "precipitation_sum", "precipitation_hours"]

FEATURE_SETS = {
    "calendar": CALENDAR_FEATURES,
    "calendar+hours": CALENDAR_FEATURES + HOURS_FEATURES,
    "calendar+hours+weather": CALENDAR_FEATURES + HOURS_FEATURES + WEATHER_FEATURES,
}

LGBM_PARAMS = {
    "objective": "regression_l1", "n_estimators": 600, "learning_rate": 0.03, "num_leaves": 31,
    "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "verbose": -1, "random_state": 42,
}


def _hours_between(open_time, close_time):
    o = pd.to_datetime(open_time, format="%H:%M", errors="coerce")
    c = pd.to_datetime(close_time, format="%H:%M", errors="coerce")
    hours = (c - o).dt.total_seconds() / 3600
    return hours.where(hours > 0, hours + 24)  # closes after midnight


def build_frame(crowd, calendar, weather):
    """One row per park and day: features plus `crowd_percent` (NaN where there is no measured score).

    crowd: raw_crowd_calendar rows; calendar: calendar_features; weather: raw_weather_daily.
    Closed days and COVID-closure days are dropped.
    """
    df = crowd.merge(calendar, on="date", how="left").merge(weather, on="date", how="left", suffixes=("", "_weather"))
    df = df[~df["is_closed"].astype(bool) & ~df["event_covid_closure"].astype(bool)].copy()
    dates = pd.to_datetime(df["date"])
    df["year"] = dates.dt.year
    df["open_hours"] = _hours_between(df["open_time"], df["close_time"])
    df["has_extra_hours"] = df["extra_hours_open"].notna()
    for c in ("days_to_next_holiday", "days_since_last_holiday"):
        df[c] = df[c].clip(upper=30)
    df["park_id"] = df["park_id"].astype("category")
    measured = ~df["is_forecast"].astype(bool) & df["crowd_percent"].notna()
    df.loc[~measured, "crowd_percent"] = np.nan
    return df.sort_values(["park_id", "date"]).reset_index(drop=True)


def _matrix(df, features):
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == bool or X[c].dtype == object:
            X[c] = X[c].astype(float)
    return X


def fit(train, features, params=None):
    import lightgbm as lgb

    rows = train.dropna(subset=["crowd_percent"])
    model = lgb.LGBMRegressor(**{**LGBM_PARAMS, **(params or {})})
    model.fit(_matrix(rows, features), rows["crowd_percent"], categorical_feature=["park_id"])
    return model


def predict(model, df, features):
    return np.clip(model.predict(_matrix(df, features)), 0, 100)


def seasonal_naive(history, target):
    """Same park, same weekday, 52 weeks earlier."""
    prior = history[["park_id", "date", "crowd_percent"]].copy()
    prior["date"] = pd.to_datetime(prior["date"]) + pd.Timedelta(days=364)
    out = target[["park_id", "date"]].assign(date=pd.to_datetime(target["date"]))
    return out.merge(prior, on=["park_id", "date"], how="left")["crowd_percent"].to_numpy()


def profile_mean(history, target):
    """Average score for the park, month and weekday over the training history."""
    keys = ["park_id", "month", "day_of_week"]
    means = history.dropna(subset=["crowd_percent"]).groupby(keys, observed=True)["crowd_percent"].mean().rename("pred")
    return target[keys].merge(means.reset_index(), on=keys, how="left")["pred"].to_numpy()


def score(actual, pred):
    ok = ~np.isnan(pred)
    err = np.abs(actual[ok] - pred[ok])
    return {"mae": float(err.mean()), "within_10": float((err <= 10).mean()), "n": int(ok.sum())}


def backtest(df, test_years, feature_sets=FEATURE_SETS):
    """Out-of-time backtest: for each test year, train on every earlier day and score that year.

    Returns one row per (test_year, method) with MAE, share of days within 10 points, and n.
    """
    results = []
    dates = pd.to_datetime(df["date"])
    for year in test_years:
        train = df[dates.dt.year < year]
        test = df[(dates.dt.year == year)].dropna(subset=["crowd_percent"])
        actual = test["crowd_percent"].to_numpy()
        preds = {"baseline: same weekday last year": seasonal_naive(train, test),
                 "baseline: park x month x weekday mean": profile_mean(train, test)}
        for name, features in feature_sets.items():
            preds[f"lightgbm: {name}"] = predict(fit(train, features), test, features)
        for method, pred in preds.items():
            results.append({"test_year": year, "method": method, **score(actual, np.asarray(pred, dtype=float))})
    return pd.DataFrame(results)
