"""Monitoring the live forecast: accuracy against a production-like replay, drift by month, and the rules
that start a retune.

**Expected error comes from a walk-forward replay, not the yearly backtest.** Production refits every Monday
and forecasts up to 90 days ahead, so the replay does the same: each Monday, fit on every earlier day and
forecast the next 90. A yearly backtest (fit once, predict a whole year) is staler than production, so its
errors run larger, and a band built from it would almost never be crossed. The constants below were chosen
by replaying 2019-2026 (ops/replay_triggers.py). The error trigger fired five times: the 2020 reopening, the
spring and late-summer 2021 swings, 16 days after Epic Universe opened (May 2025), and spring 2026. It never
fired in 2022-2024.

**Live accuracy against the band**
- Why: the replay says how accurate the model should be. Only forecasts scored against what happened show
  how accurate it is.
- Plain meaning: the 28-day error is the average miss, in crowd-score points, of the forecasts for the last
  28 days. The band runs from the 5th to the 95th percentile of the replay's 28-day errors over the
  reference period. Above the band means worse than 95% of comparable stretches.
- Assumptions: the reference period (the two years ending 90 days before) is typical of what's coming.
  A disruption is therefore judged against calmer times, and it widens the band once it's 90 days old.

**Drift by month (PSI, the population stability index)**
- Why: accuracy needs measured days, which arrive slowly for forecasts 90 days out. Drift shows whether this
  month looks different from the past.
- Plain meaning: cut the reference into ten equal-count bins (deciles). PSI is 0 when this month falls into
  them in the same proportions, and grows as more of it lands in different bins. Two lines:
  - Crowd level vs the same calendar month in the three years before: did crowds change? (Seasonality is
    held fixed, so December isn't "drift" for being busy.)
  - The model's errors vs the replay's errors: did the model's misses change? Only this line can start a
    retune. A crowd shift the model already explains isn't a reason to retune.
- Assumptions: the familiar 0.1 / 0.2 lines assume large samples. A month is only about 100 park-days, and
  past months differ from one another anyway, so even an unremarkable month scores well above 0.2. The
  threshold is calibrated instead. For each past month in the reference, resample its weeks many times,
  score each draw against the whole reference, and note the 99th percentile. The threshold is the median of
  those, so one extreme month (a closure, a park opening) can't raise the bar for everything after it.
  Weeks are resampled whole, all parks together, because neighbouring days aren't independent, and weeks
  are assumed roughly independent of one another. In the 2019-2026 replay the thresholds came out between 1
  and 1.3, five or more times the textbook 0.2. Model-error drift was flagged in June 2020 and August 2021;
  crowd level in 13 of 74 months, including the 2020 reopening, Hurricane Ian and the Epic opening.

**Retune rules (evaluate)**
- A retune starts when any of three triggers fires:
  - The 28-day error (forecasts 0-14 days ahead) stays above the band for 14 days in a row.
  - The model-error PSI is above its threshold for a month.
  - 90 days have passed since the last tuning run (the backstop).
- Only evidence gathered after the last tuning run counts.
- At most one run starts per 14 days (Free Edition compute).
- If an automatic retune for a problem found no reliable gain and a problem fires again, settings probably
  can't fix it (a new park, a closure, a missing feature). The monitor then stops retuning and asks for a
  person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from parks_lib.forecast import fit, predict

HORIZONS = {"0-14 days": (0, 14), "15-90 days": (15, 90)}
TRIGGER_HORIZON = "0-14 days"
WINDOW_DAYS = 28          # rolling error window
MIN_WINDOW_DAYS = 14      # a window needs this many measured days to count
BAND_QUANTILES = (0.05, 0.95)
REFERENCE_DAYS = 730      # the reference period: two years...
REFERENCE_GAP_DAYS = 90   # ...ending this long before the period being judged
MIN_REFERENCE_DAYS = 180  # fewer measured days of reference than this and there is no band or threshold
SUSTAIN_DAYS = 14         # days in a row above the band before the error trigger fires
PSI_BINS = 10
PSI_ALPHA = 0.01          # each reference month's threshold is its 99th percentile
PSI_DRAWS = 500           # per reference month
CROWD_REFERENCE_YEARS = 3
MIN_MONTH_DAYS = 14       # a month needs this many measured days to get a PSI
BACKSTOP_DAYS = 90
COOLDOWN_DAYS = 14
REPLAY_STEP_DAYS = 7
REPLAY_HORIZON_DAYS = 90
PROBLEM_TRIGGERS = ("error", "drift")


def as_day(value):
    """A date or timestamp, timezone-aware or not, as a naive midnight Timestamp."""
    t = pd.Timestamp(value)
    return (t.tz_convert(None) if t.tzinfo else t).normalize()


def bucket_of(horizon: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=horizon.index, dtype="object")
    for name, (lo, hi) in HORIZONS.items():
        out[horizon.between(lo, hi)] = name
    return out


def walk_forward(df, features, params=None, start=None, end=None, step_days=REPLAY_STEP_DAYS,
                 horizon_days=REPLAY_HORIZON_DAYS):
    """Production-like replay: every `step_days` from `start`, fit on every earlier day and forecast the next
    `horizon_days`. `df` is build_frame's output. Returns park_id, date, trained_through, horizon, bucket,
    actual, pred: one row per forecast of a measured day."""
    dates = pd.to_datetime(df["date"])
    start = as_day(start) if start is not None else dates.min() + pd.Timedelta(days=365)
    start -= pd.Timedelta(days=start.weekday())   # production refits on Mondays
    end = as_day(end) if end is not None else dates.max()
    rows = []
    for cutoff in pd.date_range(start, end, freq=f"{step_days}D"):
        ahead = (dates >= cutoff) & (dates <= cutoff + pd.Timedelta(days=horizon_days))
        test = df[ahead].dropna(subset=["crowd_percent"])
        train = df[dates < cutoff]
        if test.empty or train["crowd_percent"].notna().sum() == 0:
            continue
        model = fit(train, features, params)
        rows.append(test[["park_id", "date"]].assign(trained_through=cutoff, actual=test["crowd_percent"].to_numpy(),
                                                     pred=predict(model, test, features)))
    if not rows:
        return pd.DataFrame(columns=["park_id", "date", "trained_through", "horizon", "bucket", "actual", "pred"])
    out = pd.concat(rows, ignore_index=True)
    out["park_id"] = out["park_id"].astype(int)
    out["date"] = pd.to_datetime(out["date"])
    out["horizon"] = (out["date"] - out["trained_through"]).dt.days
    out["bucket"] = bucket_of(out["horizon"])
    return out


def live_errors(history, measured):
    """Every scored forecast of a day that has since been measured.

    history: crowd_forecast_history (park_id, date, predicted_crowd, queue_times_forecast, scored_at).
    measured: park_id, date, crowd_percent (measured days only). A day scored twice keeps its last run.
    Returns park_id, date, scored_on, horizon, bucket, actual, pred, qt_pred.
    """
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"])
    scored = pd.to_datetime(h["scored_at"], utc=True)
    h["scored_on"] = scored.dt.tz_localize(None).dt.normalize()
    h["_at"] = scored
    h = h.sort_values("_at").drop_duplicates(["scored_on", "park_id", "date"], keep="last")
    m = measured[["park_id", "date", "crowd_percent"]].assign(date=lambda x: pd.to_datetime(x["date"]))
    out = h.merge(m.dropna(subset=["crowd_percent"]), on=["park_id", "date"])
    out["horizon"] = (out["date"] - out["scored_on"]).dt.days
    out["bucket"] = bucket_of(out["horizon"])
    out = out.dropna(subset=["bucket"])
    return out.rename(columns={"crowd_percent": "actual", "predicted_crowd": "pred", "queue_times_forecast": "qt_pred"})[
        ["park_id", "date", "scored_on", "horizon", "bucket", "actual", "pred", "qt_pred"]].reset_index(drop=True)


def rolling_mae(errors, pred="pred", window=WINDOW_DAYS, min_days=MIN_WINDOW_DAYS):
    """Mean absolute error of every forecast for the `window` days ending on each date (NaN where fewer than
    `min_days` of them were measured). Indexed by every calendar date in the span."""
    rows = errors.dropna(subset=[pred])
    if rows.empty:
        return pd.Series(dtype=float)
    e = (rows["actual"] - rows[pred]).abs()
    daily = pd.DataFrame({"s": e.to_numpy(), "n": 1}, index=pd.to_datetime(rows["date"])).groupby(level=0).sum()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
    roll = daily.rolling(window, min_periods=1).sum()
    days = (daily["n"] > 0).astype(int).rolling(window, min_periods=1).sum()
    return (roll["s"] / roll["n"]).where(days >= min_days)


def reference_window(as_of):
    as_of = pd.Timestamp(as_of)
    return as_of - pd.Timedelta(days=REFERENCE_DAYS + REFERENCE_GAP_DAYS), as_of - pd.Timedelta(days=REFERENCE_GAP_DAYS)


def band(replay_rolling, as_of):
    """(low, high): the BAND_QUANTILES of the replay's 28-day errors over the reference period before `as_of`."""
    if replay_rolling.empty:
        return np.nan, np.nan
    lo, hi = reference_window(as_of)
    ref = replay_rolling[(replay_rolling.index >= lo) & (replay_rolling.index < hi)].dropna()
    if len(ref) < MIN_REFERENCE_DAYS:
        return np.nan, np.nan
    q = ref.quantile(list(BAND_QUANTILES)).to_numpy()
    return float(q[0]), float(q[1])


def accuracy_table(live, replay):
    """Per horizon bucket and date: the live 28-day error, queue-times' over the same window, how many
    forecasts were scored that day, and the band as of the first of that month."""
    frames = []
    for name in HORIZONS:
        lv, rp = live[live["bucket"] == name], replay[replay["bucket"] == name]
        ours = rolling_mae(lv)
        if ours.empty:
            continue
        theirs = (rolling_mae(lv, pred="qt_pred").reindex(ours.index) if "qt_pred" in lv
                  else pd.Series(np.nan, index=ours.index))
        replay_roll = rolling_mae(rp)
        months = {m: band(replay_roll, m) for m in ours.index.to_period("M").unique().to_timestamp()}
        bands = np.array([months[d.to_period("M").to_timestamp()] for d in ours.index])
        n = lv.groupby("date").size().reindex(ours.index, fill_value=0)
        frames.append(pd.DataFrame({"bucket": name, "date": ours.index, "forecasts": n.to_numpy(),
                                    "live_mae": ours.to_numpy(), "queue_times_mae": theirs.to_numpy(),
                                    "band_low": bands[:, 0], "band_high": bands[:, 1]}))
    if not frames:
        return pd.DataFrame(columns=["bucket", "date", "forecasts", "live_mae", "queue_times_mae", "band_low", "band_high"])
    return pd.concat(frames, ignore_index=True)


def psi(reference, current, bins=PSI_BINS):
    """Population stability index of `current` against `reference`, on the reference's deciles."""
    edges = _edges(np.asarray(reference, dtype=float), bins)
    return float(_psi_counts(_counts(reference, edges), _counts(current, edges)))


def _edges(reference, bins):
    return np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))


def _counts(values, edges):
    return np.bincount(np.searchsorted(edges, np.asarray(values, dtype=float), side="right"), minlength=len(edges) + 1)


def _psi_counts(ref_counts, cur_counts):
    """PSI from bin counts; half a count is added to every bin so an empty one doesn't make it infinite."""
    ref = (ref_counts + 0.5) / (ref_counts.sum(axis=-1, keepdims=True) + 0.5 * ref_counts.shape[-1])
    cur = (cur_counts + 0.5) / (cur_counts.sum(axis=-1, keepdims=True) + 0.5 * cur_counts.shape[-1])
    return ((cur - ref) * np.log(cur / ref)).sum(axis=-1)


def psi_test(reference, current, reference_weeks, reference_months, current_weeks):
    """PSI with a threshold calibrated on the reference itself.

    For each reference month: resample as many of its weeks as `current` spans, PSI_DRAWS times, scoring each
    draw against the whole reference, and take the 1 - PSI_ALPHA quantile. The threshold is the median of
    those per-month quantiles: how unusual a typical past month gets by chance, which one extreme month
    (a closure, a park opening) can't drag up. Returns psi and threshold. Needs at least two reference months.
    """
    reference = np.asarray(reference, dtype=float)
    edges = _edges(reference, PSI_BINS)
    ref_counts = _counts(reference, edges)
    cells = pd.DataFrame({"month": np.asarray(reference_months), "week": np.asarray(reference_weeks),
                          "bin": np.searchsorted(edges, reference, side="right")})
    per_week = (cells.groupby(["month", "week", "bin"]).size().unstack(fill_value=0)
                .reindex(columns=range(len(edges) + 1), fill_value=0))
    take = max(1, len(np.unique(current_weeks)))
    rng = np.random.default_rng(0)
    quantiles = []
    for month in per_week.index.get_level_values("month").unique():
        weeks = per_week.loc[month].to_numpy()
        draws = weeks[rng.integers(0, len(weeks), size=(PSI_DRAWS, take))].sum(axis=1)
        quantiles.append(np.quantile(_psi_counts(ref_counts[None, :], draws), 1 - PSI_ALPHA))
    return {"psi": float(_psi_counts(ref_counts, _counts(current, edges))), "threshold": float(np.median(quantiles))}


def _week(dates):
    return (pd.to_datetime(pd.Series(dates)) - pd.Timestamp("2000-01-03")).dt.days.to_numpy() // 7


def crowd_drift(measured, months=None):
    """Crowd level each month against the same calendar month in the CROWD_REFERENCE_YEARS before, for the
    parks that have that history."""
    m = measured.dropna(subset=["crowd_percent"]).assign(date=lambda x: pd.to_datetime(x["date"]))
    m["month"] = m["date"].dt.to_period("M")
    rows = []
    for month in months if months is not None else sorted(m["month"].unique()):
        cur = m[m["month"] == month]
        ref = m[(m["date"].dt.month == month.month) & m["date"].dt.year.between(month.year - CROWD_REFERENCE_YEARS,
                                                                                 month.year - 1)]
        parks = set(cur["park_id"]) & set(ref["park_id"])
        cur, ref = cur[cur["park_id"].isin(parks)], ref[ref["park_id"].isin(parks)]
        if cur["date"].nunique() < MIN_MONTH_DAYS or ref["month"].nunique() < 2:
            continue
        t = psi_test(ref["crowd_percent"], cur["crowd_percent"], _week(ref["date"]), ref["month"], _week(cur["date"]))
        rows.append({"month": month.to_timestamp(), "line": "crowd level", "source": "measured", "n": len(cur), **t})
    return pd.DataFrame(rows)


def _one_per_day(errors):
    """The nearest-horizon forecast of each park-day in the trigger bucket (signed error: actual - pred)."""
    e = errors[errors["bucket"] == TRIGGER_HORIZON].sort_values("horizon").drop_duplicates(["park_id", "date"])
    return e.assign(error=e["actual"] - e["pred"])


def error_drift(replay, live=None):
    """The model's errors each month against the replay's errors over the reference period. Months with
    enough live forecasts use them; earlier months use the replay's own."""
    rp = _one_per_day(replay)
    lv = _one_per_day(live) if live is not None and len(live) else rp.iloc[0:0]
    live_days = lv.groupby(lv["date"].dt.to_period("M"))["date"].nunique()
    live_months = set(live_days[live_days >= MIN_MONTH_DAYS].index)
    months = sorted(set(rp["date"].dt.to_period("M")) | live_months)
    rows = []
    for month in months:
        source, cur = ("live", lv) if month in live_months else ("replay", rp)
        cur = cur[cur["date"].dt.to_period("M") == month]
        lo, hi = reference_window(month.to_timestamp())
        ref = rp[(rp["date"] >= lo) & (rp["date"] < hi)]
        if cur["date"].nunique() < MIN_MONTH_DAYS or ref["date"].nunique() < MIN_REFERENCE_DAYS:
            continue
        t = psi_test(ref["error"], cur["error"], _week(ref["date"]), ref["date"].dt.to_period("M"), _week(cur["date"]))
        rows.append({"month": month.to_timestamp(), "line": "model errors", "source": source, "n": len(cur), **t})
    return pd.DataFrame(rows)


@dataclass
class Trigger:
    name: str      # "error", "drift" or "backstop"
    fired: bool
    detail: str


def _trigger_rows(accuracy, since=None):
    rows = accuracy[accuracy["bucket"] == TRIGGER_HORIZON].dropna(subset=["live_mae", "band_high"]).sort_values("date")
    return rows[rows["date"] > as_day(since)] if since is not None else rows


def _runs_above(rows):
    """For each row, how many days in a row (calendar days, no gaps) the 28-day error has been above the band."""
    above = (rows["live_mae"] > rows["band_high"]).to_numpy()
    consecutive = np.r_[True, np.diff(rows["date"].to_numpy()).astype("timedelta64[D]").astype(int) == 1]
    runs, run = [], 0
    for a, c in zip(above, consecutive):
        run = run + 1 if a and c else int(a)
        runs.append(run)
    return np.array(runs, dtype=int)


def fire_dates(accuracy):
    """Every date the error trigger would have fired over a history: each time a run above the band reaches
    SUSTAIN_DAYS. For replaying the rule on the past (ops/replay_triggers.py)."""
    rows = _trigger_rows(accuracy)
    return list(rows["date"][_runs_above(rows) == SUSTAIN_DAYS]) if len(rows) else []


def error_trigger(accuracy, since=None):
    rows = _trigger_rows(accuracy, since)
    if rows.empty:
        return Trigger("error", False, "not enough measured forecasts since the last tuning run")
    run = int(_runs_above(rows)[-1])
    last = rows.iloc[-1]
    detail = f"28-day error {last['live_mae']:.1f} against a band up to {last['band_high']:.1f}"
    if run:
        detail += f", above it {run} day{'s' if run > 1 else ''} in a row"
    return Trigger("error", run >= SUSTAIN_DAYS, detail)


def drift_trigger(drift, since=None):
    if drift is None or drift.empty:
        return Trigger("drift", False, "no month of live forecasts since the last tuning run")
    rows = drift[(drift["line"] == "model errors") & (drift["source"] == "live")].sort_values("month")
    if since is not None:
        rows = rows[rows["month"] > as_day(since)]
    if rows.empty:
        return Trigger("drift", False, "no month of live forecasts since the last tuning run")
    last = rows.iloc[-1]
    fired = bool(last["psi"] > last["threshold"])
    return Trigger("drift", fired, f"{last['month']:%B %Y} model-error PSI {last['psi']:.2f} against a threshold of "
                                   f"{last['threshold']:.2f}")


def backstop_trigger(last_tuned, today):
    if last_tuned is None:
        return Trigger("backstop", True, "no tuning run on record")
    days = (as_day(today) - as_day(last_tuned)).days
    return Trigger("backstop", days >= BACKSTOP_DAYS, f"{days} of {BACKSTOP_DAYS} days since the last tuning run")


def decide(triggers, today, last_started=None, last_run=None):
    """What to do about the triggers: (action, reason). Actions: none, cooldown, needs a person, start.

    last_started: when a tuning run last started (finished or not). last_run: the latest finished tuning run
    as a dict with `trigger` (what started it) and `promoted`.
    """
    fired = [t for t in triggers if t.fired]
    if not fired:
        return "none", "no trigger fired"
    reason = "; ".join(f"{t.name}: {t.detail}" for t in fired)
    if last_started is not None:
        days = (as_day(today) - as_day(last_started)).days
        if days < COOLDOWN_DAYS:
            return "cooldown", f"{reason}. A tuning run started {days} days ago; the next can start in {COOLDOWN_DAYS - days}"
    problems = [t for t in fired if t.name in PROBLEM_TRIGGERS]
    if problems and last_run is not None and last_run.get("trigger") in PROBLEM_TRIGGERS and not last_run.get("promoted"):
        return "needs a person", (f"{reason}. The last automatic retune (for {last_run['trigger']}) found no reliable "
                                  "gain, so settings alone probably can't fix this: look for a missing feature or event")
    return "start", reason


def first_trigger(triggers):
    """The trigger name a started tuning run records: a problem before the backstop."""
    names = [t.name for t in triggers if t.fired]
    return next((n for n in PROBLEM_TRIGGERS if n in names), names[0] if names else "manual")
