"""Altair charts for the public app, in the Databricks app's validated dark palette (databricks/app/charts.py).

A series keeps its slot on every chart: the champion model wears the forecast slot (teal), your
model the second slot (amber; the brand's brighter highlight amber is too light for a series on
navy), measured scores the third (violet). Baselines are grey context lines, labelled directly.
"""

import altair as alt
import pandas as pd

import charts
from charts import CONTEXT, HIGHLIGHT, RAMP, SLOTS, SUBTEXT, SURFACE, TEXT

CHAMPION, YOURS, MEASURED = "Champion", "Your model", "Measured"
charts.SERIES_COLORS.update({CHAMPION: SLOTS[0], YOURS: SLOTS[1], MEASURED: SLOTS[2]})
BASELINE_COLORS = {"Same weekday last year": CONTEXT, "Park x month x weekday average": SUBTEXT}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def mae_by_year(frame):
    """Backtest MAE per test year: your model and the champion in their slots, baselines in grey.

    `frame`: test_year, method, mae. Lower is better, so the y axis starts at zero.
    """
    order = [YOURS, CHAMPION, *BASELINE_COLORS]
    colors = {YOURS: SLOTS[1], CHAMPION: SLOTS[0], **BASELINE_COLORS}
    present = [m for m in order if m in set(frame["method"])]
    # The two grey baselines differ by marker shape too, so the legend tells them apart
    shapes = {YOURS: "circle", CHAMPION: "circle", "Same weekday last year": "square",
              "Park x month x weekday average": "triangle-up"}
    legend = alt.Legend(orient="top", labelLimit=320)
    color = alt.Color("method:N", title=None, sort=present, legend=legend,
                      scale=alt.Scale(domain=present, range=[colors[m] for m in present]))
    shape = alt.Shape("method:N", title=None, sort=present, legend=legend,
                      scale=alt.Scale(domain=present, range=[shapes[m] for m in present]))
    x = alt.X("test_year:O", title="Test year", axis=alt.Axis(labelAngle=0))
    y = alt.Y("mae:Q", title="Mean absolute error, crowd points", scale=alt.Scale(domain=[0, 30]))
    tooltip = [alt.Tooltip("method:N", title="Method"), alt.Tooltip("test_year:O", title="Test year"),
               alt.Tooltip("mae:Q", title="MAE", format=".1f")]
    base = alt.Chart(frame).encode(x=x, y=y, color=color, tooltip=tooltip)
    is_model = alt.FieldOneOfPredicate(field="method", oneOf=[YOURS, CHAMPION])
    line = base.mark_line(strokeCap="round").encode(
        strokeWidth=alt.when(is_model).then(alt.value(2.5)).otherwise(alt.value(1.5)))
    points = base.mark_point(filled=True, size=90, stroke=SURFACE, strokeWidth=2).encode(shape=shape)
    last = frame[frame["test_year"] == frame["test_year"].max()]
    labels = (alt.Chart(last[last["method"] == YOURS]).mark_text(align="left", dx=10, color=TEXT, fontSize=12)
              .encode(x=x, y=y, text=alt.Text("mae:Q", format=".1f")))
    return alt.layer(line, points, labels).properties(height=320)


def importance(series, top=12, champion=False):
    """Share of the model's total gain per feature, the top `top` features, in the model's slot colour."""
    frame = series.head(top).rename("share").rename_axis("feature").reset_index()
    # labelOverlap off: Vega otherwise drops every other name when rows are tight
    y = alt.Y("feature:N", title=None, sort=frame["feature"].tolist(),
              axis=alt.Axis(ticks=False, domain=False, labelOverlap=False, labelLimit=260))
    x = alt.X("share:Q", title="Share of the model's gain", axis=alt.Axis(format="%"))
    base = alt.Chart(frame).encode(x=x, y=y, tooltip=[alt.Tooltip("feature:N", title="Feature"),
                                                       alt.Tooltip("share:Q", title="Share", format=".1%")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=16, color=SLOTS[0] if champion else SLOTS[1])
    text = base.mark_text(align="left", dx=6, color=SUBTEXT, fontSize=12).encode(text=alt.Text("share:Q", format=".0%"))
    return alt.layer(bars, text).properties(height=30 * len(frame))


def prediction_lines(frame):
    """Measured score against the champion's and your model's backtest predictions for one park-year,
    as weekly averages (a year of daily values for three series is unreadable)."""
    weekly = (frame.set_index("date").groupby("series")["value"].resample("W-MON", label="left", closed="left")
              .mean().reset_index().dropna())
    present = [s for s in (MEASURED, CHAMPION, YOURS) if s in set(weekly["series"])]
    return charts.lines(weekly, "date:T", "Crowd score, weekly average", present, y_domain=[0, 100], height=300)


def year_calendar(days, selected=None):
    """A park's year as a calendar: weeks across, weekdays down, colour = crowd score. Click a day.

    `days`: date, crowd_percent, holiday_name, day_key (YYYY-MM-DD, the selection field).
    """
    days = days.assign(week=days["date"].dt.isocalendar().week.astype(int),
                       weekday=days["date"].dt.day_of_week.map(dict(enumerate(WEEKDAYS))),
                       month=days["date"].dt.strftime("%b"))
    # Week columns run in calendar order: ISO week 1 can start in late December and week 52/53 in January
    days["col"] = (days["date"] - pd.to_timedelta(days["date"].dt.day_of_week, unit="D")).dt.strftime("%Y-%m-%d")
    pick = alt.selection_point(name="day", fields=["day_key"], on="click", clear="dblclick", toggle=False)
    x = alt.X("col:O", title=None, sort="ascending",
              axis=alt.Axis(labels=False, ticks=False, domain=False))
    y = alt.Y("weekday:O", title=None, sort=WEEKDAYS, axis=alt.Axis(ticks=False, domain=False))
    tooltip = [alt.Tooltip("date:T", title="Date", format="%a %b %-d, %Y"),
               alt.Tooltip("crowd_percent:Q", title="Crowd score", format=".0f"),
               alt.Tooltip("holiday_name:N", title="Holiday")]
    cells = (alt.Chart(days).mark_rect(cornerRadius=3, stroke=SURFACE, strokeWidth=2)
             .encode(x=x, y=y, tooltip=tooltip,
                     color=alt.Color("crowd_percent:Q", title="Crowd score", scale=alt.Scale(domain=[0, 100], range=RAMP),
                                     legend=alt.Legend(orient="right", gradientLength=150)),
                     opacity=alt.when(pick).then(alt.value(1)).otherwise(alt.value(0.9)))
             .add_params(pick))
    months = days.groupby("month", sort=False)["col"].min().reset_index()
    month_labels = (alt.Chart(months).mark_text(align="left", baseline="bottom", dy=-4, color=SUBTEXT, fontSize=12)
                    .encode(x=alt.X("col:O", sort="ascending"), y=alt.value(0), text="month:N"))
    layers = [cells, month_labels]
    if selected is not None:
        ring = days[days["day_key"] == selected]
        layers.append(alt.Chart(ring).mark_rect(cornerRadius=3, fillOpacity=0, stroke=HIGHLIGHT, strokeWidth=2.5)
                      .encode(x=x, y=y))
    return alt.layer(*layers).properties(height=7 * 26)


def contributions(frame):
    """How each group of features moved one day's prediction from the model's average day.

    `frame`: group, points. Bars grow from zero, one colour; the sign is the direction, labelled.
    """
    y = alt.Y("group:N", title=None, sort=frame["group"].tolist(), axis=alt.Axis(ticks=False, domain=False, labelLimit=220, labelOverlap=False))
    x = alt.X("points:Q", title="Push on the prediction, crowd points")
    base = alt.Chart(frame).encode(x=x, y=y, tooltip=[alt.Tooltip("group:N", title="Features"),
                                                       alt.Tooltip("points:Q", title="Points", format="+.1f")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=18, color=SLOTS[0])
    pos = base.transform_filter(alt.datum.points >= 0).mark_text(align="left", dx=6, color=TEXT, fontSize=12) \
        .encode(text=alt.Text("points:Q", format="+.1f"))
    # Negative labels sit just right of zero: left of the bar they would run into the group names
    neg = base.transform_filter(alt.datum.points < 0).mark_text(align="left", dx=6, color=TEXT, fontSize=12) \
        .encode(x=alt.datum(0), text=alt.Text("points:Q", format="+.1f"))
    zero = alt.Chart(pd.DataFrame({"points": [0]})).mark_rule(color=SUBTEXT, strokeWidth=1).encode(x="points:Q")
    return alt.layer(bars, zero, pos, neg).properties(height=34 * len(frame))


def weights(frame):
    """Synthetic-control weights per comparison park, one colour."""
    y = alt.Y("park:N", title=None, sort=frame["park"].tolist(), axis=alt.Axis(ticks=False, domain=False, labelLimit=260, labelOverlap=False))
    x = alt.X("weight:Q", title="Weight in the no-Epic estimate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]))
    base = alt.Chart(frame).encode(x=x, y=y, tooltip=[alt.Tooltip("park:N", title="Park"),
                                                       alt.Tooltip("weight:Q", title="Weight", format=".0%")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=16, color=SLOTS[0])
    text = base.mark_text(align="left", dx=6, color=SUBTEXT, fontSize=12).encode(text=alt.Text("weight:Q", format=".0%"))
    return alt.layer(bars, text).properties(height=28 * len(frame))


OPTUNA = SLOTS[2]   # the palette's violet: Optuna's best and the suggested bands on the lab's sliders
BAND_FILL = "rgba(124, 128, 234, 0.38)"   # OPTUNA, faint enough that the track still reads on top of it
MIN_BAND = 2.0       # percent of the track, so a tight band (5-6 on a 0-12 slider) stays visible


def slider_guides(sliders, best, bands):
    """CSS that draws Optuna's best (a violet tick) and a suggested band (a faint violet bar behind the track)
    on each slider. No native widget does this, so it targets the sliders' `st-key-<key>` classes.

    `sliders`: widget key -> (min, max). `best`, `bands`: widget key -> value, (low, high).
    The track fills the width of the slider's positioned inner box and the thumb's centre travels from 0%
    to 100% of it, so a value's position is its share of the range.
    """
    def pct(key, value):
        lo, hi = sliders[key]
        return 100 * min(max((value - lo) / (hi - lo), 0), 1)

    rules = []
    for key in sliders:
        box = f'.st-key-{key} [data-testid="stSlider"] [role="group"] > div'
        if key in bands:
            left, right = pct(key, bands[key][0]), pct(key, bands[key][1])
            if right - left < MIN_BAND:
                mid = (left + right) / 2
                left, right = max(mid - MIN_BAND / 2, 0), min(mid + MIN_BAND / 2, 100)
            rules.append(f'{box}::before {{content: ""; position: absolute; top: 50%; height: 12px; '
                         f"transform: translateY(-50%); left: {left:.2f}%; width: {right - left:.2f}%; "
                         f"background: {BAND_FILL}; border-radius: 6px; pointer-events: none;}}")
        if key in best:
            rules.append(f'{box}::after {{content: ""; position: absolute; top: 50%; height: 18px; width: 3px; '
                         f"transform: translate(-50%, -50%); left: {pct(key, best[key]):.2f}%; background: {OPTUNA}; "
                         f"border-radius: 2px; box-shadow: 0 0 0 1.5px {SURFACE}; z-index: 1; pointer-events: none;}}")
            rules.append(f"{box} > div:has(input) {{z-index: 2;}}")   # the thumb stays above the tick
    return "<style>\n" + "\n".join(rules) + "\n</style>"


def slider_legend(band_trials, total_trials):
    """One line naming the tick and the band, with swatches drawn like the marks themselves."""
    tick = (f'<span style="display:inline-block;width:3px;height:16px;background:{OPTUNA};border-radius:2px;'
            'vertical-align:middle;margin-right:8px"></span>')
    band = (f'<span style="display:inline-block;width:26px;height:12px;background:{BAND_FILL};border-radius:6px;'
            'vertical-align:middle;margin-right:8px"></span>')
    return (f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem 1.5rem;font-size:0.875rem;color:{SUBTEXT}">'
            f"<span>{tick}Optuna's best, what <b>Load Optuna's best</b> loads</span>"
            f"<span>{band}Suggested: where the {band_trials} best of {total_trials} trials landed</span></div>")


impact = charts.impact
placebo_bars = charts.placebo_bars
trend = charts.trend
