"""Altair charts in the brand's dark palette (OneDrive Branding/brand-guide.md).

The colours passed the dataviz palette validator against the navy surface: the three categorical
slots clear the all-pairs colour-vision and contrast checks, and the teal ramp is a single-hue,
monotone scale. A series keeps its slot on every chart, and amber marks only today.
"""

import altair as alt
import pandas as pd

from transforms import IMPACT_SERIES, TREND_SERIES, WEEKDAYS, hour_label

SURFACE = "#0e1a2b"
TEXT = "#f2f5f9"
SUBTEXT = "#9fb1c7"
HIGHLIGHT = "#f0a742"
SLOTS = ["#26a59f", "#c8820f", "#7c80ea"]
SERIES_COLORS = {  # entity -> slot, the same on every chart
    "Forecast": SLOTS[0], "Queue-times forecast": SLOTS[1], "Measured": SLOTS[2],
    "Published plan": SLOTS[0], "Published plan, scenario crowd": SLOTS[1], "Re-planned": SLOTS[2],
    # Impact study: the observed score wears the measured slot, the model's estimate the forecast slot
    "Actual": SLOTS[2], "Without Epic (estimated)": SLOTS[0],
}
CONTEXT = "#6f86a3"  # the theme's grey, for marks that are background to the one in a slot
RAMP = ["#075957", "#007370", "#248b88", "#43a4a0", "#5ebdb9", "#79d7d3"]
DARK_TEXT_FROM = 0.45  # share of the ramp's domain past which navy text out-contrasts light text
HOUR_LABELS = "(datum.value % 12 || 12) + (datum.value % 24 < 12 ? 'am' : 'pm')"  # Vega twin of hour_label
DATE_FORMAT = "%a %b %-d"


def lines(frame, x, y_title, series, *, hourly=False, y_domain=None, height=300, date_format=DATE_FORMAT):
    """Lines per series with a hover rule and a tooltip listing every series at that x.

    `frame` is long: the x column, `series` and `value`. `hourly` x values are hours of the day.
    """
    field = x.split(":")[0]
    color = alt.Color("series:N", title=None, scale=alt.Scale(domain=series, range=[SERIES_COLORS[s] for s in series]),
                      legend=alt.Legend(orient="top", symbolType="stroke", symbolStrokeWidth=2, labelLimit=320))
    y = alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=y_domain) if y_domain else alt.Undefined)
    x_enc = alt.X(x, title=None, axis=alt.Axis(labelExpr=HOUR_LABELS, tickMinStep=1) if hourly else alt.Undefined)
    hover = alt.selection_point(fields=[field], nearest=True, on="pointerover", clear="pointerout", empty=False)

    wide = frame.pivot_table(index=field, columns="series", values="value").reset_index()
    wide.columns.name = None
    if hourly:
        wide["label"] = wide[field].map(hour_label)
        tooltip = [alt.Tooltip("label:N", title="Hour")]
    else:
        tooltip = [alt.Tooltip(x, title="Date", format=date_format)]
    tooltip += [alt.Tooltip(field=s, type="quantitative", format=",.0f") for s in series if s in wide]
    rule = (alt.Chart(wide).mark_rule(color=SUBTEXT, strokeWidth=1)
            .encode(x=x_enc, opacity=alt.when(hover).then(alt.value(0.5)).otherwise(alt.value(0)), tooltip=tooltip)
            .add_params(hover))

    base = alt.Chart(frame).encode(x=x_enc, y=y, color=color)
    return alt.layer(
        base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round"),
        base.mark_point(filled=True, size=64, stroke=SURFACE, strokeWidth=2).encode(
            opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0))),
        rule,
    ).properties(height=height)


def trend(frame, today):
    """Measured crowd score, our forecast and queue-times' forecast on one 0-100 axis, today in amber."""
    present = set(frame["series"])
    chart = lines(frame.rename(columns={"crowd": "value"}), "date:T", "Crowd score",
                  [s for s in TREND_SERIES if s in present],
                  y_domain=[0, 100])  # no end labels: they land on the lines here; legend + tooltip carry identity
    marker = pd.DataFrame({"date": [pd.Timestamp(today)], "label": ["Today"]})
    today_rule = alt.Chart(marker).mark_rule(color=HIGHLIGHT, strokeWidth=1.5).encode(x="date:T")
    today_text = (alt.Chart(marker).mark_text(align="left", baseline="top", dx=4, y=2, color=SUBTEXT, fontSize=12)
                  .encode(x="date:T", text="label:N"))
    return alt.layer(chart, today_rule, today_text)


def calendar(grid, today):
    """Forecast crowd per day as a calendar: weeks across, weekdays down, today ringed in amber."""
    x = alt.X("week_label:O", title=None, sort=alt.EncodingSortField("week", op="min"),
              axis=alt.Axis(orient="top", labelAngle=0, ticks=False, domain=False))
    y = alt.Y("weekday:O", title=None, sort=WEEKDAYS, axis=alt.Axis(ticks=False, domain=False))
    tooltip = [alt.Tooltip("date:T", title="Date", format=DATE_FORMAT),
               alt.Tooltip("predicted_crowd:Q", title="Forecast", format=".0f"),
               alt.Tooltip("queue_times_forecast:Q", title="Queue-times forecast", format=".0f"),
               alt.Tooltip("holiday_name:N", title="Holiday")]
    base = alt.Chart(grid).encode(x=x, y=y, tooltip=tooltip)
    cells = base.mark_rect(cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color("predicted_crowd:Q", title="Crowd score", scale=alt.Scale(domain=[0, 100], range=RAMP),
                        legend=alt.Legend(orient="right", gradientLength=180)))
    days = base.mark_text(fontSize=12).encode(
        text="day:Q",
        color=alt.when(alt.datum.predicted_crowd >= 100 * DARK_TEXT_FROM).then(alt.value(SURFACE))
        .otherwise(alt.value(TEXT)))
    ring = (alt.Chart(grid[grid["date"] == pd.Timestamp(today)]).encode(x=x, y=y)
            .mark_rect(cornerRadius=4, fillOpacity=0, stroke=HIGHLIGHT, strokeWidth=2.5))
    return alt.layer(cells, days, ring).properties(height=7 * 40)


def location_heatmap(hourly, field, title, order):
    """Locations down, hours across, one sequential measure (crew or wait) per cell."""
    top = max(float(hourly[field].max() or 0), 1.0)
    x = alt.X("hour:O", title=None,
              axis=alt.Axis(orient="top", labelAngle=0, ticks=False, domain=False, labelExpr=HOUR_LABELS))
    y = alt.Y("location:N", title=None, sort=order, axis=alt.Axis(labelLimit=260, ticks=False, domain=False))
    tooltip = [alt.Tooltip("location:N", title="Location"), alt.Tooltip("hour_label:N", title="Hour"),
               alt.Tooltip("crew:Q", title="Crew"), alt.Tooltip("wait_minutes:Q", title="Wait, minutes"),
               alt.Tooltip("arrivals:Q", title="Guests arriving", format=","),
               alt.Tooltip("skipped:Q", title="Guests who skipped", format=",")]
    return (alt.Chart(hourly.assign(hour_label=hourly["hour"].map(hour_label)))
            .mark_rect(cornerRadius=3, stroke=SURFACE, strokeWidth=2)
            .encode(x=x, y=y, tooltip=tooltip,
                    color=alt.Color(f"{field}:Q", title=title, scale=alt.Scale(domain=[0, top], range=RAMP),
                                    legend=alt.Legend(orient="right", gradientLength=140)))
            .properties(height=40 * len(order)))


def impact(frame, opening):
    """Actual monthly crowd score against the no-Epic estimate, with the opening marked in amber."""
    chart = lines(frame, "month:T", "Crowd score, monthly average", IMPACT_SERIES, y_domain=[0, 100],
                  date_format="%b %Y")
    marker = pd.DataFrame({"month": [opening], "label": ["Epic opens"]})
    rule = alt.Chart(marker).mark_rule(color=HIGHLIGHT, strokeWidth=1.5).encode(x="month:T")
    text = (alt.Chart(marker).mark_text(align="left", baseline="top", dx=4, y=2, color=SUBTEXT, fontSize=12)
            .encode(x="month:T", text="label:N"))
    return alt.layer(chart, rule, text)


def placebo_bars(frame):
    """Each park's estimated effect from zero; the studied park in teal, the placebo parks in grey."""
    # An explicit order: a sort by field doesn't survive layering with the label and zero rule
    y = alt.Y("park:N", title=None, sort=frame.sort_values("effect")["park"].tolist(),
              axis=alt.Axis(labelLimit=320, ticks=False, domain=False))
    x = alt.X("effect:Q", title="Change after the opening, crowd score points")
    tooltip = [alt.Tooltip("park:N", title="Park"), alt.Tooltip("effect:Q", title="Change, points", format="+.1f")]
    base = alt.Chart(frame).encode(x=x, y=y, tooltip=tooltip)
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.when(alt.datum.is_treated).then(alt.value(SLOTS[0])).otherwise(alt.value(CONTEXT)))
    label = (alt.Chart(frame[frame["is_treated"]]).encode(x=x, y=y, text=alt.Text("effect:Q", format="+.1f"))
             .mark_text(align="right", dx=-6, color=TEXT, fontSize=12))
    zero = alt.Chart(pd.DataFrame({"effect": [0]})).mark_rule(color=SUBTEXT, strokeWidth=1).encode(x="effect:Q")
    return alt.layer(bars, zero, label).properties(height=30 * len(frame))
