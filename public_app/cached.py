"""Cached data and model runs shared by the pages. Nothing here reaches a network: the data is the
snapshot in data/ and every model is trained in this process."""

import json
import time

import streamlit as st

import lab


@st.cache_data
def daily():
    return lab.load_daily()


@st.cache_data
def monthly():
    return lab.load_monthly()


@st.cache_data
def forecast():
    return lab.load_forecast()


@st.cache_data
def optuna_study():
    return lab.load_study()


@st.cache_data
def meta():
    return json.loads((lab.DATA / "meta.json").read_text(encoding="utf-8"))


@st.cache_data(max_entries=40, show_spinner=False)
def backtest(groups, params):
    """(predictions, importance, seconds) for a feature-group tuple and a sorted params tuple."""
    start = time.perf_counter()
    preds, importance = lab.run_backtest(daily(), lab.features_for(groups), dict(params))
    return preds, importance, time.perf_counter() - start


@st.cache_data
def baselines():
    return lab.baselines(daily())


@st.cache_resource
def champion_model():
    """The champion's settings trained on every measured day, for explaining individual days."""
    features = lab.features_for(lab.CHAMPION_GROUPS)
    return lab.fit(daily(), features), features


@st.cache_data(max_entries=200, show_spinner=False)
def study(unit, donors, pre_start, method, fake_opening):
    return lab.run_study(monthly(), unit, list(donors), pre_start, method, fake_opening)
