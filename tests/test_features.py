from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.analyst import (
    aligned_actual_eps,
    enrich_historical_consensus,
    historical_eps_features,
    historical_surprise_features,
)
from src.features.fundamentals import point_in_time_fundamentals
from src.features.market import event_returns, market_features


def test_fundamental_ratios_are_calculated(quarterly_fundamentals):
    features, visible = point_in_time_fundamentals(quarterly_fundamentals, pd.Timestamp("2024-03-01", tz="UTC"))
    assert len(visible) == 8
    assert features["gross_margin"] == 0.5
    assert features["operating_margin"] == 0.2
    assert features["current_ratio"] == 2.0
    assert features["revenue_yoy"] > 0


def test_market_features_use_only_prior_prices(prices):
    as_of = pd.Timestamp("2024-06-01", tz="UTC")
    before = market_features(prices, "TEST", as_of)
    modified = prices.copy()
    modified.loc[modified["date"] >= as_of, "adj_close"] *= 100
    after = market_features(modified, "TEST", as_of)
    assert before == after


def test_zero_denominators_do_not_create_infinite_features(quarterly_fundamentals):
    history = quarterly_fundamentals.copy()
    history.loc[history.index[-2], ["revenue", "eps_diluted"]] = 0.0
    history.loc[history.index[-2], "operating_income"] = 10.0

    features, _ = point_in_time_fundamentals(history, pd.Timestamp("2024-03-01", tz="UTC"))
    numeric = np.asarray([value for value in features.values() if isinstance(value, (int, float))])

    assert not np.isinf(numeric).any()


def test_daily_price_visibility_respects_earnings_timing() -> None:
    dates = pd.date_range("2024-01-08", periods=7, freq="B", tz="UTC")
    prices = pd.concat([
        pd.DataFrame({"ticker": ticker, "date": dates, "adj_close": values, "volume": 100.0})
        for ticker, values in (
            ("TEST", [100.0, 110.0, 121.0, 133.1, 146.41, 161.051, 177.1561]),
            ("SPY", [100.0] * 7),
        )
    ], ignore_index=True)

    before_event = pd.Timestamp("2024-01-09 12:00:00", tz="UTC")  # 07:00 New York
    after_event = pd.Timestamp("2024-01-09 21:00:00", tz="UTC")  # 16:00 New York

    assert market_features(prices, "TEST", before_event)["price_asof"] == 100.0
    assert market_features(prices, "TEST", after_event)["price_asof"] == 110.0
    assert event_returns(prices, "TEST", before_event)["return_1d_target"] == pytest.approx(0.10)
    assert event_returns(prices, "TEST", after_event)["return_1d_target"] == pytest.approx(0.10)


def test_yahoo_eps_history_keeps_targets_and_consensus_on_same_split_basis() -> None:
    events = pd.DataFrame({
        "earnings_date": pd.date_range("2023-01-01", periods=5, freq="QE", tz="UTC"),
        "actual_eps": [0.10, 0.20, 0.30, 0.40, 0.50],
    })
    as_of = events["earnings_date"].iloc[-1]

    quarterly = historical_eps_features(events, as_of, "quarterly")
    annual = historical_eps_features(events, as_of, "annual")

    assert quarterly["lag_eps_diluted"] == 0.40
    assert quarterly["ttm_eps"] == pytest.approx(1.0)
    assert annual["lag_eps_diluted"] == pytest.approx(1.0)
    assert aligned_actual_eps(events, as_of, "annual") == pytest.approx(1.4)


def test_missing_surprise_is_not_counted_as_an_earnings_miss() -> None:
    events = pd.DataFrame({
        "earnings_date": pd.date_range("2023-01-01", periods=3, freq="QE", tz="UTC"),
        "actual_eps": [1.1, np.nan, 0.9],
        "consensus_eps": [1.0, 1.0, 1.0],
    })

    features = historical_surprise_features(events, pd.Timestamp("2024-01-01", tz="UTC"))

    assert features["earnings_beat_rate_4"] == pytest.approx(0.5)


def test_historical_consensus_recovers_eps_and_builds_point_in_time_annual_proxy() -> None:
    events = pd.DataFrame({
        "ticker": "TEST",
        "earnings_date": pd.date_range("2023-01-01", periods=5, freq="QE", tz="UTC"),
        "consensus_eps": [1.0, 1.1, np.nan, 1.3, 99.0],
        "actual_eps": [1.1, 1.2, 1.32, 1.4, 99.0],
        "eps_surprise_pct": [10.0, None, 10.0, None, None],
    })

    enriched = enrich_historical_consensus(events)

    assert enriched.loc[2, "consensus_eps"] == pytest.approx(1.2)
    assert enriched.loc[2, "consensus_eps_source"] == "derived_from_yahoo_surprise"
    assert enriched.loc[3, "annual_consensus_eps"] == pytest.approx(4.6)
    assert enriched.loc[3, "annual_consensus_eps_source"] == "derived_trailing_quarterly_consensus"
    # A later estimate must not change the annual proxy available at row 3.
    assert enriched.loc[3, "annual_consensus_eps"] != enriched.loc[4, "annual_consensus_eps"]


def test_ambiguous_sign_crossing_surprise_does_not_invent_consensus() -> None:
    events = pd.DataFrame({
        "ticker": ["TEST"],
        "earnings_date": [pd.Timestamp("2024-01-01", tz="UTC")],
        "consensus_eps": [np.nan],
        "actual_eps": [1.0],
        "eps_surprise_pct": [200.0],
    })

    enriched = enrich_historical_consensus(events)

    assert pd.isna(enriched.loc[0, "consensus_eps"])
