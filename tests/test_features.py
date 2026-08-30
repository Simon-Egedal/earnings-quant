from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.fundamentals import point_in_time_fundamentals
from src.features.market import market_features


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
