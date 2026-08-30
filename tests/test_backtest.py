from __future__ import annotations

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.walk_forward import expanding_year_folds, split_fold


def test_signals_costs_and_metrics():
    frame = pd.DataFrame({
        "earnings_date": pd.date_range("2025-01-01", periods=3, freq="30D", tz="UTC"),
        "predicted_abnormal_return_3d": [.03, -.04, .01], "probability_up": [.7, .3, .7],
        "abnormal_return_3d": [.05, -.02, .03],
    })
    config = {"backtest": {"long_threshold": .02, "short_threshold": -.02, "minimum_confidence": .55, "commission": 0, "slippage": .001}}
    results, metrics = run_backtest(frame, config)
    assert results["signal"].tolist() == ["LONG", "SHORT", "NO_TRADE"]
    assert results["strategy_return"].round(3).tolist() == [.048, .018, 0]
    assert metrics["overall"]["number_of_signals"] == 2


def test_walk_forward_is_strictly_chronological():
    folds = expanding_year_folds(2019, 2022, 2024)
    assert folds[0].train_years == (2019, 2020)
    frame = pd.DataFrame({"event_year": [2019, 2020, 2021, 2022], "earnings_date": pd.to_datetime(["2019-02-01", "2020-02-01", "2021-02-01", "2022-02-01"])})
    train, validation, test = split_fold(frame, folds[0])
    assert train["event_year"].max() < validation["event_year"].min() < test["event_year"].min()

