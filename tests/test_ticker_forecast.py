from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ticker_forecast import (
    accuracy_score_01,
    build_supervised_history,
    prepare_quarterly_history,
    select_model,
)


def _history(periods: int = 28) -> pd.DataFrame:
    index = np.arange(periods, dtype=float)
    return pd.DataFrame({
        "period_end": pd.date_range("2019-03-31", periods=periods, freq="QE"),
        "filed_at": pd.date_range("2019-05-01", periods=periods, freq="QE"),
        "statement_type": "quarterly",
        "fiscal_period": [f"Q{int(value % 4) + 1}" for value in index],
        "eps_diluted": 1.0 + index * 0.04,
        "revenue": 1_000.0 + index * 25.0,
        "operating_margin": 0.20 + index * 0.001,
        "free_cash_flow": 100.0 + index * 3.0,
    })


def test_accuracy_score_has_clear_zero_to_one_endpoints() -> None:
    actual = pd.Series([100.0, 200.0])
    assert accuracy_score_01(actual, actual) == 1.0
    assert accuracy_score_01(actual, pd.Series([0.0, 0.0])) == 0.0


def test_supervised_features_do_not_see_current_target() -> None:
    history = _history(12)
    features, targets = build_supervised_history(history)
    assert features.loc[0, "revenue_lag1"] == history.loc[3, "revenue"]
    assert targets.loc[0, "revenue"] == history.loc[4, "revenue"]


def test_prepare_history_uses_only_visible_last_ten_years() -> None:
    history = _history(28)
    amendment = history.iloc[[10]].copy()
    amendment["filed_at"] = pd.Timestamp("2026-01-01", tz="UTC")
    amendment["revenue"] = 999_999.0
    raw = pd.concat([history, amendment], ignore_index=True)

    prepared = prepare_quarterly_history(raw, as_of=pd.Timestamp("2025-01-01", tz="UTC"), years=10)

    assert prepared.loc[prepared["period_end"].eq(history.loc[10, "period_end"]), "revenue"].iloc[0] != 999_999.0


def test_model_search_reports_chronological_accuracy() -> None:
    selected, attempts, fitted = select_model(
        _history(), threshold=0.8, minimum_train_rows=8, backtest_rows=6, seed=42
    )

    assert 0.0 <= selected.score <= 1.0
    assert attempts
    assert selected.name in {attempt.name for attempt in attempts}
    assert set(fitted) == {"models", "next_features"}


def test_model_search_rejects_too_little_history() -> None:
    with pytest.raises(ValueError, match="usable quarterly periods"):
        select_model(_history(10), minimum_train_rows=8)
