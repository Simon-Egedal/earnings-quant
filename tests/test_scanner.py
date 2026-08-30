from __future__ import annotations

import pandas as pd
import pytest

from src.scanner import earnings_scanner


def test_growth_comparisons_use_comparable_prior_year_quarter() -> None:
    live = pd.DataFrame({
        "predicted_eps": [1.20],
        "consensus_eps": [1.10],
        "prior_year_eps": [1.00],
        "predicted_revenue": [125.0],
        "consensus_revenue": [120.0],
        "prior_year_revenue": [100.0],
    })

    result = earnings_scanner._add_growth_comparisons(live)

    assert result.loc[0, "predicted_eps_growth_yoy"] == pytest.approx(0.20)
    assert result.loc[0, "consensus_eps_growth_yoy"] == pytest.approx(0.10)
    assert result.loc[0, "predicted_revenue_growth_yoy"] == pytest.approx(0.25)
    assert result.loc[0, "consensus_revenue_growth_yoy"] == pytest.approx(0.20)


def test_reported_comparison_ignores_non_metric_balance_sheet_dates() -> None:
    history = pd.DataFrame({
        "period_end": pd.to_datetime(["2025-08-16", "2025-09-16", "2026-05-23", "2026-06-24"]),
        "filed_at": pd.to_datetime(["2025-09-20", "2025-09-20", "2026-06-26", "2026-06-26"], utc=True),
        "revenue": [100.0, None, 125.0, None],
    })

    latest, comparable = earnings_scanner._reported_comparison(
        history, pd.Timestamp("2026-09-11", tz="UTC"), "revenue"
    )

    assert latest == 125.0
    assert comparable == 100.0


def test_scan_upcoming_restricts_calendar_to_selected_tickers(monkeypatch) -> None:
    calendar = pd.DataFrame({"ticker": ["AAPL", "NVDA", "MSFT"], "earnings_date": pd.date_range("2026-09-01", periods=3)})
    captured: dict[str, object] = {}

    monkeypatch.setattr(earnings_scanner, "get_upcoming_calendar", lambda config, days: calendar)

    def fake_scan_events(config, events, top):
        captured["events"] = events
        captured["top"] = top
        return events

    monkeypatch.setattr(earnings_scanner, "scan_events", fake_scan_events)
    result = earnings_scanner.scan_upcoming({}, days=30, top=5, tickers=["nvda"])

    assert result["ticker"].tolist() == ["NVDA"]
    assert captured["top"] == 5


def test_prediction_quality_rejects_undertrained_and_implausible_forecast() -> None:
    row = pd.Series({
        "statement_type": "quarterly",
        "revenue_history_count": 4,
        "eps_diluted_history_count": 4,
        "lag_revenue": 220_000_000.0,
        "consensus_revenue": 220_000_000.0,
        "predicted_revenue": 35_000_000_000.0,
        "predicted_operating_margin": -0.02,
        "predicted_fcf": 7_000_000_000.0,
    })

    reasons = earnings_scanner._prediction_quality_reasons(
        row, training_ticker_count=5, scanner_config={"minimum_training_tickers": 20}
    )

    assert any("only 5 tickers" in reason for reason in reasons)
    assert any("revenue forecast" in reason and "analyst consensus" in reason for reason in reasons)
