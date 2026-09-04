from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.models.ticker_forecast as ticker_forecast
from src.providers.alpha_vantage import AlphaVantageProvider, normalize_alpha_vantage_fundamentals


def test_alpha_vantage_statements_normalize_to_model_schema() -> None:
    income = {"quarterlyReports": [{
        "fiscalDateEnding": "2025-03-31", "reportedCurrency": "NOK",
        "totalRevenue": "1000", "grossProfit": "300", "operatingIncome": "100",
        "netIncome": "75",
    }]}
    balance = {"quarterlyReports": [{
        "fiscalDateEnding": "2025-03-31", "totalAssets": "2000",
        "totalCurrentAssets": "800", "totalCurrentLiabilities": "600",
        "cashAndCashEquivalentsAtCarryingValue": "200", "longTermDebt": "350",
        "totalShareholderEquity": "900", "commonStockSharesOutstanding": "100",
    }]}
    cash = {"quarterlyReports": [{
        "fiscalDateEnding": "2025-03-31", "operatingCashflow": "120",
        "capitalExpenditures": "-20",
    }]}
    earnings = {"quarterlyEarnings": [{
        "fiscalDateEnding": "2025-03-31", "reportedDate": "2025-04-24",
        "reportedEPS": "0.75",
    }]}

    frame = normalize_alpha_vantage_fundamentals("ATEA.OL", income, balance, cash, earnings)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["ticker"] == "ATEA.OL"
    assert row["revenue"] == 1000
    assert row["eps_diluted"] == 0.75
    assert row["free_cash_flow"] == 100
    assert row["statement_type"] == "quarterly"
    assert row["data_source"] == "Alpha Vantage"
    assert row["currency"] == "NOK"
    assert row["filed_at"] == pd.Timestamp("2025-04-25")


def test_exchange_qualified_symbol_does_not_match_unrelated_us_company() -> None:
    provider = object.__new__(AlphaVantageProvider)
    provider._request = lambda *args, **kwargs: {"bestMatches": [
        {
            "1. symbol": "ATEA", "2. name": "Astea International Inc",
            "4. region": "United States", "8. currency": "USD", "9. matchScore": "1.0",
        },
        {
            "1. symbol": "0JWO.LON", "2. name": "Atea ASA",
            "4. region": "United Kingdom", "8. currency": "NOK", "9. matchScore": "0.6667",
        },
    ]}

    assert provider.resolve_symbol("ATEA.OL") == "0JWO.LON"


def _config(tmp_path: Path) -> dict:
    return {
        "project": {"data_dir": tmp_path},
        "sec": {
            "user_agent": "Tests tests@example.com", "requests_per_second": 5,
            "cache_days": 7,
        },
        "alpha_vantage": {"api_key_env": "TEST_ALPHA_KEY", "cache_days": 7},
    }


def test_global_fallback_explains_missing_api_key(monkeypatch, tmp_path: Path) -> None:
    class MissingSEC:
        def __init__(self, *args, **kwargs):
            pass

        def fundamentals(self, ticker: str) -> pd.DataFrame:
            raise KeyError(ticker)

    monkeypatch.setattr(ticker_forecast, "SECCompanyFactsProvider", MissingSEC)
    monkeypatch.delenv("TEST_ALPHA_KEY", raising=False)

    with pytest.raises(ValueError, match="TEST_ALPHA_KEY"):
        ticker_forecast.load_ticker_fundamentals("ATEA.OL", _config(tmp_path))


def test_api_key_can_be_loaded_from_ignored_dotenv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TEST_ALPHA_KEY", raising=False)
    (tmp_path / ".env").write_text("TEST_ALPHA_KEY=local-free-key\n", encoding="utf-8")

    assert ticker_forecast._load_api_key("TEST_ALPHA_KEY", tmp_path) == "local-free-key"


def test_global_fallback_uses_alpha_vantage(monkeypatch, tmp_path: Path) -> None:
    expected = pd.DataFrame({"ticker": ["ATEA.OL"]})

    class MissingSEC:
        def __init__(self, *args, **kwargs):
            pass

        def fundamentals(self, ticker: str) -> pd.DataFrame:
            raise KeyError(ticker)

    class FakeAlpha:
        def __init__(self, cache_dir: Path, api_key: str, cache_days: int):
            assert api_key == "free-key"

        def fundamentals(self, ticker: str) -> pd.DataFrame:
            return expected

    monkeypatch.setattr(ticker_forecast, "SECCompanyFactsProvider", MissingSEC)
    monkeypatch.setattr(ticker_forecast, "AlphaVantageProvider", FakeAlpha)
    monkeypatch.setenv("TEST_ALPHA_KEY", "free-key")

    frame, source = ticker_forecast.load_ticker_fundamentals("ATEA.OL", _config(tmp_path))

    assert frame is expected
    assert source == "Alpha Vantage"
