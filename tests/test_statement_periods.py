from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.builder import EventDatasetBuilder
from src.features.fundamentals import infer_next_statement_type, point_in_time_fundamentals
from src.providers.sec import classify_statement_type, combine_total_debt, derive_fourth_quarters
from src.providers.yahoo import YahooFinanceProvider


def test_sec_fact_cadence_classification_separates_quarter_ytd_and_year() -> None:
    assert classify_statement_type("10-Q", "Q1", 90) == "quarterly"
    assert classify_statement_type("10-Q", "Q2", 181) == "year_to_date"
    assert classify_statement_type("10-K", "FY", 365) == "annual"
    assert classify_statement_type("10-K", "FY", None, instant=True) == "annual"


def test_total_debt_combines_long_and_short_without_treating_missing_as_zero() -> None:
    frame = pd.DataFrame({
        "long_term_debt": [90.0, np.nan, np.nan],
        "short_term_debt": [10.0, 5.0, np.nan],
        "total_debt": [np.nan, 7.0, np.nan],
    })

    result = combine_total_debt(frame)

    assert result["total_debt"].iloc[0] == 100.0
    assert result["total_debt"].iloc[1] == 7.0
    assert np.isnan(result["total_debt"].iloc[2])


def test_fourth_quarter_is_derived_from_annual_less_first_three_quarters() -> None:
    periods = pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"])
    frame = pd.DataFrame({
        "ticker": "TEST",
        "period_end": periods,
        "period_start": pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01", "2025-01-01"]),
        "filed_at": pd.to_datetime(["2025-05-01", "2025-08-01", "2025-11-01", "2026-02-01"]),
        "fiscal_year": 2025,
        "fiscal_period": ["Q1", "Q2", "Q3", "FY"],
        "form": ["10-Q", "10-Q", "10-Q", "10-K"],
        "accession": ["q1", "q2", "q3", "fy"],
        "statement_type": ["quarterly", "quarterly", "quarterly", "annual"],
        "duration_days": [89, 90, 91, 364],
        "frame": None,
        "revenue": [20.0, 25.0, 30.0, 110.0],
        "eps_diluted": [1.0, 1.1, 1.2, 4.8],
    })

    result = derive_fourth_quarters(frame)
    fourth = result.loc[result["form"].eq("DERIVED-Q4")].iloc[0]

    assert fourth["statement_type"] == "quarterly"
    assert fourth["revenue"] == 35.0
    assert fourth["eps_diluted"] == 1.5


def test_annual_features_use_full_year_history_without_four_quarter_sum() -> None:
    history = pd.DataFrame({
        "ticker": "TEST",
        "period_end": pd.to_datetime(["2023-12-31", "2024-12-31"]),
        "filed_at": pd.to_datetime(["2024-02-15", "2025-02-15"]),
        "statement_type": "annual",
        "revenue": [400.0, 440.0],
        "eps_diluted": [4.0, 4.5],
        "operating_income": [80.0, 99.0],
    })

    features, _ = point_in_time_fundamentals(
        history, pd.Timestamp("2025-03-01", tz="UTC"), "annual"
    )

    assert features["statement_is_annual"] == 1.0
    assert features["lag_revenue"] == 440.0
    assert features["ttm_revenue"] == 440.0
    assert np.isclose(features["revenue_yoy"], 0.10)


def test_trailing_revenue_requires_four_consecutive_periods() -> None:
    history = pd.DataFrame({
        "ticker": "TEST",
        "period_end": pd.date_range("2024-03-31", periods=6, freq="QE"),
        "filed_at": pd.date_range("2024-05-01", periods=6, freq="QE"),
        "statement_type": "quarterly",
        "revenue": [100.0, None, 110.0, 120.0, None, 130.0],
        "eps_diluted": [1.0, None, 1.1, 1.2, None, 1.3],
    })

    features, _ = point_in_time_fundamentals(
        history, pd.Timestamp("2026-01-01", tz="UTC"), "quarterly"
    )

    assert features["revenue_history_count"] == 4.0
    assert np.isnan(features["ttm_revenue"])
    assert np.isnan(features["revenue_qoq"])


def test_next_statement_after_q3_is_annual() -> None:
    history = pd.DataFrame({
        "filed_at": pd.to_datetime(["2025-05-01", "2025-08-01", "2025-11-01"], utc=True),
        "form": ["10-Q", "10-Q", "10-Q"],
        "fiscal_period": ["Q1", "Q2", "Q3"],
        "accession": ["q1", "q2", "q3"],
    })

    assert infer_next_statement_type(history, pd.Timestamp("2026-01-01", tz="UTC")) == ("annual", "FY")


def test_dataset_uses_annual_target_for_event_after_q3(quarterly_fundamentals, prices) -> None:
    quarterly = quarterly_fundamentals.copy()
    quarterly["statement_type"] = "quarterly"
    prior_annual = quarterly.iloc[-1].copy()
    prior_annual.update({
        "period_end": pd.Timestamp("2022-12-31"), "filed_at": pd.Timestamp("2023-02-15"),
        "fiscal_period": "FY", "form": "10-K", "accession": "annual-2022",
        "statement_type": "annual", "revenue": 600.0, "eps_diluted": 5.0,
    })
    future_annual = prior_annual.copy()
    future_annual.update({
        "period_end": pd.Timestamp("2023-12-31"), "filed_at": pd.Timestamp("2024-02-15"),
        "accession": "annual-2023", "revenue": 700.0, "eps_diluted": 6.0,
    })
    fundamentals = pd.concat([quarterly, pd.DataFrame([prior_annual, future_annual])], ignore_index=True)
    earnings = pd.DataFrame({
        "ticker": ["TEST"], "earnings_date": [pd.Timestamp("2024-01-25", tz="UTC")],
        "actual_eps": [1.5], "consensus_eps": [1.4], "timing": ["After Market Close"],
    })

    dataset = EventDatasetBuilder().build(fundamentals, earnings, prices)

    assert len(dataset) == 1
    assert dataset.loc[0, "statement_type"] == "annual"
    assert dataset.loc[0, "actual_revenue"] == 700.0
    assert dataset.loc[0, "actual_eps"] == 6.0
    assert pd.isna(dataset.loc[0, "consensus_eps"])


def test_dataset_uses_trailing_quarterly_consensus_for_annual_event(
    quarterly_fundamentals, prices
) -> None:
    quarterly = quarterly_fundamentals.copy()
    quarterly["statement_type"] = "quarterly"
    prior_annual = quarterly.iloc[-1].copy()
    prior_annual.update({
        "period_end": pd.Timestamp("2022-12-31"), "filed_at": pd.Timestamp("2023-02-15"),
        "fiscal_period": "FY", "form": "10-K", "accession": "annual-2022",
        "statement_type": "annual", "revenue": 600.0, "eps_diluted": 5.0,
    })
    future_annual = prior_annual.copy()
    future_annual.update({
        "period_end": pd.Timestamp("2023-12-31"), "filed_at": pd.Timestamp("2024-02-15"),
        "accession": "annual-2023", "revenue": 700.0, "eps_diluted": 6.0,
    })
    fundamentals = pd.concat(
        [quarterly, pd.DataFrame([prior_annual, future_annual])], ignore_index=True
    )
    earnings = pd.DataFrame({
        "ticker": "TEST",
        "earnings_date": pd.to_datetime([
            "2023-04-25", "2023-07-25", "2023-10-25", "2024-01-25"
        ], utc=True),
        "actual_eps": [1.0, 1.1, 1.2, 1.5],
        "consensus_eps": [0.9, 1.0, 1.1, 1.4],
        "timing": "After Market Close",
    })

    dataset = EventDatasetBuilder().build(fundamentals, earnings, prices)
    annual = dataset.loc[dataset["earnings_date"].eq(earnings["earnings_date"].iloc[-1])].iloc[0]

    assert annual["statement_type"] == "annual"
    assert annual["quarterly_consensus_eps"] == 1.4
    assert annual["consensus_eps"] == 4.4
    assert annual["consensus_eps_available"] == 1.0
    assert annual["consensus_eps_is_derived"] == 1.0


def test_live_analyst_snapshot_keeps_quarterly_and_annual_estimates() -> None:
    class Instrument:
        def get_calendar(self):
            return {"Earnings Average": 1.5, "Revenue Average": 25.0}

        def get_eps_trend(self):
            return pd.DataFrame({"current": [1.5, 6.5]}, index=["0q", "0y"])

        def get_eps_revisions(self):
            return pd.DataFrame()

        def get_revenue_estimate(self):
            return pd.DataFrame({"avg": [25.0, 110.0]}, index=["0q", "0y"])

    class FakeYFinance:
        @staticmethod
        def Ticker(ticker):
            return Instrument()

    provider = YahooFinanceProvider.__new__(YahooFinanceProvider)
    provider.yf = FakeYFinance()

    snapshot = provider.analyst_snapshot("TEST")

    assert snapshot["consensus_eps"] == 1.5
    assert snapshot["annual_consensus_eps"] == 6.5
    assert snapshot["consensus_revenue"] == 25.0
    assert snapshot["annual_consensus_revenue"] == 110.0


def test_historical_yahoo_schema_accepts_revenue_consensus_when_available() -> None:
    class Instrument:
        def get_earnings_dates(self, limit=100):
            frame = pd.DataFrame({
                "EPS Estimate": [1.5],
                "Reported EPS": [1.6],
                "Surprise(%)": [6.67],
                "Revenue Estimate": [25.0],
                "Reported Revenue": [26.0],
            }, index=[pd.Timestamp("2025-01-01", tz="UTC")])
            frame.index.name = "Earnings Date"
            return frame

    class FakeYFinance:
        @staticmethod
        def Ticker(ticker):
            return Instrument()

    provider = YahooFinanceProvider.__new__(YahooFinanceProvider)
    provider.yf = FakeYFinance()

    history = provider.historical_earnings("TEST")

    assert history.loc[0, "consensus_eps"] == 1.5
    assert history.loc[0, "consensus_revenue"] == 25.0
    assert history.loc[0, "actual_revenue_yahoo"] == 26.0
