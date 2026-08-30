from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def quarterly_fundamentals() -> pd.DataFrame:
    periods = pd.date_range("2022-03-31", periods=8, freq="QE")
    filed = periods + pd.offsets.Day(35)
    revenue = np.arange(100, 180, 10, dtype=float)
    return pd.DataFrame({
        "ticker": "TEST", "period_end": periods, "filed_at": filed,
        "revenue": revenue, "gross_profit": revenue * .5, "operating_income": revenue * .2,
        "net_income": revenue * .12, "eps_diluted": np.arange(1, 1.8, .1),
        "operating_cash_flow": revenue * .18, "capital_expenditures": revenue * .04,
        "free_cash_flow": revenue * .14, "cash": 50.0, "total_assets": 500.0,
        "current_assets": 200.0, "current_liabilities": 100.0, "total_debt": 80.0,
        "stockholders_equity": 300.0, "shares_outstanding": 100.0,
        "fiscal_year": periods.year, "fiscal_period": ["Q1", "Q2", "Q3", "FY"] * 2,
        "form": "10-Q", "accession": [f"a{i}" for i in range(8)],
    })


@pytest.fixture
def prices() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", "2025-03-01", freq="B", tz="UTC")
    rows = []
    for ticker, drift in (("TEST", .0005), ("SPY", .0002)):
        close = 100 * np.cumprod(np.repeat(1 + drift, len(dates)))
        rows.append(pd.DataFrame({"date": dates, "ticker": ticker, "close": close, "adj_close": close, "volume": 1_000_000.0}))
    return pd.concat(rows, ignore_index=True)
