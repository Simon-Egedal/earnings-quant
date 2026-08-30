from __future__ import annotations

import pandas as pd

from src.data.collector import _merge_existing


def test_partial_collection_keeps_existing_tickers_and_replaces_updated_rows(tmp_path) -> None:
    path = tmp_path / "raw" / "prices.parquet"
    path.parent.mkdir()
    existing = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01"], utc=True),
        "close": [100.0, 200.0],
    })
    existing.to_parquet(path, index=False)
    update = pd.DataFrame({
        "ticker": ["AAPL"],
        "date": pd.to_datetime(["2025-01-01"], utc=True),
        "close": [101.0],
    })

    merged = _merge_existing(update, path, ["ticker", "date"])

    assert set(merged["ticker"]) == {"AAPL", "MSFT"}
    assert merged.loc[merged["ticker"].eq("AAPL"), "close"].item() == 101.0
