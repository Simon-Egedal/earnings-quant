from __future__ import annotations

from pathlib import Path

import pandas as pd


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA", "JPM",
    "WMT", "LLY", "V", "ORCL", "MA", "NFLX", "XOM", "COST", "JNJ", "HD", "PG", "ABBV",
    "BAC", "KO", "PLTR", "AMD", "CSCO", "PM", "CVX", "UNH", "IBM", "GE", "MCD", "CAT",
]


def load_universe(config: dict, cache_dir: Path, refresh: bool = False) -> list[str]:
    configured = config.get("tickers") or []
    if configured:
        return sorted({str(t).upper() for t in configured})
    cache = cache_dir / "sp500_universe.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)["ticker"].tolist()
    try:
        table = pd.read_html(SP500_URL, attrs={"id": "constituents"})[0]
        frame = pd.DataFrame({"ticker": table["Symbol"].str.replace(".", "-", regex=False).str.upper()})
    except Exception:
        frame = pd.DataFrame({"ticker": FALLBACK})
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame["ticker"].tolist()

