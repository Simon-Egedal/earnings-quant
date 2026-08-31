from __future__ import annotations

from io import StringIO
from pathlib import Path
import random

import pandas as pd
import requests


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA", "JPM",
    "WMT", "LLY", "V", "ORCL", "MA", "NFLX", "XOM", "COST", "JNJ", "HD", "PG", "ABBV",
    "BAC", "KO", "PLTR", "AMD", "CSCO", "PM", "CVX", "UNH", "IBM", "GE", "MCD", "CAT",
    "CRM", "ADBE", "INTC", "QCOM", "TXN", "AMAT", "DIS", "WFC", "GS", "MS", "AXP", "C",
    "BLK", "MRK", "TMO", "ABT", "AMGN", "GILD", "PEP", "NEE", "ACN", "ADI", "ADP", "AEP",
    "AFL", "AJG", "ALL", "AMT", "AON", "APD", "APO", "BA", "BK", "BKNG", "BMY", "BSX",
    "CB", "CCI", "CL", "CME", "CMCSA", "COP", "CRWD", "CTAS", "DHR", "DUK", "EMR", "EOG",
    "ETN", "F", "FDX", "GD", "GM", "HON", "ICE", "ISRG", "KDP", "KMB", "LIN", "LMT", "LOW",
    "MDT", "MMC", "MO", "MPC", "MU", "NOC", "NOW", "OXY", "PANW", "PFE", "PNC", "REGN",
    "RTX", "SBUX", "SCHW", "SO", "SPGI", "SYK", "T", "TJX", "TMUS", "UPS", "USB", "VZ",
    "ZTS", "DE", "ELV", "HCA", "MAR", "MCO", "MDLZ", "NKE",
]


def _fetch_sp500() -> list[str]:
    response = requests.get(
        SP500_URL,
        headers={"User-Agent": "earnings-quant/0.1 research universe loader"},
        timeout=30,
    )
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text), attrs={"id": "constituents"})[0]
    return table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().tolist()


def select_universe(
    tickers: list[str], size: int | None, seed: int = 42,
    preferred: list[str] | None = None,
) -> list[str]:
    """Return a stable broad sample, retaining already collected symbols first."""
    unique = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()))
    if size is None or size <= 0 or len(unique) <= size:
        return unique
    available = set(unique)
    retained = list(dict.fromkeys(
        str(ticker).strip().upper() for ticker in (preferred or [])
        if str(ticker).strip().upper() in available
    ))
    if len(retained) >= size:
        selected = set(random.Random(seed).sample(retained, size))
    else:
        remaining = [ticker for ticker in unique if ticker not in set(retained)]
        selected = set(retained) | set(random.Random(seed).sample(remaining, size - len(retained)))
    return [ticker for ticker in unique if ticker in selected]


def load_universe(config: dict, cache_dir: Path, refresh: bool = False) -> list[str]:
    configured = config.get("tickers") or []
    if configured:
        return sorted({str(t).upper() for t in configured})
    cache = cache_dir / "sp500_universe.parquet"
    minimum_size = max(1, int(config.get("target_size", 100)))
    if cache.exists() and not refresh:
        cached = pd.read_parquet(cache)["ticker"].dropna().astype(str).tolist()
        if len(cached) >= minimum_size:
            return cached
    try:
        tickers = _fetch_sp500()
    except Exception:
        tickers = FALLBACK
    frame = pd.DataFrame({"ticker": list(dict.fromkeys(tickers))})
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame["ticker"].tolist()
