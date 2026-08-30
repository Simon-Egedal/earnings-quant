from __future__ import annotations

import numpy as np
import pandas as pd

from src.event_time import event_session_date, event_timing
from .fundamentals import safe_divide


def _price_dates(prices: pd.DataFrame) -> pd.Series:
    if "_session_date" in prices:
        return prices["_session_date"]
    return pd.to_datetime(prices["date"], utc=True, errors="coerce").dt.date


def _series_before(
    prices: pd.DataFrame, ticker: str, as_of: pd.Timestamp, timing: str = ""
) -> pd.DataFrame:
    session_date = event_session_date(as_of)
    if session_date is None:
        return prices.iloc[0:0].copy()
    price_dates = _price_dates(prices)
    known_after_close = event_timing(as_of, timing) == "after_market"
    visible = price_dates <= session_date if known_after_close else price_dates < session_date
    return prices.loc[(prices["ticker"] == ticker) & visible].sort_values("date")


def _trailing_return(close: pd.Series, sessions: int) -> float:
    return safe_divide(close.iloc[-1] - close.iloc[-1 - sessions], close.iloc[-1 - sessions]) if len(close) > sessions else np.nan


def market_features(
    prices: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    benchmark: str = "SPY",
    timing: str = "",
) -> dict[str, float]:
    stock = _series_before(prices, ticker, as_of, timing)
    market = _series_before(prices, benchmark, as_of, timing)
    if stock.empty:
        return {}
    close = pd.to_numeric(stock["adj_close"] if "adj_close" in stock else stock["close"], errors="coerce").dropna()
    volume = pd.to_numeric(stock.get("volume", pd.Series(dtype=float)), errors="coerce").dropna()
    daily = close.pct_change(fill_method=None)
    output = {f"return_{n}d": _trailing_return(close, n) for n in (5, 20, 60, 120)}
    output.update({
        "distance_from_52w_high": safe_divide(close.iloc[-1], close.tail(252).max()) - 1,
        "volatility_20d": daily.tail(20).std() * np.sqrt(252),
        "volatility_60d": daily.tail(60).std() * np.sqrt(252),
        "volume_change": safe_divide(volume.tail(5).mean(), volume.tail(20).mean()) - 1 if len(volume) >= 20 else np.nan,
        "relative_volume": safe_divide(volume.iloc[-1], volume.tail(20).mean()) if len(volume) >= 20 else np.nan,
        "price_asof": float(close.iloc[-1]),
    })
    if not market.empty:
        market_close = pd.to_numeric(market["adj_close"] if "adj_close" in market else market["close"], errors="coerce").dropna()
        for n in (20, 60):
            output[f"spy_return_{n}d"] = _trailing_return(market_close, n)
            output[f"relative_return_{n}d"] = output[f"return_{n}d"] - output[f"spy_return_{n}d"]
    return {key: float(value) for key, value in output.items()}


def event_returns(prices: pd.DataFrame, ticker: str, as_of: pd.Timestamp, benchmark: str = "SPY", timing: str = "") -> dict[str, float]:
    """Post-event close-to-close returns, adjusted for before/after-market timing."""
    dates = _price_dates(prices)
    event_date = event_session_date(as_of)
    normalized_timing = event_timing(as_of, timing)

    def calculate(symbol: str) -> dict[int, float]:
        frame = prices.loc[prices["ticker"] == symbol].copy()
        frame["_date"] = dates.loc[frame.index]
        frame["_close"] = pd.to_numeric(
            frame["adj_close"] if "adj_close" in frame else frame["close"], errors="coerce"
        )
        frame = frame.dropna(subset=["_date", "_close"]).sort_values("_date").reset_index(drop=True)
        if frame.empty or event_date is None:
            return {}
        eligible = frame.index[
            frame["_date"] <= event_date
            if normalized_timing == "after_market"
            else frame["_date"] < event_date
        ]
        if len(eligible) == 0:
            return {}
        base_position = int(eligible[-1])
        base_close = float(frame.loc[base_position, "_close"])
        result: dict[int, float] = {}
        for horizon in (1, 3, 5):
            position = base_position + horizon
            if position < len(frame):
                result[horizon] = safe_divide(
                    frame.loc[position, "_close"] - base_close, base_close
                )
        return result

    stock, market = calculate(ticker), calculate(benchmark)
    output: dict[str, float] = {}
    for horizon in (1, 3, 5):
        output[f"return_{horizon}d_target"] = stock.get(horizon, np.nan)
        output[f"abnormal_return_{horizon}d"] = stock.get(horizon, np.nan) - market.get(horizon, np.nan)
    return output
