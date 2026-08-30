from __future__ import annotations

import numpy as np
import pandas as pd

from .fundamentals import safe_divide


def _series_before(prices: pd.DataFrame, ticker: str, as_of: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    return prices.loc[(prices["ticker"] == ticker) & (dates < cutoff)].sort_values("date")


def _trailing_return(close: pd.Series, sessions: int) -> float:
    return safe_divide(close.iloc[-1] - close.iloc[-1 - sessions], close.iloc[-1 - sessions]) if len(close) > sessions else np.nan


def market_features(prices: pd.DataFrame, ticker: str, as_of: pd.Timestamp, benchmark: str = "SPY") -> dict[str, float]:
    stock = _series_before(prices, ticker, as_of)
    market = _series_before(prices, benchmark, as_of)
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
    dates = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    event_day = pd.Timestamp(as_of)
    event_day = event_day.tz_localize("UTC") if event_day.tzinfo is None else event_day.tz_convert("UTC")
    event_date = event_day.date()

    def calculate(symbol: str) -> dict[int, float]:
        frame = prices.loc[prices["ticker"] == symbol].copy()
        frame["_date"] = dates.loc[frame.index].dt.date
        frame = frame.sort_values("_date")
        close = pd.to_numeric(frame["adj_close"] if "adj_close" in frame else frame["close"], errors="coerce")
        before = frame.index[frame["_date"] < event_date]
        if len(before) == 0:
            return {}
        base_position = frame.index.get_loc(before[-1])
        after_close = "after" in timing.lower() or "amc" in timing.lower()
        # For AMC, event-day close is still pre-event, so skip it.
        offset = 1 if after_close else 0
        result: dict[int, float] = {}
        for horizon in (1, 3, 5):
            position = base_position + horizon + offset
            if position < len(frame):
                result[horizon] = safe_divide(close.iloc[position] - close.iloc[base_position], close.iloc[base_position])
        return result

    stock, market = calculate(ticker), calculate(benchmark)
    output: dict[str, float] = {}
    for horizon in (1, 3, 5):
        output[f"return_{horizon}d_target"] = stock.get(horizon, np.nan)
        output[f"abnormal_return_{horizon}d"] = stock.get(horizon, np.nan) - market.get(horizon, np.nan)
    return output
