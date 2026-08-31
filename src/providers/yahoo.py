from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.event_time import event_timing
from src.logging_utils import log


def _snake(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("%", "pct").replace("/", "_").split())


class YahooFinanceProvider:
    """Thin, defensive wrapper around yfinance's public research endpoints."""

    def __init__(self) -> None:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required; run: python -m pip install -r requirements.txt") from exc
        self.yf = yf

    def upcoming_earnings(self, days: int = 14, minimum_market_cap: float = 0) -> pd.DataFrame:
        start = date.today()
        end = start + timedelta(days=days)
        calendar_type = getattr(self.yf, "Calendars", None)
        if calendar_type is None:
            raise RuntimeError("Installed yfinance lacks Calendars; upgrade yfinance")
        calendar = calendar_type(start=start, end=end)
        pages: list[pd.DataFrame] = []
        for offset in range(0, 1000, 100):
            page = calendar.get_earnings_calendar(
                market_cap=minimum_market_cap or None,
                filter_most_active=False,
                limit=100,
                offset=offset,
            )
            if page is None or page.empty:
                break
            pages.append(page.reset_index())
            if len(page) < 100:
                break
        if not pages:
            return pd.DataFrame(columns=["ticker", "company", "earnings_date", "timing"])
        frame = pd.concat(pages, ignore_index=True)
        frame.columns = [_snake(column) for column in frame.columns]
        aliases = {
            "symbol": "ticker", "company_name": "company", "company": "company",
            "earnings_date": "earnings_date", "start_date_time": "earnings_date",
            "eps_estimate": "consensus_eps", "reported_eps": "actual_eps",
        }
        frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
        if "ticker" not in frame:
            raise RuntimeError(f"Unexpected Yahoo calendar columns: {list(frame.columns)}")
        date_column = next((c for c in ("earnings_date", "event_start_date") if c in frame), None)
        if date_column:
            frame["earnings_date"] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
        if "timing" not in frame:
            frame["timing"] = [event_timing(value) for value in frame["earnings_date"]]
        log("SCAN", "%d companies reporting within %d days", len(frame), days)
        return frame.drop_duplicates(["ticker", "earnings_date"]).reset_index(drop=True)

    def historical_earnings(self, ticker: str, limit: int = 100) -> pd.DataFrame:
        frame = self.yf.Ticker(ticker).get_earnings_dates(limit=min(limit, 100))
        if frame is None or frame.empty:
            return pd.DataFrame()
        frame = frame.reset_index()
        frame.columns = [_snake(column) for column in frame.columns]
        frame = frame.rename(columns={
            "earnings_date": "earnings_date", "eps_estimate": "consensus_eps",
            "reported_eps": "actual_eps", "surprise(pct)": "eps_surprise_pct",
            "surprise_(pct)": "eps_surprise_pct",
            "revenue_estimate": "consensus_revenue",
            "reported_revenue": "actual_revenue_yahoo",
            "revenue_surprise(pct)": "revenue_surprise_pct",
            "revenue_surprise_(pct)": "revenue_surprise_pct",
        })
        for column in ("consensus_eps", "consensus_revenue"):
            if column not in frame:
                frame[column] = pd.NA
        frame["ticker"] = ticker.upper()
        frame["earnings_date"] = pd.to_datetime(frame["earnings_date"], utc=True, errors="coerce")
        frame["timing"] = [event_timing(value) for value in frame["earnings_date"]]
        return frame.dropna(subset=["earnings_date"]).sort_values("earnings_date")

    def price_history(self, tickers: Iterable[str], start: str, end: str | None = None) -> pd.DataFrame:
        symbols = sorted(set(tickers))
        if not symbols:
            return pd.DataFrame()
        raw = self.yf.download(symbols, start=start, end=end, auto_adjust=False, actions=False, progress=False, group_by="column")
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            frame = raw.stack(level=1, future_stack=True).rename_axis(index=["date", "ticker"]).reset_index()
        else:
            frame = raw.reset_index()
            frame["ticker"] = symbols[0]
        frame.columns = [_snake(column) for column in frame.columns]
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        return frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    def metadata(self, ticker: str) -> dict[str, Any]:
        instrument = self.yf.Ticker(ticker)
        try:
            info = instrument.get_info()
        except Exception:
            info = {}
        fast = instrument.fast_info
        return {
            "ticker": ticker.upper(), "company": info.get("longName") or info.get("shortName") or ticker.upper(),
            "sector": info.get("sector"), "industry": info.get("industry"),
            "market_cap": info.get("marketCap") or getattr(fast, "market_cap", None),
            "enterprise_value": info.get("enterpriseValue"), "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"), "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "enterprise_to_revenue": info.get("enterpriseToRevenue"),
            "enterprise_to_ebitda": info.get("enterpriseToEbitda"), "price_to_book": info.get("priceToBook"),
        }

    def analyst_snapshot(self, ticker: str) -> dict[str, Any]:
        instrument = self.yf.Ticker(ticker)
        output: dict[str, Any] = {"ticker": ticker.upper()}
        try:
            calendar = instrument.get_calendar() or {}
            output.update({
                "consensus_eps": calendar.get("Earnings Average"),
                "consensus_revenue": calendar.get("Revenue Average"),
            })
        except Exception:
            pass
        for method_name, prefix in (("get_eps_trend", "eps"), ("get_eps_revisions", "eps_revisions"), ("get_revenue_estimate", "revenue")):
            try:
                table = getattr(instrument, method_name)()
                if table is not None and not table.empty:
                    quarterly = table.loc["0q"] if "0q" in table.index else table.iloc[0]
                    output.update({f"{prefix}_{_snake(key)}": value for key, value in quarterly.items()})
                    if "0y" in table.index:
                        annual = table.loc["0y"]
                        output.update({f"annual_{prefix}_{_snake(key)}": value for key, value in annual.items()})
            except Exception:
                continue
        output.setdefault("consensus_eps", output.get("eps_current"))
        output.setdefault("consensus_revenue", output.get("revenue_avg"))
        output.setdefault("annual_consensus_eps", output.get("annual_eps_current"))
        output.setdefault("annual_consensus_revenue", output.get("annual_revenue_avg"))
        return output
