from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.logging_utils import log


BASE_URL = "https://www.alphavantage.co/query"


def _number(value: object) -> float:
    if value in (None, "", "None", "null", "-"):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _quarterly(payload: dict[str, Any]) -> list[dict[str, Any]]:
    reports = payload.get("quarterlyReports", [])
    return reports if isinstance(reports, list) else []


def normalize_alpha_vantage_fundamentals(
    ticker: str,
    income: dict[str, Any],
    balance: dict[str, Any],
    cash_flow: dict[str, Any],
    earnings: dict[str, Any],
) -> pd.DataFrame:
    """Normalize Alpha Vantage quarterly statements into the SEC-shaped schema."""
    records: dict[str, dict[str, Any]] = {}

    def record(report: dict[str, Any]) -> dict[str, Any] | None:
        period = str(report.get("fiscalDateEnding", ""))
        if not period:
            return None
        return records.setdefault(period, {"period_end": period})

    for report in _quarterly(income):
        row = record(report)
        if row is None:
            continue
        row.update({
            "revenue": _number(report.get("totalRevenue")),
            "gross_profit": _number(report.get("grossProfit")),
            "operating_income": _number(report.get("operatingIncome")),
            "net_income": _number(report.get("netIncome")),
            "currency": report.get("reportedCurrency"),
        })
    for report in _quarterly(balance):
        row = record(report)
        if row is None:
            continue
        total_debt = _number(report.get("shortLongTermDebtTotal"))
        if pd.isna(total_debt):
            long_term = _number(report.get("longTermDebt"))
            short_term = _number(report.get("shortTermDebt"))
            components = [value for value in (long_term, short_term) if pd.notna(value)]
            total_debt = float(sum(components)) if components else np.nan
        row.update({
            "cash": _number(report.get("cashAndCashEquivalentsAtCarryingValue")),
            "total_assets": _number(report.get("totalAssets")),
            "current_assets": _number(report.get("totalCurrentAssets")),
            "current_liabilities": _number(report.get("totalCurrentLiabilities")),
            "total_debt": total_debt,
            "stockholders_equity": _number(report.get("totalShareholderEquity")),
            "shares_outstanding": _number(report.get("commonStockSharesOutstanding")),
        })
    for report in _quarterly(cash_flow):
        row = record(report)
        if row is None:
            continue
        operating_cash_flow = _number(report.get("operatingCashflow"))
        capital_expenditures = _number(report.get("capitalExpenditures"))
        free_cash_flow = _number(report.get("freeCashFlow"))
        if pd.isna(free_cash_flow) and pd.notna(operating_cash_flow) and pd.notna(capital_expenditures):
            free_cash_flow = operating_cash_flow - abs(capital_expenditures)
        row.update({
            "operating_cash_flow": operating_cash_flow,
            "capital_expenditures": capital_expenditures,
            "free_cash_flow": free_cash_flow,
        })

    quarterly_earnings = earnings.get("quarterlyEarnings", [])
    if isinstance(quarterly_earnings, list):
        for report in quarterly_earnings:
            row = record(report)
            if row is None:
                continue
            row["eps_diluted"] = _number(report.get("reportedEPS"))
            row["reported_date"] = report.get("reportedDate")

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records.values())
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    reported = pd.to_datetime(frame.get("reported_date"), errors="coerce")
    ninety_days = pd.to_timedelta(np.full(len(frame), 90, dtype="int64"), unit="D")
    one_day = pd.to_timedelta(np.ones(len(frame), dtype="int64"), unit="D")
    estimated_filing = frame["period_end"] + ninety_days
    # Alpha Vantage exposes report dates but not a timestamp. Match the SEC
    # provider's conservative convention and make the data visible next day.
    frame["filed_at"] = reported.fillna(estimated_filing) + one_day
    frame["filed_date"] = reported.fillna(estimated_filing)
    frame["ticker"] = ticker.upper()
    frame["fiscal_year"] = frame["period_end"].dt.year
    frame["fiscal_period"] = "Q" + frame["period_end"].dt.quarter.astype("Int64").astype(str)
    frame["statement_type"] = "quarterly"
    frame["form"] = "ALPHA_VANTAGE"
    frame["accession"] = "AV:" + frame["period_end"].dt.strftime("%Y-%m-%d")
    frame["data_source"] = "Alpha Vantage"
    return frame.dropna(subset=["period_end"]).sort_values("period_end").reset_index(drop=True)


class AlphaVantageProvider:
    """Cached client for free-tier global quarterly fundamentals."""

    def __init__(self, cache_dir: Path, api_key: str, cache_days: int = 7) -> None:
        if not str(api_key).strip():
            raise ValueError("Alpha Vantage API key is missing")
        self.api_key = str(api_key).strip()
        self.cache_dir = Path(cache_dir) / "alpha_vantage"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_days = int(cache_days)
        self.minimum_interval = 1.1
        self.last_request = 0.0
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.75,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return datetime.now(UTC) - modified < timedelta(days=self.cache_days)

    def _request(self, function: str, query: str, *, symbol: bool = True) -> dict[str, Any]:
        safe_query = "".join(character if character.isalnum() else "_" for character in query.upper())
        path = self.cache_dir / f"{function.lower()}_{safe_query}.json"
        if self._fresh(path):
            return json.loads(path.read_text(encoding="utf-8"))
        parameters = {"function": function, "apikey": self.api_key}
        parameters["symbol" if symbol else "keywords"] = query
        payload: dict[str, Any] = {}
        for attempt in range(3):
            delay = self.minimum_interval - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            try:
                response = self.session.get(BASE_URL, params=parameters, timeout=30)
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Alpha Vantage request failed ({type(exc).__name__})"
                ) from None
            self.last_request = time.monotonic()
            if not response.ok:
                raise RuntimeError(f"Alpha Vantage returned HTTP {response.status_code}")
            try:
                payload = response.json()
            except requests.JSONDecodeError:
                raise RuntimeError("Alpha Vantage returned an invalid response") from None
            error = payload.get("Error Message") or payload.get("Note") or payload.get("Information")
            rate_limited = error and "request" in str(error).lower() and "limit" in str(error).lower()
            if rate_limited and attempt < 2:
                time.sleep(self.minimum_interval)
                continue
            if error:
                raise RuntimeError(f"Alpha Vantage: {error}")
            break
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(path)
        return payload

    def resolve_symbol(self, ticker: str) -> str:
        requested = ticker.strip().upper()
        base, _, suffix = requested.partition(".")
        payload = self._request("SYMBOL_SEARCH", base, symbol=False)
        matches = payload.get("bestMatches", [])
        if not isinstance(matches, list) or not matches:
            return requested
        region_hints = {
            "OL": ("NORWAY", "NOK"), "CO": ("DENMARK", "DKK"),
            "ST": ("SWEDEN", "SEK"), "HE": ("FINLAND", "EUR"),
            "L": ("UNITED KINGDOM", "GBP"), "DE": ("GERMANY", "EUR"),
            "PA": ("FRANCE", "EUR"), "AS": ("NETHERLANDS", "EUR"),
            "TO": ("CANADA", "CAD"), "AX": ("AUSTRALIA", "AUD"),
        }

        def rank(item: dict[str, Any]) -> tuple[float, float]:
            candidate = str(item.get("1. symbol", "")).upper()
            candidate_base = candidate.split(".", 1)[0]
            region = str(item.get("4. region", "")).upper()
            currency = str(item.get("8. currency", "")).upper()
            hint = region_hints.get(suffix)
            compatibility = float(candidate == requested) * 10.0
            if hint:
                # For exchange-qualified input, region/currency identity matters
                # more than sharing a base code with an unrelated US security.
                compatibility += float(candidate_base == base) * 2.0
                compatibility += float(hint[0] in region) * 8.0 + float(currency == hint[1]) * 6.0
            else:
                compatibility += float(candidate_base == base) * 4.0
            try:
                match_score = float(item.get("9. matchScore", 0))
            except (TypeError, ValueError):
                match_score = 0.0
            return compatibility, match_score

        best = max(matches, key=rank)
        resolved = str(best.get("1. symbol") or requested).upper()
        log("DATA", "Resolved %s to Alpha Vantage symbol %s", requested, resolved)
        return resolved

    def fundamentals(self, ticker: str) -> pd.DataFrame:
        symbol = self.resolve_symbol(ticker)
        log("DATA", "Downloading %s Alpha Vantage fundamentals...", symbol)
        income = self._request("INCOME_STATEMENT", symbol)
        if not _quarterly(income):
            raise ValueError(
                f"Alpha Vantage recognizes {ticker} as {symbol}, but provides no quarterly "
                "financial statements for it"
            )
        balance = self._request("BALANCE_SHEET", symbol)
        cash_flow = self._request("CASH_FLOW", symbol)
        earnings = self._request("EARNINGS", symbol)
        frame = normalize_alpha_vantage_fundamentals(
            ticker, income, balance, cash_flow, earnings
        )
        if frame.empty:
            raise ValueError(f"Alpha Vantage returned no quarterly fundamentals for {ticker}")
        log("DATA", "%d Alpha Vantage quarterly periods loaded for %s", len(frame), ticker.upper())
        return frame
