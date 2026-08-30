from __future__ import annotations

import json
import re
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


COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Ordered aliases: the first matching concept wins when a filing reports aliases.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditures": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
    ),
    "cash": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "total_assets": ("Assets",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "short_term_debt": ("ShortTermBorrowings", "LongTermDebtCurrent"),
    "total_debt": ("LongTermDebtAndFinanceLeaseObligationsCurrent", "DebtCurrent"),
    "stockholders_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "shares_outstanding": ("EntityCommonStockSharesOutstanding",),
}

INSTANT_METRICS = {
    "cash", "total_assets", "current_assets", "current_liabilities", "long_term_debt",
    "short_term_debt", "total_debt", "stockholders_equity", "shares_outstanding",
}


class SECCompanyFactsProvider:
    """Rate-limited SEC Company Facts client with disk caching."""

    def __init__(self, cache_dir: Path, user_agent: str, requests_per_second: float = 5, cache_days: int = 7):
        if "@" not in user_agent:
            raise ValueError("SEC user_agent must identify the requester and include a contact email")
        self.cache_dir = Path(cache_dir) / "sec"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_days = cache_days
        self.minimum_interval = 1.0 / max(float(requests_per_second), 0.1)
        self.last_request = 0.0
        self.session = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Accept": "application/json"})

    def _fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return datetime.now(UTC) - modified < timedelta(days=self.cache_days)

    def _get_json(self, url: str, cache_path: Path, refresh: bool = False) -> dict[str, Any]:
        if not refresh and self._fresh(cache_path):
            return json.loads(cache_path.read_text(encoding="utf-8"))
        delay = self.minimum_interval - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)
        response = self.session.get(url, timeout=30)
        self.last_request = time.monotonic()
        response.raise_for_status()
        payload = response.json()
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(cache_path)
        return payload

    def ticker_map(self, refresh: bool = False) -> dict[str, int]:
        payload = self._get_json(COMPANY_TICKERS_URL, self.cache_dir / "company_tickers.json", refresh)
        return {str(item["ticker"]).upper(): int(item["cik_str"]) for item in payload.values()}

    def company_facts(self, ticker: str, refresh: bool = False) -> dict[str, Any]:
        symbol = ticker.upper().replace("-", ".")
        cik = self.ticker_map(refresh=False).get(symbol)
        if cik is None:
            raise KeyError(f"No SEC CIK found for {ticker}")
        log("DATA", "Downloading %s SEC fundamentals...", ticker.upper())
        return self._get_json(
            COMPANY_FACTS_URL.format(cik=cik), self.cache_dir / f"CIK{cik:010d}.json", refresh
        )

    @staticmethod
    def _unit_entries(concept: dict[str, Any]) -> list[dict[str, Any]]:
        units = concept.get("units", {})
        for preferred in ("USD", "USD/shares", "shares", "pure"):
            if preferred in units:
                return units[preferred]
        return next(iter(units.values()), [])

    def quarterly_fundamentals(self, ticker: str, refresh: bool = False) -> pd.DataFrame:
        payload = self.company_facts(ticker, refresh)
        gaap = payload.get("facts", {}).get("us-gaap", {})
        dei = payload.get("facts", {}).get("dei", {})
        rows: list[dict[str, Any]] = []
        for metric, aliases in CONCEPTS.items():
            for priority, alias in enumerate(aliases):
                concept = gaap.get(alias) or dei.get(alias)
                if not concept:
                    continue
                for item in self._unit_entries(concept):
                    form = str(item.get("form", ""))
                    if form not in {"10-Q", "10-Q/A", "10-K", "10-K/A"}:
                        continue
                    start, end = item.get("start"), item.get("end")
                    if not end or item.get("val") is None:
                        continue
                    if metric not in INSTANT_METRICS:
                        if not start:
                            continue
                        duration = (pd.Timestamp(end) - pd.Timestamp(start)).days
                        if duration > 150:  # Excludes year-to-date and annual duration facts.
                            continue
                    rows.append({
                        "ticker": ticker.upper(), "metric": metric, "concept": alias,
                        "priority": priority, "period_end": end, "filed_at": item.get("filed"),
                        "fiscal_year": item.get("fy"), "fiscal_period": item.get("fp"),
                        "form": form, "accession": item.get("accn"), "value": item["val"],
                    })
        if not rows:
            return pd.DataFrame()
        long = pd.DataFrame(rows)
        long["period_end"] = pd.to_datetime(long["period_end"], errors="coerce")
        long["filed_at"] = pd.to_datetime(long["filed_at"], errors="coerce")
        long = long.dropna(subset=["period_end", "filed_at", "accession"])
        long = long.sort_values(["priority", "filed_at"]).drop_duplicates(
            ["accession", "period_end", "metric"], keep="first"
        )
        index = ["ticker", "period_end", "filed_at", "fiscal_year", "fiscal_period", "form", "accession"]
        wide = long.pivot_table(index=index, columns="metric", values="value", aggfunc="last").reset_index()
        wide.columns.name = None
        if "total_debt" not in wide:
            wide["total_debt"] = np.nan
        long_debt = wide["long_term_debt"].fillna(0) if "long_term_debt" in wide else pd.Series(0.0, index=wide.index)
        short_debt = wide["short_term_debt"].fillna(0) if "short_term_debt" in wide else pd.Series(0.0, index=wide.index)
        debt_parts = long_debt + short_debt
        wide["total_debt"] = wide["total_debt"].fillna(debt_parts)
        if {"operating_cash_flow", "capital_expenditures"}.issubset(wide):
            wide["free_cash_flow"] = wide["operating_cash_flow"] - wide["capital_expenditures"].abs()
        log("DATA", "%d quarterly SEC reports loaded for %s", len(wide), ticker.upper())
        return wide.sort_values(["filed_at", "period_end"]).reset_index(drop=True)
