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
    "total_debt": (
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "DebtAndCapitalLeaseObligations",
    ),
    "stockholders_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "shares_outstanding": ("EntityCommonStockSharesOutstanding",),
}

INSTANT_METRICS = {
    "cash", "total_assets", "current_assets", "current_liabilities", "long_term_debt",
    "short_term_debt", "total_debt", "stockholders_equity", "shares_outstanding",
}

ADDITIVE_METRICS = {
    "revenue", "cost_of_revenue", "gross_profit", "operating_income", "net_income",
    "eps_diluted", "operating_cash_flow", "capital_expenditures", "free_cash_flow",
}


def combine_total_debt(statements: pd.DataFrame) -> pd.DataFrame:
    """Use a true combined-debt fact, falling back to long plus short debt.

    Missing debt components remain missing; absence of an XBRL tag is not proof
    that a company has zero debt.
    """
    output = statements.copy()
    direct = pd.to_numeric(
        output.get("total_debt", pd.Series(np.nan, index=output.index)), errors="coerce"
    )
    components = [
        pd.to_numeric(output[column], errors="coerce")
        for column in ("long_term_debt", "short_term_debt")
        if column in output
    ]
    parts = (
        pd.concat(components, axis=1).sum(axis=1, min_count=1)
        if components
        else pd.Series(np.nan, index=output.index)
    )
    output["total_debt"] = direct.combine_first(parts)
    return output


def classify_statement_type(
    form: str, fiscal_period: str | None, duration_days: int | None, *, instant: bool = False
) -> str:
    """Classify an SEC fact without mixing quarter, YTD, and annual durations."""
    normalized_form = str(form).upper().removesuffix("/A")
    normalized_period = str(fiscal_period or "").upper()
    if instant:
        return "annual" if normalized_form == "10-K" or normalized_period == "FY" else "quarterly"
    if duration_days is None:
        return "other"
    if normalized_form == "10-Q":
        return "quarterly" if duration_days <= 150 else "year_to_date"
    if normalized_form == "10-K":
        if duration_days >= 250:
            return "annual"
        return "quarterly" if duration_days <= 150 else "year_to_date"
    if normalized_period == "FY" and duration_days >= 250:
        return "annual"
    return "quarterly" if duration_days <= 150 else "year_to_date"


def derive_fourth_quarters(statements: pd.DataFrame) -> pd.DataFrame:
    """Derive missing standalone Q4 values as annual less the first three quarters."""
    if statements.empty or "statement_type" not in statements:
        return statements
    derived: list[dict[str, Any]] = []
    annuals = statements.loc[statements["statement_type"].eq("annual")]
    for _, annual in annuals.iterrows():
        annual_start = pd.to_datetime(annual.get("period_start"), errors="coerce")
        annual_end = pd.to_datetime(annual.get("period_end"), errors="coerce")
        annual_filed = pd.to_datetime(annual.get("filed_at"), errors="coerce")
        if pd.isna(annual_start) or pd.isna(annual_end) or pd.isna(annual_filed):
            continue
        quarter_ends = pd.to_datetime(statements["period_end"], errors="coerce")
        quarter_filed = pd.to_datetime(statements["filed_at"], errors="coerce")
        quarters = statements.loc[
            statements["statement_type"].eq("quarterly")
            & quarter_ends.ge(annual_start) & quarter_ends.lt(annual_end)
            & quarter_filed.le(annual_filed)
        ].copy()
        if quarters.empty:
            continue
        quarters["_period_end"] = quarter_ends.loc[quarters.index]
        quarters["_filed_at"] = quarter_filed.loc[quarters.index]
        duration_columns = [metric for metric in ADDITIVE_METRICS if metric in quarters]
        quarters = quarters.loc[quarters[duration_columns].notna().any(axis=1)]
        selected_periods = sorted(quarters["_period_end"].dropna().unique())[-3:]
        if len(selected_periods) != 3:
            continue
        record = annual.to_dict()
        last_quarter_end = pd.Timestamp(selected_periods[-1])
        record.update({
            "period_start": last_quarter_end + pd.DateOffset(days=1),
            "duration_days": int((annual_end - last_quarter_end).days),
            "fiscal_period": "Q4",
            "form": "DERIVED-Q4",
            "accession": f"{annual.get('accession')}:Q4",
            "statement_type": "quarterly",
            "frame": None,
        })
        for metric in ADDITIVE_METRICS:
            if metric not in statements:
                continue
            annual_value = pd.to_numeric(pd.Series([annual.get(metric)]), errors="coerce").iloc[0]
            quarter_values: list[float] = []
            for period in selected_periods:
                observations = quarters.loc[quarters["_period_end"].eq(period)].sort_values("_filed_at")[metric]
                observations = pd.to_numeric(observations, errors="coerce").dropna()
                quarter_values.append(float(observations.iloc[-1]) if not observations.empty else np.nan)
            record[metric] = (
                float(annual_value - sum(quarter_values))
                if pd.notna(annual_value) and all(pd.notna(value) for value in quarter_values)
                else np.nan
            )
        if pd.notna(record.get("revenue", np.nan)) or pd.notna(record.get("eps_diluted", np.nan)):
            derived.append(record)
    if not derived:
        return statements
    return pd.concat([statements, pd.DataFrame(derived)], ignore_index=True)


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

    def fundamentals(self, ticker: str, refresh: bool = False) -> pd.DataFrame:
        """Collect quarterly, annual, and YTD SEC facts with explicit cadence metadata."""
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
                    duration: int | None = None
                    if metric not in INSTANT_METRICS:
                        if not start:
                            continue
                        duration = (pd.Timestamp(end) - pd.Timestamp(start)).days
                    statement_type = classify_statement_type(
                        form, item.get("fp"), duration, instant=metric in INSTANT_METRICS
                    )
                    rows.append({
                        "ticker": ticker.upper(), "metric": metric, "concept": alias,
                        "priority": priority, "period_end": end, "filed_at": item.get("filed"),
                        "filed_date": item.get("filed"),
                        "fiscal_year": item.get("fy"), "fiscal_period": item.get("fp"),
                        "form": form, "accession": item.get("accn"), "value": item["val"],
                        "statement_type": statement_type, "period_start": start,
                        "duration_days": duration, "frame": item.get("frame"),
                    })
        if not rows:
            return pd.DataFrame()
        long = pd.DataFrame(rows)
        long["period_end"] = pd.to_datetime(long["period_end"], errors="coerce")
        long["filed_date"] = pd.to_datetime(long["filed_date"], errors="coerce")
        # Company Facts exposes a date, not a dissemination timestamp. Treat the
        # fact as visible only after that calendar day to avoid same-day leakage.
        long["filed_at"] = long["filed_date"] + pd.DateOffset(days=1)
        long["period_start"] = pd.to_datetime(long["period_start"], errors="coerce")
        long = long.dropna(subset=["period_end", "filed_at", "accession"])
        duration_periods = long.loc[
            ~long["metric"].isin(INSTANT_METRICS),
            ["accession", "statement_type", "period_end"],
        ].drop_duplicates()
        for row_index in long.index[long["metric"].isin(INSTANT_METRICS)]:
            row = long.loc[row_index]
            candidates = duration_periods.loc[
                (duration_periods["accession"] == row["accession"])
                & (duration_periods["statement_type"] == row["statement_type"])
            ]
            if candidates.empty:
                continue
            distances = (candidates["period_end"] - row["period_end"]).dt.days.abs()
            if distances.min() <= 45:
                long.at[row_index, "period_end"] = candidates.loc[distances.idxmin(), "period_end"]
        long = long.sort_values(["priority", "filed_at"]).drop_duplicates(
            ["accession", "period_end", "statement_type", "metric"], keep="first"
        )
        index = [
            "ticker", "period_end", "filed_at", "fiscal_year", "fiscal_period",
            "form", "accession", "statement_type", "filed_date",
        ]
        wide = long.pivot_table(index=index, columns="metric", values="value", aggfunc="last").reset_index()
        wide.columns.name = None
        period_metadata = long.groupby(index, dropna=False).agg(
            period_start=("period_start", "min"),
            duration_days=("duration_days", "max"),
            frame=("frame", lambda values: next((value for value in values if pd.notna(value)), None)),
        ).reset_index()
        wide = wide.merge(period_metadata, on=index, how="left")
        wide = combine_total_debt(wide)
        if {"operating_cash_flow", "capital_expenditures"}.issubset(wide):
            wide["free_cash_flow"] = wide["operating_cash_flow"] - wide["capital_expenditures"].abs()
        wide = derive_fourth_quarters(wide)
        counts = wide["statement_type"].value_counts().to_dict()
        log("DATA", "%d SEC statement periods loaded for %s: %s", len(wide), ticker.upper(), counts)
        return wide.sort_values(["filed_at", "period_end"]).reset_index(drop=True)

    def quarterly_fundamentals(self, ticker: str, refresh: bool = False) -> pd.DataFrame:
        """Backward-compatible alias; returned rows now include all statement cadences."""
        return self.fundamentals(ticker, refresh)
