from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src.logging_utils import log
from .analyst import historical_surprise_features
from .fundamentals import point_in_time_fundamentals, safe_divide
from .market import event_returns, market_features


class EventDatasetBuilder:
    """Construct one point-in-time row per earnings event."""

    def __init__(self, benchmark: str = "SPY") -> None:
        self.benchmark = benchmark

    @staticmethod
    def _future_report(fundamentals: pd.DataFrame, event_date: pd.Timestamp) -> pd.Series | None:
        filed = pd.to_datetime(fundamentals["filed_at"], utc=True, errors="coerce")
        cutoff = pd.Timestamp(event_date)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        period_end = pd.to_datetime(fundamentals["period_end"], utc=True, errors="coerce")
        candidates = fundamentals.loc[
            (filed >= cutoff) & (filed <= cutoff + timedelta(days=90))
            & (period_end >= cutoff - timedelta(days=150)) & (period_end <= cutoff + timedelta(days=15))
        ].copy()
        if candidates.empty:
            return None
        candidates["_period_end"] = period_end.loc[candidates.index]
        latest_period = candidates["_period_end"].max()
        return candidates.loc[candidates["_period_end"] == latest_period].sort_values("filed_at").iloc[0]

    def build(self, fundamentals: pd.DataFrame, earnings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for ticker, company_events in earnings.groupby("ticker"):
            company_fundamentals = fundamentals.loc[fundamentals["ticker"] == ticker]
            for _, event in company_events.sort_values("earnings_date").iterrows():
                event_date = pd.Timestamp(event["earnings_date"])
                features, visible = point_in_time_fundamentals(company_fundamentals, event_date)
                if not features:
                    continue
                row = event.to_dict()
                row.update(features)
                row.update(historical_surprise_features(company_events, event_date))
                row.update(market_features(prices, ticker, event_date, self.benchmark))
                row.update(event_returns(prices, ticker, event_date, self.benchmark, str(event.get("timing", ""))))
                price = row.get("price_asof", np.nan)
                shares = row.get("latest_shares_outstanding", np.nan)
                market_cap = price * shares if pd.notna(price) and pd.notna(shares) else np.nan
                enterprise_value = market_cap + row.get("latest_total_debt", 0) - row.get("latest_cash", 0)
                row.update({
                    "market_cap_event": market_cap,
                    "pe": safe_divide(price, row.get("ttm_eps", np.nan)),
                    "forward_pe": safe_divide(price, event.get("consensus_eps", np.nan) * 4),
                    "price_to_sales": safe_divide(market_cap, row.get("ttm_revenue", np.nan)),
                    "ev_to_revenue": safe_divide(enterprise_value, row.get("ttm_revenue", np.nan)),
                    "price_to_book": safe_divide(market_cap, row.get("latest_stockholders_equity", np.nan)),
                    "fcf_yield": safe_divide(row.get("ttm_free_cash_flow", np.nan), market_cap),
                })
                report = self._future_report(company_fundamentals, event_date)
                if report is not None:
                    row["actual_revenue"] = report.get("revenue", np.nan)
                    row["actual_operating_margin"] = safe_divide(report.get("operating_income", np.nan), report.get("revenue", np.nan))
                    row["actual_fcf"] = report.get("free_cash_flow", np.nan)
                    row["target_filed_at"] = report.get("filed_at")
                    if pd.isna(row.get("actual_eps", np.nan)):
                        row["actual_eps"] = report.get("eps_diluted", np.nan)
                row["max_feature_filed_at"] = visible["filed_at"].max()
                row["event_year"] = event_date.year
                rows.append(row)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["eps_surprise"] = [safe_divide(a - e, abs(e)) for a, e in zip(frame.get("actual_eps", np.nan), frame.get("consensus_eps", np.nan))]
            if {"actual_revenue", "consensus_revenue"}.issubset(frame):
                frame["revenue_surprise"] = [safe_divide(a - e, abs(e)) for a, e in zip(frame["actual_revenue"], frame["consensus_revenue"])]
            for column in ("pe", "forward_pe", "price_to_sales", "ev_to_revenue", "price_to_book", "fcf_yield"):
                if column in frame:
                    expanding_mean = frame.groupby("ticker")[column].transform(lambda values: values.shift().expanding(4).mean())
                    expanding_std = frame.groupby("ticker")[column].transform(lambda values: values.shift().expanding(4).std())
                    frame[f"{column}_zscore"] = (frame[column] - expanding_mean) / expanding_std.replace(0, np.nan)
            log("FEATURES", "Built %d leakage-audited earnings-event rows", len(frame))
        return frame


def assert_no_lookahead(dataset: pd.DataFrame) -> None:
    if dataset.empty:
        return
    event = pd.to_datetime(dataset["earnings_date"], utc=True, errors="coerce")
    filed = pd.to_datetime(dataset["max_feature_filed_at"], utc=True, errors="coerce")
    violations = dataset.loc[filed >= event]
    if not violations.empty:
        samples = violations[["ticker", "earnings_date", "max_feature_filed_at"]].head().to_dict("records")
        raise ValueError(f"Look-ahead detected in {len(violations)} rows: {samples}")
