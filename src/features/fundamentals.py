from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(float(denominator)) < 1e-12:
        return np.nan
    return float(numerator) / float(denominator)


def _growth(values: pd.Series, periods: int) -> float:
    if len(values) <= periods:
        return np.nan
    return safe_divide(values.iloc[-1] - values.iloc[-1 - periods], abs(values.iloc[-1 - periods]))


def infer_next_statement_type(history: pd.DataFrame, as_of: pd.Timestamp) -> tuple[str, str]:
    """Infer the next filing cadence and fiscal period from the latest visible filing."""
    if history.empty:
        return "quarterly", "Q"
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    filed_at = pd.to_datetime(history["filed_at"], utc=True, errors="coerce")
    visible = history.loc[filed_at < cutoff].copy()
    if visible.empty:
        return "quarterly", "Q"
    visible["_filed_at"] = filed_at.loc[visible.index]
    forms = visible.get("form", pd.Series("", index=visible.index)).astype(str).str.upper()
    visible = visible.loc[forms.str.startswith(("10-Q", "10-K"))]
    if visible.empty:
        return "quarterly", "Q"
    filing_columns = [column for column in ("accession", "_filed_at", "form", "fiscal_period") if column in visible]
    filings = visible[filing_columns].drop_duplicates().sort_values("_filed_at")
    latest = filings.iloc[-1]
    fiscal_period = str(latest.get("fiscal_period", "")).upper()
    if fiscal_period == "Q3":
        return "annual", "FY"
    next_period = {"FY": "Q1", "Q1": "Q2", "Q2": "Q3"}.get(fiscal_period, "Q")
    return "quarterly", next_period


def point_in_time_fundamentals(
    history: pd.DataFrame, as_of: pd.Timestamp, statement_type: str = "quarterly"
) -> tuple[dict[str, object], pd.DataFrame]:
    """Build features from filings strictly earlier than ``as_of``.

    The returned snapshot is useful to audit exactly which source rows were visible.
    """
    if history.empty:
        return {}, history.copy()
    cutoff = pd.Timestamp(as_of)
    filed = pd.to_datetime(history["filed_at"], utc=True, errors="coerce")
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    cadence = history.get("statement_type")
    cadence_mask = cadence.eq(statement_type) if cadence is not None else pd.Series(True, index=history.index)
    visible = history.loc[(filed < cutoff) & cadence_mask].copy()
    if visible.empty:
        return {}, visible
    visible["filed_at"] = filed.loc[visible.index]
    visible["period_end"] = pd.to_datetime(visible["period_end"], errors="coerce")
    # An amendment may replace a previously visible filing, but future amendments cannot.
    # Later filings often repeat comparative facts sparsely. Consolidate each period
    # column-by-column so a later sparse filing does not erase previously public facts.
    consolidated: list[dict] = []
    metadata = {
        "ticker", "period_end", "filed_at", "fiscal_year", "fiscal_period", "form", "accession",
        "statement_type", "period_start", "duration_days", "frame", "filed_date",
    }
    value_columns = [column for column in visible.columns if column not in metadata]
    for period_end, group in visible.sort_values("filed_at").groupby("period_end"):
        record: dict[str, object] = {"period_end": period_end, "filed_at": group["filed_at"].max()}
        for column in value_columns:
            values = group[column].dropna()
            record[column] = values.iloc[-1] if not values.empty else np.nan
        consolidated.append(record)
    visible = pd.DataFrame(consolidated).sort_values("period_end")
    duration_metrics = [
        column for column in (
            "revenue", "cost_of_revenue", "gross_profit", "operating_income", "net_income",
            "eps_diluted", "operating_cash_flow", "capital_expenditures", "free_cash_flow",
        ) if column in visible
    ]
    if duration_metrics:
        statement_rows = visible[duration_metrics].notna().any(axis=1)
        if statement_rows.any():
            visible = visible.loc[statement_rows]
    visible = visible.tail(8)
    latest = visible.iloc[-1]
    periods_per_year = 1 if statement_type == "annual" else 4
    output: dict[str, object] = {
        "statement_type": statement_type,
        "statement_is_annual": float(statement_type == "annual"),
        "fundamental_asof_lag_days": float((cutoff.tz_localize(None) - visible["period_end"].iloc[-1]).days),
        "visible_filing_count": float(len(visible)),
    }
    for metric in ("revenue", "eps_diluted", "operating_cash_flow", "free_cash_flow", "total_assets", "shares_outstanding"):
        if metric in visible:
            values = pd.to_numeric(visible[metric], errors="coerce")
            output[f"{metric}_qoq"] = _growth(values, 1)
            output[f"{metric}_yoy"] = _growth(values, periods_per_year)
            if len(values.dropna()) >= 4:
                recent = values.tail(4).to_numpy(dtype=float)
                output[f"{metric}_trend_4q"] = float(np.polyfit(np.arange(4), recent, 1)[0]) if np.isfinite(recent).all() else np.nan
    revenue = latest.get("revenue", np.nan)
    equity = latest.get("stockholders_equity", np.nan)
    assets = latest.get("total_assets", np.nan)
    debt = latest.get("total_debt", np.nan)
    margins = {
        "gross_margin": safe_divide(latest.get("gross_profit", np.nan), revenue),
        "operating_margin": safe_divide(latest.get("operating_income", np.nan), revenue),
        "net_margin": safe_divide(latest.get("net_income", np.nan), revenue),
        "fcf_margin": safe_divide(latest.get("free_cash_flow", np.nan), revenue),
    }
    output.update(margins)
    output.update({
        "roa": safe_divide(latest.get("net_income", np.nan), assets),
        "roe": safe_divide(latest.get("net_income", np.nan), equity),
        "debt_to_equity": safe_divide(debt, equity),
        "debt_to_assets": safe_divide(debt, assets),
        "cash_to_debt": safe_divide(latest.get("cash", np.nan), debt),
        "current_ratio": safe_divide(latest.get("current_assets", np.nan), latest.get("current_liabilities", np.nan)),
        "asset_growth": _growth(pd.to_numeric(visible.get("total_assets", pd.Series(dtype=float)), errors="coerce"), 4),
        "share_dilution": _growth(pd.to_numeric(visible.get("shares_outstanding", pd.Series(dtype=float)), errors="coerce"), 4),
    })
    for metric in ("shares_outstanding", "stockholders_equity", "total_debt", "cash"):
        output[f"latest_{metric}"] = float(latest.get(metric, np.nan))
    trailing_periods = 1 if statement_type == "annual" else 4
    for metric in ("revenue", "net_income", "free_cash_flow"):
        values = pd.to_numeric(visible.get(metric, pd.Series(dtype=float)), errors="coerce")
        output[f"ttm_{metric}"] = float(values.tail(trailing_periods).sum(min_count=trailing_periods))
    eps_values = pd.to_numeric(visible.get("eps_diluted", pd.Series(dtype=float)), errors="coerce")
    output["ttm_eps"] = float(eps_values.tail(trailing_periods).sum(min_count=trailing_periods))
    for name, value in margins.items():
        numerator = {"gross_margin": "gross_profit", "operating_margin": "operating_income", "net_margin": "net_income", "fcf_margin": "free_cash_flow"}[name]
        series = pd.to_numeric(visible.get(numerator, pd.Series(dtype=float)), errors="coerce") / pd.to_numeric(visible.get("revenue", pd.Series(dtype=float)), errors="coerce")
        output[f"{name}_change_qoq"] = series.iloc[-1] - series.iloc[-2] if len(series) >= 2 else np.nan
        yoy_offset = periods_per_year + 1
        output[f"{name}_change_yoy"] = series.iloc[-1] - series.iloc[-yoy_offset] if len(series) >= yoy_offset else np.nan
    revenue_series = pd.to_numeric(visible.get("revenue", pd.Series(dtype=float)), errors="coerce")
    eps_series = pd.to_numeric(visible.get("eps_diluted", pd.Series(dtype=float)), errors="coerce")
    revenue_growth = revenue_series.pct_change(fill_method=None)
    eps_growth = eps_series.pct_change(fill_method=None)
    output["revenue_acceleration"] = revenue_growth.iloc[-1] - revenue_growth.iloc[-2] if len(revenue_growth) >= 3 else np.nan
    output["eps_acceleration"] = eps_growth.iloc[-1] - eps_growth.iloc[-2] if len(eps_growth) >= 3 else np.nan
    # Model A predictors may use lagged levels, never the future target quarter.
    for metric in ("revenue", "eps_diluted", "operating_margin", "free_cash_flow"):
        output[f"lag_{metric}"] = margins[metric] if metric == "operating_margin" else float(latest.get(metric, np.nan))
    return output, visible
