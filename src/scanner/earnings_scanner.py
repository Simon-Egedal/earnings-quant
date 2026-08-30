from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.analyst import historical_surprise_features, normalize_live_analyst_features
from src.features.fundamentals import infer_next_statement_type, point_in_time_fundamentals, safe_divide
from src.features.market import market_features
from src.logging_utils import log
from src.models.training import ModelBundle, add_financial_predictions
from src.providers import SECCompanyFactsProvider, YahooFinanceProvider


def _align(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = np.nan
    return output


def get_upcoming_calendar(config: dict, days: int) -> pd.DataFrame:
    """Return the same eligible Yahoo earnings calendar used by the scanner."""
    return YahooFinanceProvider().upcoming_earnings(days, config["universe"]["minimum_market_cap"])


def _add_growth_comparisons(live: pd.DataFrame) -> pd.DataFrame:
    """Add display-only growth comparisons without changing model inputs."""
    output = live.copy()
    comparisons = (
        ("eps", "prior_year_eps"),
        ("revenue", "prior_year_revenue"),
    )
    for metric, prior_column in comparisons:
        if prior_column not in output:
            continue
        prior = pd.to_numeric(output[prior_column], errors="coerce").abs().replace(0, np.nan)
        for source in ("predicted", "consensus"):
            value_column = f"{source}_{metric}"
            if value_column in output:
                values = pd.to_numeric(output[value_column], errors="coerce")
                output[f"{source}_{metric}_growth_yoy"] = (values - output[prior_column]) / prior
    return output


def _reported_comparison(
    history: pd.DataFrame, as_of: pd.Timestamp, metric: str, statement_type: str = "quarterly"
) -> tuple[float, float]:
    """Return latest and comparable prior-year reported values visible at ``as_of``."""
    if history.empty or metric not in history:
        return np.nan, np.nan
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    filed_at = pd.to_datetime(history["filed_at"], utc=True, errors="coerce")
    values = pd.to_numeric(history[metric], errors="coerce")
    periods = pd.to_datetime(history["period_end"], errors="coerce")
    cadence = history.get("statement_type")
    cadence_mask = cadence.eq(statement_type) if cadence is not None else pd.Series(True, index=history.index)
    visible = pd.DataFrame({
        "period_end": periods,
        "filed_at": filed_at,
        "value": values,
    }).loc[(filed_at < cutoff) & values.notna() & periods.notna() & cadence_mask]
    if visible.empty:
        return np.nan, np.nan
    visible = visible.sort_values("filed_at").drop_duplicates("period_end", keep="last").sort_values("period_end")
    latest = visible.iloc[-1]
    if statement_type == "annual":
        return float(latest["value"]), float(latest["value"])
    # The upcoming report follows the latest quarter; its prior-year comparison
    # is therefore approximately nine months before the latest reported period.
    comparison_period = latest["period_end"] - pd.DateOffset(months=9)
    distances = (visible["period_end"] - comparison_period).dt.days.abs()
    comparable = visible.loc[distances.idxmin()]
    if distances.min() > 75:
        return float(latest["value"]), np.nan
    return float(latest["value"]), float(comparable["value"])


def scan_events(config: dict, calendar: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """Score supplied upcoming events with the project's existing two-stage models."""
    data_dir = Path(config["project"]["data_dir"])
    bundle_path = data_dir / "models" / "model_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("No trained models found; run collect, build-dataset, and train first")
    bundle: ModelBundle = joblib.load(bundle_path)
    yahoo = YahooFinanceProvider()
    sec = SECCompanyFactsProvider(data_dir / "cache", config["sec"]["user_agent"], config["sec"]["requests_per_second"], config["sec"]["cache_days"])
    if calendar.empty:
        return pd.DataFrame()
    symbols = calendar["ticker"].astype(str).str.upper().unique().tolist()
    prices = yahoo.price_history([*symbols, config["collection"].get("price_symbol", "SPY")], start=(pd.Timestamp.now() - pd.Timedelta(days=550)).date().isoformat())
    history_path = data_dir / "raw" / "earnings.parquet"
    historical = pd.read_parquet(history_path) if history_path.exists() else pd.DataFrame()
    rows: list[dict] = []
    for _, event in calendar.iterrows():
        ticker = str(event["ticker"]).upper()
        try:
            fundamentals = sec.fundamentals(ticker)
            analyst = yahoo.analyst_snapshot(ticker)
            metadata = yahoo.metadata(ticker)
            event_date = pd.Timestamp(event["earnings_date"])
            statement_type, expected_period = infer_next_statement_type(fundamentals, event_date)
            features, _ = point_in_time_fundamentals(fundamentals, event_date, statement_type)
            if not features:
                log("WARN", "%s scan skipped: insufficient visible filings", ticker)
                continue
            row = event.to_dict() | metadata | features | normalize_live_analyst_features(analyst)
            row["statement_type"] = statement_type
            row["expected_fiscal_period"] = expected_period
            if statement_type == "annual":
                row["quarterly_consensus_eps"] = row.get("consensus_eps", np.nan)
                row["quarterly_consensus_revenue"] = row.get("consensus_revenue", np.nan)
                row["consensus_eps"] = row.get("annual_consensus_eps", np.nan)
                row["consensus_revenue"] = row.get("annual_consensus_revenue", np.nan)
            latest_eps, prior_eps = _reported_comparison(fundamentals, event_date, "eps_diluted", statement_type)
            latest_revenue, prior_revenue = _reported_comparison(fundamentals, event_date, "revenue", statement_type)
            row.update({
                "latest_reported_eps": latest_eps,
                "latest_reported_revenue": latest_revenue,
                "prior_year_eps": prior_eps,
                "prior_year_revenue": prior_revenue,
            })
            row.update(market_features(prices, ticker, event_date, config["collection"].get("price_symbol", "SPY")))
            if not historical.empty:
                row.update(historical_surprise_features(historical[historical["ticker"] == ticker], event_date))
            price, shares = row.get("price_asof", np.nan), row.get("latest_shares_outstanding", np.nan)
            market_cap = price * shares if pd.notna(price) and pd.notna(shares) else metadata.get("market_cap", np.nan)
            enterprise_value = market_cap + row.get("latest_total_debt", 0) - row.get("latest_cash", 0)
            row.update({
                "market_cap_event": market_cap, "pe": safe_divide(price, row.get("ttm_eps", np.nan)),
                "forward_pe": safe_divide(price, row.get("consensus_eps", np.nan) * (1 if statement_type == "annual" else 4)),
                "price_to_sales": safe_divide(market_cap, row.get("ttm_revenue", np.nan)),
                "ev_to_revenue": safe_divide(enterprise_value, row.get("ttm_revenue", np.nan)),
                "price_to_book": safe_divide(market_cap, row.get("latest_stockholders_equity", np.nan)),
                "fcf_yield": safe_divide(row.get("ttm_free_cash_flow", np.nan), market_cap),
                "event_year": event_date.year,
            })
            rows.append(row)
        except Exception as exc:
            log("WARN", "%s scan skipped: %s", ticker, exc)
    live = pd.DataFrame(rows)
    if live.empty:
        return live
    financial_columns = bundle.financial.numeric_features + bundle.financial.categorical_features
    live = add_financial_predictions(live, bundle.financial.predict(_align(live, financial_columns)))
    live = _add_growth_comparisons(live)
    reaction_columns = bundle.reaction.numeric_features + bundle.reaction.categorical_features
    live = live.join(bundle.reaction.predict(_align(live, reaction_columns)))
    probability = live["probability_up"]
    threshold_long = float(config["backtest"]["long_threshold"])
    threshold_short = float(config["backtest"]["short_threshold"])
    minimum = float(config["backtest"]["minimum_confidence"])
    live["signal"] = np.select([
        (live["predicted_abnormal_return_3d"] > threshold_long) & (probability >= minimum),
        (live["predicted_abnormal_return_3d"] < threshold_short) & ((1 - probability) >= minimum),
    ], ["LONG", "SHORT"], default="NO_TRADE")
    live["confidence_score"] = np.maximum(probability, 1 - probability)
    live["confidence"] = pd.cut(live["confidence_score"], [0, .6, .75, 1], labels=["LOW", "MEDIUM", "HIGH"], include_lowest=True)
    live = live.sort_values(["predicted_abnormal_return_3d", "confidence_score", "market_cap_event"], key=lambda series: series.abs() if series.name == "predicted_abnormal_return_3d" else series, ascending=False)
    columns = [
        "ticker", "company", "earnings_date", "timing", "statement_type", "expected_fiscal_period",
        "sector", "consensus_eps", "predicted_eps",
        "predicted_eps_surprise", "predicted_eps_growth_yoy", "consensus_eps_growth_yoy",
        "consensus_revenue", "predicted_revenue", "predicted_revenue_surprise",
        "predicted_revenue_growth_yoy", "consensus_revenue_growth_yoy",
        "predicted_operating_margin", "predicted_fcf",
        "predicted_abnormal_return_3d", "probability_up", "confidence", "signal",
    ]
    result = live.reindex(columns=columns)
    return result.head(top).reset_index(drop=True) if top is not None else result.reset_index(drop=True)


def scan_upcoming(
    config: dict, days: int, top: int, tickers: list[str] | None = None
) -> pd.DataFrame:
    """Load upcoming events and optionally restrict scoring to selected tickers."""
    calendar = get_upcoming_calendar(config, days)
    if tickers:
        selected = {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
        calendar = calendar.loc[calendar["ticker"].astype(str).str.upper().isin(selected)].copy()
    return scan_events(config, calendar, top)
