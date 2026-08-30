from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.analyst import historical_eps_features, historical_surprise_features, normalize_live_analyst_features
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


def _prediction_quality_reasons(
    row: pd.Series, training_ticker_count: int, scanner_config: dict
) -> list[str]:
    """Return reasons a forecast must not be converted into a trading signal."""
    reasons: list[str] = []
    minimum_tickers = int(scanner_config.get("minimum_training_tickers", 20))
    if training_ticker_count < minimum_tickers:
        reasons.append(f"model trained on only {training_ticker_count} tickers (minimum {minimum_tickers})")
    statement_type = str(row.get("statement_type", "quarterly"))
    minimum_history = int(scanner_config.get(
        "minimum_annual_history" if statement_type == "annual" else "minimum_quarterly_history",
        2 if statement_type == "annual" else 4,
    ))
    for metric, label in (("revenue", "revenue"), ("eps_diluted", "EPS")):
        count = pd.to_numeric(pd.Series([row.get(f"{metric}_history_count")]), errors="coerce").iloc[0]
        if pd.isna(count) or count < minimum_history:
            reasons.append(f"only {int(count) if pd.notna(count) else 0} usable {label} periods")
    predicted_revenue = pd.to_numeric(pd.Series([row.get("predicted_revenue")]), errors="coerce").iloc[0]
    if pd.isna(predicted_revenue) or predicted_revenue <= 0:
        reasons.append("revenue forecast is missing or non-positive")
    else:
        maximum_ratio = float(scanner_config.get("maximum_revenue_anchor_ratio", 5.0))
        for column, label in (("lag_revenue", "latest reported revenue"), ("consensus_revenue", "analyst consensus")):
            anchor = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(anchor) and anchor > 0:
                ratio = predicted_revenue / anchor
                if ratio > maximum_ratio or ratio < 1 / maximum_ratio:
                    reasons.append(f"revenue forecast is {ratio:.1f}x {label}")
    predicted_margin = pd.to_numeric(pd.Series([row.get("predicted_operating_margin")]), errors="coerce").iloc[0]
    if pd.notna(predicted_margin) and not -1 <= predicted_margin <= 1:
        reasons.append("operating-margin forecast is outside -100% to +100%")
    predicted_fcf = pd.to_numeric(pd.Series([row.get("predicted_fcf")]), errors="coerce").iloc[0]
    if pd.notna(predicted_fcf) and pd.notna(predicted_revenue) and abs(predicted_fcf) > 2 * predicted_revenue:
        reasons.append("free-cash-flow forecast is more than twice revenue")
    return reasons


def _model_quality_reasons(model_dir: Path, scanner_config: dict) -> list[str]:
    """Reject signals when saved walk-forward evidence does not beat baselines."""
    path = model_dir / "holdout_metrics.json"
    if not path.exists():
        return ["model has not been walk-forward evaluated; run evaluate"]
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        overall = metrics["walk_forward_overall"]
        auc = float(overall["reaction_classification"]["roc_auc"])
        r2 = float(overall["reaction_regression"]["r2"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ["saved walk-forward metrics are missing or invalid; run evaluate"]
    reasons: list[str] = []
    minimum_auc = float(scanner_config.get("minimum_walk_forward_auc", 0.52))
    minimum_r2 = float(scanner_config.get("minimum_walk_forward_r2", 0.0))
    if not np.isfinite(auc) or auc < minimum_auc:
        reasons.append(f"walk-forward AUC {auc:.3f} is below required {minimum_auc:.3f}")
    if not np.isfinite(r2) or r2 < minimum_r2:
        reasons.append(f"walk-forward R2 {r2:.3f} is below required {minimum_r2:.3f}")
    return reasons


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
            if not historical.empty:
                company_history = historical[historical["ticker"] == ticker]
                row.update(historical_eps_features(company_history, event_date, statement_type))
            latest_eps, prior_eps = _reported_comparison(fundamentals, event_date, "eps_diluted", statement_type)
            latest_revenue, prior_revenue = _reported_comparison(fundamentals, event_date, "revenue", statement_type)
            row.update({
                "latest_reported_eps": latest_eps,
                "latest_reported_revenue": latest_revenue,
                "prior_year_eps": prior_eps,
                "prior_year_revenue": prior_revenue,
            })
            row.update(market_features(
                prices,
                ticker,
                event_date,
                config["collection"].get("price_symbol", "SPY"),
                str(event.get("timing", "")),
            ))
            if not historical.empty:
                row.update(historical_surprise_features(company_history, event_date))
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
    training_ticker_count = len(getattr(bundle, "training_tickers", ()))
    scanner_config = config.get("scanner", {})
    model_quality_reasons = _model_quality_reasons(data_dir / "models", scanner_config)
    quality_reasons = [
        model_quality_reasons + _prediction_quality_reasons(
            row, training_ticker_count, scanner_config
        )
        for _, row in live.iterrows()
    ]
    live["data_quality"] = ["OK" if not reasons else "INSUFFICIENT_DATA" for reasons in quality_reasons]
    live["quality_reason"] = ["; ".join(reasons) for reasons in quality_reasons]
    reaction_columns = bundle.reaction.numeric_features + bundle.reaction.categorical_features
    live = live.join(bundle.reaction.predict(_align(live, reaction_columns)))
    probability = live["probability_up"]
    threshold_long = float(config["backtest"]["long_threshold"])
    threshold_short = float(config["backtest"]["short_threshold"])
    minimum = float(config["backtest"]["minimum_confidence"])
    quality_ok = live["data_quality"].eq("OK")
    live["signal"] = np.select([
        quality_ok & (live["predicted_abnormal_return_3d"] > threshold_long) & (probability >= minimum),
        quality_ok & (live["predicted_abnormal_return_3d"] < threshold_short) & ((1 - probability) >= minimum),
    ], ["LONG", "SHORT"], default="NO_TRADE")
    live.loc[~quality_ok, "signal"] = "INSUFFICIENT_DATA"
    live["confidence_score"] = np.maximum(probability, 1 - probability)
    live["confidence"] = pd.cut(live["confidence_score"], [0, .6, .75, 1], labels=["LOW", "MEDIUM", "HIGH"], include_lowest=True)
    live["confidence"] = live["confidence"].astype(object)
    live.loc[~quality_ok, "confidence"] = "N/A"
    live.loc[~quality_ok, ["predicted_abnormal_return_3d", "probability_up"]] = np.nan
    live = live.sort_values(["predicted_abnormal_return_3d", "confidence_score", "market_cap_event"], key=lambda series: series.abs() if series.name == "predicted_abnormal_return_3d" else series, ascending=False)
    columns = [
        "ticker", "company", "earnings_date", "timing", "statement_type", "expected_fiscal_period",
        "sector", "consensus_eps", "predicted_eps",
        "predicted_eps_surprise", "predicted_eps_growth_yoy", "consensus_eps_growth_yoy",
        "consensus_revenue", "predicted_revenue", "predicted_revenue_surprise",
        "predicted_revenue_growth_yoy", "consensus_revenue_growth_yoy",
        "lag_eps_diluted", "lag_revenue", "lag_operating_margin", "lag_free_cash_flow",
        "predicted_operating_margin", "predicted_fcf",
        "predicted_abnormal_return_3d", "probability_up", "confidence", "signal",
        "data_quality", "quality_reason",
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
