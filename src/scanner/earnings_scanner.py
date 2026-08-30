from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.analyst import historical_surprise_features, normalize_live_analyst_features
from src.features.fundamentals import point_in_time_fundamentals, safe_divide
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


def scan_upcoming(config: dict, days: int, top: int) -> pd.DataFrame:
    data_dir = Path(config["project"]["data_dir"])
    bundle_path = data_dir / "models" / "model_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("No trained models found; run collect, build-dataset, and train first")
    bundle: ModelBundle = joblib.load(bundle_path)
    yahoo = YahooFinanceProvider()
    sec = SECCompanyFactsProvider(data_dir / "cache", config["sec"]["user_agent"], config["sec"]["requests_per_second"], config["sec"]["cache_days"])
    calendar = yahoo.upcoming_earnings(days, config["universe"]["minimum_market_cap"])
    if calendar.empty:
        return calendar
    symbols = calendar["ticker"].astype(str).str.upper().unique().tolist()
    prices = yahoo.price_history([*symbols, config["collection"].get("price_symbol", "SPY")], start=(pd.Timestamp.now() - pd.Timedelta(days=550)).date().isoformat())
    history_path = data_dir / "raw" / "earnings.parquet"
    historical = pd.read_parquet(history_path) if history_path.exists() else pd.DataFrame()
    rows: list[dict] = []
    for _, event in calendar.iterrows():
        ticker = str(event["ticker"]).upper()
        try:
            fundamentals = sec.quarterly_fundamentals(ticker)
            analyst = yahoo.analyst_snapshot(ticker)
            metadata = yahoo.metadata(ticker)
            event_date = pd.Timestamp(event["earnings_date"])
            features, _ = point_in_time_fundamentals(fundamentals, event_date)
            if not features:
                log("WARN", "%s scan skipped: insufficient visible filings", ticker)
                continue
            row = event.to_dict() | metadata | features | normalize_live_analyst_features(analyst)
            row.update(market_features(prices, ticker, event_date, config["collection"].get("price_symbol", "SPY")))
            if not historical.empty:
                row.update(historical_surprise_features(historical[historical["ticker"] == ticker], event_date))
            price, shares = row.get("price_asof", np.nan), row.get("latest_shares_outstanding", np.nan)
            market_cap = price * shares if pd.notna(price) and pd.notna(shares) else metadata.get("market_cap", np.nan)
            enterprise_value = market_cap + row.get("latest_total_debt", 0) - row.get("latest_cash", 0)
            row.update({
                "market_cap_event": market_cap, "pe": safe_divide(price, row.get("ttm_eps", np.nan)),
                "forward_pe": safe_divide(price, row.get("consensus_eps", np.nan) * 4),
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
        "ticker", "company", "earnings_date", "timing", "sector", "consensus_eps", "predicted_eps",
        "predicted_eps_surprise", "consensus_revenue", "predicted_revenue", "predicted_revenue_surprise",
        "predicted_abnormal_return_3d", "probability_up", "confidence", "signal",
    ]
    return live.reindex(columns=columns).head(top).reset_index(drop=True)

