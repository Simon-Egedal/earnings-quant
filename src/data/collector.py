from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.universe import load_universe
from src.logging_utils import log
from src.providers import SECCompanyFactsProvider, YahooFinanceProvider
from src.storage import record_artifact, write_parquet_atomic


def _combine(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    useful = [frame for frame in frames if frame is not None and not frame.empty]
    if not useful:
        return pd.DataFrame()
    return pd.concat(useful, ignore_index=True).drop_duplicates(keys, keep="last")


def collect_data(
    config: dict, tickers: list[str] | None = None, limit: int | None = None, refresh: bool = False
) -> dict[str, int]:
    data_dir = Path(config["project"]["data_dir"])
    cache_dir = data_dir / "cache"
    universe = tickers or load_universe(config["universe"], cache_dir, refresh)
    universe = universe[:limit] if limit else universe
    yahoo = YahooFinanceProvider()
    sec = SECCompanyFactsProvider(
        cache_dir, config["sec"]["user_agent"], config["sec"]["requests_per_second"], config["sec"]["cache_days"]
    )
    fundamentals: list[pd.DataFrame] = []
    earnings: list[pd.DataFrame] = []
    metadata: list[dict] = []
    for index, ticker in enumerate(universe, 1):
        log("DATA", "%d/%d collecting %s", index, len(universe), ticker)
        try:
            frame = sec.quarterly_fundamentals(ticker, refresh)
            if not frame.empty:
                fundamentals.append(frame)
        except Exception as exc:
            log("WARN", "%s SEC data skipped: %s", ticker, exc)
        try:
            frame = yahoo.historical_earnings(ticker)
            if not frame.empty:
                earnings.append(frame)
        except Exception as exc:
            log("WARN", "%s earnings history skipped: %s", ticker, exc)
        try:
            metadata.append(yahoo.metadata(ticker))
        except Exception as exc:
            log("WARN", "%s metadata skipped: %s", ticker, exc)
    price_frames: list[pd.DataFrame] = []
    symbols = list(dict.fromkeys([*universe, config["collection"].get("price_symbol", "SPY")]))
    for start in range(0, len(symbols), 50):
        batch = symbols[start:start + 50]
        try:
            price_frames.append(yahoo.price_history(batch, config["collection"]["start_date"]))
        except Exception as exc:
            log("WARN", "Price batch beginning %s skipped: %s", batch[0], exc)
    outputs = {
        "fundamentals": _combine(fundamentals, ["ticker", "period_end", "filed_at", "accession"]),
        "earnings": _combine(earnings, ["ticker", "earnings_date"]),
        "prices": _combine(price_frames, ["ticker", "date"]),
        "metadata": pd.DataFrame(metadata).drop_duplicates("ticker", keep="last") if metadata else pd.DataFrame(),
    }
    counts: dict[str, int] = {}
    db_path = cache_dir / "metadata.sqlite"
    for name, frame in outputs.items():
        path = data_dir / "raw" / f"{name}.parquet"
        if not frame.empty:
            write_parquet_atomic(frame, path)
            record_artifact(db_path, f"raw:{name}", path, len(frame))
        counts[name] = len(frame)
    log("DATA", "Collection complete: %s", counts)
    return counts

