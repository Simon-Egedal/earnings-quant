from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.builder import EventDatasetBuilder, assert_no_lookahead
from src.logging_utils import log
from src.storage import record_artifact, write_parquet_atomic


def build_dataset(config: dict) -> pd.DataFrame:
    data_dir = Path(config["project"]["data_dir"])
    paths = {name: data_dir / "raw" / f"{name}.parquet" for name in ("fundamentals", "earnings", "prices")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run collect first; missing: {', '.join(missing)}")
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    event_dates = pd.to_datetime(frames["earnings"]["earnings_date"], utc=True, errors="coerce")
    frames["earnings"] = frames["earnings"].loc[event_dates < pd.Timestamp.now(tz="UTC")].copy()
    dataset = EventDatasetBuilder(config["collection"].get("price_symbol", "SPY")).build(
        frames["fundamentals"], frames["earnings"], frames["prices"]
    )
    assert_no_lookahead(dataset)
    path = data_dir / "processed" / "earnings_events.parquet"
    write_parquet_atomic(dataset, path)
    record_artifact(data_dir / "cache" / "metadata.sqlite", "processed:earnings_events", path, len(dataset))
    log("FEATURES", "Dataset saved to %s", path)
    return dataset
