from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging_utils import log
from .metrics import grouped_metrics


def run_backtest(predictions: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, dict]]:
    settings = config["backtest"]
    long_threshold = float(settings["long_threshold"])
    short_threshold = float(settings["short_threshold"])
    minimum_confidence = float(settings["minimum_confidence"])
    frame = predictions.copy().sort_values("earnings_date")
    predicted = pd.to_numeric(frame["predicted_abnormal_return_3d"], errors="coerce")
    probability = pd.to_numeric(frame["probability_up"], errors="coerce")
    long = (predicted > long_threshold) & (probability >= minimum_confidence)
    short = (predicted < short_threshold) & ((1 - probability) >= minimum_confidence)
    frame["signal"] = np.select([long, short], ["LONG", "SHORT"], default="NO_TRADE")
    frame["confidence"] = np.where(frame["signal"] == "LONG", probability, np.where(frame["signal"] == "SHORT", 1 - probability, np.maximum(probability, 1 - probability)))
    frame["confidence_group"] = pd.cut(frame["confidence"], [0, 0.6, 0.75, 1], labels=["LOW", "MEDIUM", "HIGH"], include_lowest=True)
    if "market_cap_event" in frame:
        frame["market_cap_group"] = pd.cut(frame["market_cap_event"], [0, 10e9, 50e9, np.inf], labels=["$2-10B", "$10-50B", "$50B+"])
    side = frame["signal"].map({"LONG": 1.0, "SHORT": -1.0, "NO_TRADE": 0.0})
    costs = np.where(side != 0, float(settings["commission"]) + 2 * float(settings["slippage"]), 0.0)
    frame["strategy_return"] = side * pd.to_numeric(frame["abnormal_return_3d"], errors="coerce") - costs
    frame.loc[frame["signal"] == "NO_TRADE", "strategy_return"] = 0.0
    frame["cumulative_return"] = (1 + frame["strategy_return"].fillna(0)).cumprod() - 1
    log("BACKTEST", "Generated %d signals from %d events", int((frame["signal"] != "NO_TRADE").sum()), len(frame))
    return frame, grouped_metrics(frame)

