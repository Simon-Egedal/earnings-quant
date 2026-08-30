from __future__ import annotations

import numpy as np
import pandas as pd

from .fundamentals import safe_divide


def historical_surprise_features(events: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    if events.empty:
        return {}
    dates = pd.to_datetime(events["earnings_date"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    prior = events.loc[dates < cutoff].sort_values("earnings_date").tail(4).copy()
    if prior.empty:
        return {}
    if "eps_surprise" not in prior and {"actual_eps", "consensus_eps"}.issubset(prior):
        prior["eps_surprise"] = [
            safe_divide(a - e, abs(e)) for a, e in zip(prior["actual_eps"], prior["consensus_eps"])
        ]
    if "eps_surprise" not in prior:
        return {}
    output = {
        "average_eps_surprise_4": prior["eps_surprise"].mean(),
        "earnings_beat_rate_4": (prior["eps_surprise"] > 0).mean(),
    }
    if "revenue_surprise" in prior:
        output["average_revenue_surprise_4"] = prior["revenue_surprise"].mean()
        output["revenue_beat_rate_4"] = (prior["revenue_surprise"] > 0).mean()
    return {key: float(value) for key, value in output.items()}


def normalize_live_analyst_features(snapshot: dict) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in snapshot.items():
        if key == "ticker":
            continue
        try:
            output[key] = float(value)
        except (TypeError, ValueError):
            continue
    current = output.get("eps_current")
    for days in (7, 30, 60, 90):
        old = output.get(f"eps_{days}daysago")
        output[f"eps_estimate_change_{days}d"] = safe_divide(current - old, abs(old))
    return output
