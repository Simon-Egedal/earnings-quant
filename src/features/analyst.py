from __future__ import annotations

import numpy as np
import pandas as pd

from .fundamentals import safe_divide


def historical_eps_features(
    events: pd.DataFrame, as_of: pd.Timestamp, statement_type: str = "quarterly"
) -> dict[str, float]:
    """Build split-consistent EPS features from Yahoo's reported EPS series.

    Yahoo adjusts its historical reported EPS and consensus to the same current
    share basis. SEC Company Facts values are point-in-time values and can be on
    a pre-split basis, so mixing the two creates false EPS surprises.
    """
    if events.empty or "actual_eps" not in events:
        return {}
    dates = pd.to_datetime(events["earnings_date"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    prior = events.loc[dates < cutoff].copy()
    if prior.empty:
        return {}
    prior["_event_date"] = dates.loc[prior.index]
    prior = prior.sort_values("_event_date").tail(8)
    values = pd.to_numeric(prior["actual_eps"], errors="coerce")

    def growth(offset: int) -> float:
        if len(values) <= offset:
            return np.nan
        return safe_divide(values.iloc[-1] - values.iloc[-1 - offset], abs(values.iloc[-1 - offset]))

    trailing_four = values.tail(4)
    ttm_eps = float(trailing_four.sum()) if len(trailing_four) == 4 and trailing_four.notna().all() else np.nan
    latest = float(values.iloc[-1]) if pd.notna(values.iloc[-1]) else np.nan
    recent = values.tail(4)
    trend = (
        float(np.polyfit(np.arange(4), recent.to_numpy(dtype=float), 1)[0])
        if len(recent) == 4 and recent.notna().all()
        else np.nan
    )
    changes = [
        safe_divide(values.iloc[index] - values.iloc[index - 1], abs(values.iloc[index - 1]))
        for index in range(max(1, len(values) - 2), len(values))
    ]
    acceleration = (
        changes[-1] - changes[-2]
        if len(changes) >= 2 and pd.notna(changes[-1]) and pd.notna(changes[-2])
        else np.nan
    )
    return {
        "eps_diluted_history_count": float(values.notna().sum()),
        "eps_diluted_qoq": growth(1),
        "eps_diluted_yoy": growth(4),
        "eps_diluted_trend_4q": trend,
        "eps_acceleration": acceleration,
        "lag_eps_diluted": ttm_eps if statement_type == "annual" else latest,
        "ttm_eps": ttm_eps,
    }


def aligned_actual_eps(
    events: pd.DataFrame, event_date: pd.Timestamp, statement_type: str = "quarterly"
) -> float:
    """Return reported EPS in the same split-adjusted units as Yahoo consensus."""
    if events.empty or "actual_eps" not in events:
        return np.nan
    dates = pd.to_datetime(events["earnings_date"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(event_date)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    through_event = events.loc[dates <= cutoff].copy()
    through_event["_event_date"] = dates.loc[through_event.index]
    values = pd.to_numeric(
        through_event.sort_values("_event_date").tail(4)["actual_eps"], errors="coerce"
    )
    if statement_type == "annual":
        return float(values.sum()) if len(values) == 4 and values.notna().all() else np.nan
    if values.empty or pd.isna(values.iloc[-1]):
        return np.nan
    return float(values.iloc[-1])


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
    valid_eps = pd.to_numeric(prior["eps_surprise"], errors="coerce").dropna()
    if valid_eps.empty:
        return {}
    output = {
        "average_eps_surprise_4": valid_eps.mean(),
        "earnings_beat_rate_4": (valid_eps > 0).mean(),
    }
    if "revenue_surprise" in prior:
        valid_revenue = pd.to_numeric(prior["revenue_surprise"], errors="coerce").dropna()
        if not valid_revenue.empty:
            output["average_revenue_surprise_4"] = valid_revenue.mean()
            output["revenue_beat_rate_4"] = (valid_revenue > 0).mean()
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
