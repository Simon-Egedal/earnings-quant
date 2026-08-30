from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd


NEW_YORK = ZoneInfo("America/New_York")


def event_timing(value: object, label: object = "") -> str:
    """Normalize an earnings event to before-, during-, or after-market timing.

    Yahoo's historical endpoint usually encodes timing in the timestamp rather
    than in a separate field. Midnight timestamps are treated as date-only and
    therefore unknown instead of being shifted to the prior New York date.
    """
    text = "" if label is None else str(label).strip().lower()
    if text in {"nan", "<na>", "none"}:
        text = ""
    if any(token in text for token in ("after", "amc", "post-market", "post market")):
        return "after_market"
    if any(token in text for token in ("before", "bmo", "pre-market", "pre market")):
        return "before_market"

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp) or (timestamp.hour == 0 and timestamp.minute == 0):
        return "unknown"
    local = timestamp.tz_convert(NEW_YORK)
    minutes = local.hour * 60 + local.minute
    if minutes >= 16 * 60:
        return "after_market"
    if minutes < 9 * 60 + 30:
        return "before_market"
    return "during_market"


def event_session_date(value: object) -> object:
    """Return the US trading-session date represented by an event timestamp."""
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    if timestamp.hour == 0 and timestamp.minute == 0:
        return timestamp.date()
    return timestamp.tz_convert(NEW_YORK).date()
