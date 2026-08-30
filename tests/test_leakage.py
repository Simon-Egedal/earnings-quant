from __future__ import annotations

import pandas as pd
import pytest

from src.features.builder import EventDatasetBuilder, assert_no_lookahead
from src.features.fundamentals import point_in_time_fundamentals


def test_future_filing_is_never_visible(quarterly_fundamentals):
    event = pd.Timestamp("2023-08-01", tz="UTC")
    future = quarterly_fundamentals.iloc[[-1]].copy()
    future["period_end"] = pd.Timestamp("2023-09-30")
    future["filed_at"] = pd.Timestamp("2023-11-05")
    future["revenue"] = 999999.0
    history = quarterly_fundamentals.copy()
    history.loc[len(history)] = future.iloc[0]
    features, visible = point_in_time_fundamentals(history, event)
    assert (pd.to_datetime(visible["filed_at"], utc=True) < event).all()
    assert features["lag_revenue"] != 999999.0


def test_dataset_records_and_audits_feature_provenance(quarterly_fundamentals, prices):
    earnings = pd.DataFrame({
        "ticker": ["TEST"], "earnings_date": [pd.Timestamp("2023-08-01", tz="UTC")],
        "actual_eps": [1.3], "consensus_eps": [1.2], "timing": ["Before Market Open"],
    })
    dataset = EventDatasetBuilder().build(quarterly_fundamentals, earnings, prices)
    assert len(dataset) == 1
    assert_no_lookahead(dataset)
    assert pd.Timestamp(dataset.iloc[0]["max_feature_filed_at"]) < dataset.iloc[0]["earnings_date"]


def test_audit_rejects_same_day_or_future_filing():
    bad = pd.DataFrame({
        "ticker": ["TEST"], "earnings_date": ["2024-01-01T12:00:00Z"],
        "max_feature_filed_at": ["2024-01-01T12:00:00Z"],
    })
    with pytest.raises(ValueError, match="Look-ahead"):
        assert_no_lookahead(bad)
