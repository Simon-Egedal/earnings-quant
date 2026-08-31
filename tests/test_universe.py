from __future__ import annotations

import pandas as pd

from src.data import universe


def test_select_universe_is_stable_and_not_an_alphabetical_prefix() -> None:
    tickers = [f"T{index:03d}" for index in range(500)]

    first = universe.select_universe(tickers, 100, seed=42)
    second = universe.select_universe(tickers, 100, seed=42)

    assert first == second
    assert len(first) == 100
    assert len(set(first)) == 100
    assert first != tickers[:100]


def test_select_universe_keeps_collected_tickers_and_fills_to_target() -> None:
    tickers = [f"T{index:03d}" for index in range(500)]
    collected = ["T010", "T200", "T499"]

    selected = universe.select_universe(tickers, 100, seed=42, preferred=collected)

    assert len(selected) == 100
    assert set(collected).issubset(selected)


def test_undersized_cached_universe_is_refreshed(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "sp500_universe.parquet"
    pd.DataFrame({"ticker": ["A", "B"]}).to_parquet(cache, index=False)
    refreshed = [f"T{index:03d}" for index in range(120)]
    monkeypatch.setattr(universe, "_fetch_sp500", lambda: refreshed)

    result = universe.load_universe(
        {"tickers": [], "target_size": 100}, tmp_path, refresh=False
    )

    assert result == refreshed
    assert pd.read_parquet(cache)["ticker"].tolist() == refreshed


def test_large_cached_universe_is_reused(tmp_path, monkeypatch) -> None:
    cached = [f"T{index:03d}" for index in range(120)]
    pd.DataFrame({"ticker": cached}).to_parquet(
        tmp_path / "sp500_universe.parquet", index=False
    )

    def unexpected_fetch() -> list[str]:
        raise AssertionError("complete cache should be reused")

    monkeypatch.setattr(universe, "_fetch_sp500", unexpected_fetch)

    assert universe.load_universe(
        {"tickers": [], "target_size": 100}, tmp_path, refresh=False
    ) == cached
