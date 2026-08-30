from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train_years: tuple[int, ...]
    validation_year: int
    test_year: int


def expanding_year_folds(first_train_year: int, first_test_year: int, final_test_year: int) -> list[WalkForwardFold]:
    folds: list[WalkForwardFold] = []
    for test_year in range(first_test_year, final_test_year + 1):
        validation = test_year - 1
        train = tuple(range(first_train_year, validation))
        if train:
            folds.append(WalkForwardFold(train, validation, test_year))
    return folds


def split_fold(frame: pd.DataFrame, fold: WalkForwardFold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame["event_year"].isin(fold.train_years)]
    validation = frame[frame["event_year"] == fold.validation_year]
    test = frame[frame["event_year"] == fold.test_year]
    if not train.empty and not validation.empty and train["earnings_date"].max() >= validation["earnings_date"].min():
        raise ValueError("Chronological overlap between train and validation")
    if not validation.empty and not test.empty and validation["earnings_date"].max() >= test["earnings_date"].min():
        raise ValueError("Chronological overlap between validation and test")
    return train, validation, test

