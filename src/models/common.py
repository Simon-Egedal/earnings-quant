from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


IDENTIFIERS = {
    "ticker", "company", "earnings_date", "timing", "fiscal_period", "target_filed_at",
    "max_feature_filed_at", "accession", "form",
}
LEAKAGE_PREFIXES = ("actual_", "abnormal_return_", "return_1d_target", "return_3d_target", "return_5d_target")
LEAKAGE_COLUMNS = {"eps_surprise", "eps_surprise_pct", "revenue_surprise", "reported_eps"}


def replace_infinite(values):
    """Treat infinite numeric features as missing before sklearn validation."""
    if isinstance(values, pd.DataFrame):
        return values.replace([np.inf, -np.inf], np.nan)
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array), array, np.nan)


def available_features(frame: pd.DataFrame, extra_exclude: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    excluded = IDENTIFIERS | LEAKAGE_COLUMNS | set(extra_exclude)
    candidates = [
        column for column in frame.columns
        if column not in excluded and not any(column.startswith(prefix) for prefix in LEAKAGE_PREFIXES)
    ]
    categorical = [column for column in ("sector", "industry", "statement_type") if column in candidates]
    numeric = [column for column in candidates if column not in categorical and pd.api.types.is_numeric_dtype(frame[column])]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("finite", FunctionTransformer(replace_infinite, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = [("numeric", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def finite_target(frame: pd.DataFrame, target: str) -> pd.Series:
    return pd.to_numeric(frame[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
