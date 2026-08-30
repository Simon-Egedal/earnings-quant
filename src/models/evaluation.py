from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    mask = pd.to_numeric(actual, errors="coerce").notna() & pd.to_numeric(predicted, errors="coerce").notna()
    y, p = actual.loc[mask].astype(float), predicted.loc[mask].astype(float)
    if y.empty:
        return {"count": 0}
    return {
        "count": int(len(y)), "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
    }


def classification_metrics(actual_return: pd.Series, probability_up: pd.Series) -> dict[str, float]:
    mask = actual_return.notna() & probability_up.notna()
    y, p = (actual_return.loc[mask] > 0).astype(int), probability_up.loc[mask]
    if y.empty:
        return {"count": 0}
    return {
        "count": int(len(y)), "accuracy": float(accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
    }

