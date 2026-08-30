from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_charts(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    ordered = frame.sort_values("earnings_date").copy()
    equity = (1 + ordered["strategy_return"].fillna(0)).cumprod()
    plt.figure(figsize=(9, 4)); plt.plot(pd.to_datetime(ordered["earnings_date"]), equity); plt.title("Cumulative event-study return"); plt.ylabel("Growth of $1"); save("cumulative_returns.png")
    plt.figure(figsize=(9, 4)); plt.fill_between(pd.to_datetime(ordered["earnings_date"]), equity / equity.cummax() - 1, 0); plt.title("Drawdown"); save("drawdown.png")
    plt.figure(figsize=(6, 6)); plt.scatter(ordered["predicted_abnormal_return_3d"], ordered["abnormal_return_3d"], alpha=.5); plt.axhline(0, color="grey"); plt.axvline(0, color="grey"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Prediction vs actual abnormal return"); save("prediction_vs_actual.png")
    hit = (np.sign(ordered["predicted_abnormal_return_3d"]) == np.sign(ordered["abnormal_return_3d"])).astype(float).rolling(50, min_periods=10).mean()
    plt.figure(figsize=(9, 4)); plt.plot(pd.to_datetime(ordered["earnings_date"]), hit); plt.ylim(0, 1); plt.title("Rolling 50-event directional hit rate"); save("rolling_hit_rate.png")
    unique_predictions = ordered["predicted_abnormal_return_3d"].dropna().nunique()
    if unique_predictions >= 2:
        buckets = pd.qcut(ordered["predicted_abnormal_return_3d"], q=min(10, unique_predictions), duplicates="drop")
        bucket_results = ordered.groupby(buckets, observed=True)["abnormal_return_3d"].mean()
        plt.figure(figsize=(9, 4)); bucket_results.plot(kind="bar"); plt.title("Actual return by predicted-return bucket"); plt.xticks(rotation=45, ha="right"); save("return_buckets.png")
    return paths


def generate_feature_importance(model_bundle: Any, output_dir: Path, top: int = 25) -> Path | None:
    """Save reaction-model importance when the selected estimator exposes it."""
    pipeline = model_bundle.reaction.regression_model
    if pipeline is None:
        return None
    estimator = pipeline.named_steps["model"]
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        coefficients = getattr(estimator, "coef_", None)
        values = np.abs(np.asarray(coefficients)).ravel() if coefficients is not None else None
    if values is None:
        return None
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pd.Series(values, index=names).nlargest(top).sort_values()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "feature_importance.png"
    plt.figure(figsize=(8, max(4, len(importance) * .25)))
    importance.plot(kind="barh")
    plt.title("Reaction model feature importance")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path
