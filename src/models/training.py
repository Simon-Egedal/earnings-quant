from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.builder import assert_no_lookahead
from src.features.fundamentals import safe_divide
from src.logging_utils import log
from .common import available_features
from .evaluation import classification_metrics, regression_metrics
from .financial_model import FinancialForecaster
from .reaction_model import ReactionForecaster


@dataclass
class ModelBundle:
    financial: FinancialForecaster
    reaction: ReactionForecaster
    trained_at: str
    train_period: tuple[int, int]


def add_financial_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in predictions:
        output[column] = predictions[column]
    if {"predicted_eps", "consensus_eps"}.issubset(output):
        output["predicted_eps_surprise"] = [safe_divide(p - c, abs(c)) for p, c in zip(output["predicted_eps"], output["consensus_eps"])]
    if {"predicted_revenue", "consensus_revenue"}.issubset(output):
        output["predicted_revenue_surprise"] = [safe_divide(p - c, abs(c)) for p, c in zip(output["predicted_revenue"], output["consensus_revenue"])]
    return output


def _oof_financial(frame: pd.DataFrame, targets: list[str], numeric: list[str], categorical: list[str], seed: int, minimum_rows: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for year in sorted(frame["event_year"].astype(int).unique()):
        train, test = frame[frame["event_year"] < year], frame[frame["event_year"] == year]
        if len(train) < minimum_rows:
            continue
        model = FinancialForecaster(targets, numeric, categorical, seed).fit(train)
        if model.models:
            pieces.append(add_financial_predictions(test, model.predict(test)))
    return pd.concat(pieces).sort_index() if pieces else pd.DataFrame()


def train_project(dataset: pd.DataFrame, config: dict, model_dir: Path) -> ModelBundle:
    assert_no_lookahead(dataset)
    model_config = config["model"]
    seed = int(model_config.get("random_seed", 42))
    final_year = int(config["walk_forward"]["final_test_year"])
    first_year = int(config["walk_forward"].get("first_train_year", dataset["event_year"].min()))
    training = dataset.loc[(dataset["event_year"] >= first_year) & (dataset["event_year"] < final_year)].copy()
    if training.empty:
        raise ValueError(f"No training rows before final test year {final_year}")
    targets = [target for target in model_config["financial_targets"] if target in training]
    financial_numeric, financial_categorical = available_features(training, extra_exclude=targets)
    log("TRAIN", "Generating expanding-window financial forecasts...")
    stacked = _oof_financial(training, targets, financial_numeric, financial_categorical, seed, int(model_config.get("minimum_train_rows", 100)))
    if stacked.empty:
        raise ValueError("Not enough chronological data to create out-of-fold financial predictions")
    financial = FinancialForecaster(targets, financial_numeric, financial_categorical, seed).fit(training)
    reaction_numeric, reaction_categorical = available_features(stacked, extra_exclude=targets)
    target = model_config.get("reaction_target", "abnormal_return_3d")
    reaction = ReactionForecaster(target, reaction_numeric, reaction_categorical, seed).fit(stacked)
    bundle = ModelBundle(financial, reaction, datetime.now(UTC).isoformat(), (int(training["event_year"].min()), int(training["event_year"].max())))
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "model_bundle.joblib")
    metadata = {
        "trained_at": bundle.trained_at, "train_period": bundle.train_period,
        "financial_features": financial_numeric + financial_categorical,
        "reaction_features": reaction_numeric + reaction_categorical,
        "financial_models": financial.selected_models, "reaction_models": reaction.selected_models,
        "financial_validation": financial.validation_metrics, "reaction_validation": reaction.validation_metrics,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    log("TRAIN", "Saved two-stage model bundle trained on %d-%d", *bundle.train_period)
    return bundle


def evaluate_project(dataset: pd.DataFrame, config: dict, model_dir: Path) -> tuple[pd.DataFrame, dict]:
    bundle: ModelBundle = joblib.load(model_dir / "model_bundle.joblib")
    final_year = int(config["walk_forward"]["final_test_year"])
    holdout = dataset.loc[dataset["event_year"] == final_year].copy()
    if holdout.empty:
        raise ValueError(f"No rows in untouched final test year {final_year}")
    scored = add_financial_predictions(holdout, bundle.financial.predict(holdout))
    reaction = bundle.reaction.predict(scored)
    scored = scored.join(reaction)
    metrics: dict[str, dict] = {}
    for target in bundle.financial.targets:
        predicted = f"predicted_{target.removeprefix('actual_')}"
        metrics[target] = regression_metrics(scored[target], scored[predicted])
    target = config["model"].get("reaction_target", "abnormal_return_3d")
    metrics["reaction_regression"] = regression_metrics(scored[target], scored["predicted_abnormal_return_3d"])
    metrics["reaction_classification"] = classification_metrics(scored[target], scored["probability_up"])
    scored.to_parquet(model_dir / "holdout_predictions.parquet", index=False)
    walk_forward_parts: list[pd.DataFrame] = []
    fold_metrics: dict[str, dict] = {}
    first_test = int(config["walk_forward"].get("first_test_year", final_year))
    first_train = int(config["walk_forward"].get("first_train_year", dataset["event_year"].min()))
    targets = [target for target in config["model"]["financial_targets"] if target in dataset]
    seed = int(config["model"].get("random_seed", 42))
    minimum_rows = int(config["model"].get("minimum_train_rows", 100))
    for test_year in range(first_test, final_year):
        training = dataset.loc[(dataset["event_year"] >= first_train) & (dataset["event_year"] < test_year)].copy()
        testing = dataset.loc[dataset["event_year"] == test_year].copy()
        if len(training) < minimum_rows or testing.empty:
            continue
        financial_numeric, financial_categorical = available_features(training, extra_exclude=targets)
        stacked = _oof_financial(training, targets, financial_numeric, financial_categorical, seed, minimum_rows)
        if stacked.empty:
            log("WARN", "Walk-forward %d skipped: insufficient stacked training rows", test_year)
            continue
        fold_financial = FinancialForecaster(targets, financial_numeric, financial_categorical, seed).fit(training)
        reaction_numeric, reaction_categorical = available_features(stacked, extra_exclude=targets)
        fold_reaction = ReactionForecaster(config["model"].get("reaction_target", "abnormal_return_3d"), reaction_numeric, reaction_categorical, seed).fit(stacked)
        fold_scored = add_financial_predictions(testing, fold_financial.predict(testing))
        fold_scored = fold_scored.join(fold_reaction.predict(fold_scored))
        fold_scored["walk_forward_test_year"] = test_year
        walk_forward_parts.append(fold_scored)
        fold_metrics[str(test_year)] = {
            "reaction_regression": regression_metrics(fold_scored[target], fold_scored["predicted_abnormal_return_3d"]),
            "reaction_classification": classification_metrics(fold_scored[target], fold_scored["probability_up"]),
        }
        log("EVALUATE", "Walk-forward test %d: %d events", test_year, len(fold_scored))
    scored["walk_forward_test_year"] = final_year
    walk_forward_parts.append(scored)
    fold_metrics[str(final_year)] = {
        "reaction_regression": metrics["reaction_regression"],
        "reaction_classification": metrics["reaction_classification"],
    }
    walk_forward = pd.concat(walk_forward_parts, ignore_index=True).sort_values("earnings_date")
    walk_forward.to_parquet(model_dir / "walk_forward_predictions.parquet", index=False)
    metrics["walk_forward_folds"] = fold_metrics
    metrics["walk_forward_overall"] = {
        "reaction_regression": regression_metrics(walk_forward[target], walk_forward["predicted_abnormal_return_3d"]),
        "reaction_classification": classification_metrics(walk_forward[target], walk_forward["probability_up"]),
    }
    (model_dir / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    log("EVALUATE", "Evaluated untouched %d holdout: %d events", final_year, len(scored))
    return scored, metrics
