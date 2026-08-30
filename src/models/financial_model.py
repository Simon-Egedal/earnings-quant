from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .common import finite_target, make_preprocessor


def _regressors(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=5000),
        "random_forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=200, l2_regularization=1.0, random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    except ImportError:
        pass
    return models


@dataclass
class FinancialForecaster:
    targets: list[str]
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    seed: int = 42
    models: dict[str, Pipeline] = field(default_factory=dict)
    selected_models: dict[str, str] = field(default_factory=dict)
    validation_metrics: dict[str, dict] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "FinancialForecaster":
        years = sorted(frame["event_year"].dropna().astype(int).unique())
        validation_year = years[-1] if len(years) > 1 else None
        fit_rows = frame["event_year"] < validation_year if validation_year else pd.Series(True, index=frame.index)
        validation_rows = frame["event_year"] == validation_year if validation_year else fit_rows
        for target in self.targets:
            if target not in frame:
                continue
            y = finite_target(frame, target)
            train_mask = fit_rows & y.notna()
            valid_mask = validation_rows & y.notna()
            if train_mask.sum() < 10 or valid_mask.sum() < 2:
                train_mask = y.notna()
                valid_mask = y.notna()
            best_name, best_rmse, scores = "", np.inf, {}
            for name, estimator in _regressors(self.seed).items():
                pipeline = Pipeline([
                    ("preprocess", make_preprocessor(self.numeric_features, self.categorical_features, scale=name in {"linear", "ridge", "lasso"})),
                    ("model", estimator),
                ])
                try:
                    pipeline.fit(frame.loc[train_mask], y.loc[train_mask])
                    prediction = pipeline.predict(frame.loc[valid_mask])
                    score = float(np.sqrt(mean_squared_error(y.loc[valid_mask], prediction)))
                    scores[name] = score
                    if score < best_rmse:
                        best_name, best_rmse = name, score
                except (ValueError, TypeError, FloatingPointError):
                    continue
            if not best_name:
                continue
            final = Pipeline([
                ("preprocess", make_preprocessor(self.numeric_features, self.categorical_features, scale=best_name in {"linear", "ridge", "lasso"})),
                ("model", _regressors(self.seed)[best_name]),
            ])
            all_valid = y.notna()
            final.fit(frame.loc[all_valid], y.loc[all_valid])
            self.models[target] = final
            self.selected_models[target] = best_name
            prediction = final.predict(frame.loc[all_valid])
            self.validation_metrics[target] = {
                "selected": best_name, "candidate_rmse": scores,
                "mae_fit": float(mean_absolute_error(y.loc[all_valid], prediction)),
                "rmse_fit": float(np.sqrt(mean_squared_error(y.loc[all_valid], prediction))),
                "r2_fit": float(r2_score(y.loc[all_valid], prediction)),
            }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for target, model in self.models.items():
            result[f"predicted_{target.removeprefix('actual_')}"] = model.predict(frame)
        return result

