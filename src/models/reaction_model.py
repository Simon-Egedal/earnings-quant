from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline

from .common import finite_target, make_preprocessor


def _reaction_regressors(seed: int) -> dict[str, object]:
    return {
        "linear": LinearRegression(), "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=200, l2_regularization=1.0, random_state=seed),
    }


def _classifiers(seed: int) -> dict[str, object]:
    return {
        "logistic": LogisticRegression(C=1.0, max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0, random_state=seed),
    }


@dataclass
class ReactionForecaster:
    target: str
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    seed: int = 42
    regression_model: Pipeline | None = None
    classification_model: Pipeline | None = None
    selected_models: dict[str, str] = field(default_factory=dict)
    validation_metrics: dict[str, dict] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "ReactionForecaster":
        y = finite_target(frame, self.target)
        valid = y.notna()
        data, y = frame.loc[valid], y.loc[valid]
        years = sorted(data["event_year"].astype(int).unique())
        validation_year = years[-1] if len(years) > 1 else None
        train = data["event_year"] < validation_year if validation_year else pd.Series(True, index=data.index)
        validation = data["event_year"] == validation_year if validation_year else train
        if train.sum() < 10 or validation.sum() < 2:
            train = validation = pd.Series(True, index=data.index)
        reg_scores: dict[str, float] = {}
        for name, estimator in _reaction_regressors(self.seed).items():
            pipeline = Pipeline([("preprocess", make_preprocessor(self.numeric_features, self.categorical_features, name in {"linear", "ridge"})), ("model", estimator)])
            try:
                pipeline.fit(data.loc[train], y.loc[train])
                reg_scores[name] = float(np.sqrt(mean_squared_error(y.loc[validation], pipeline.predict(data.loc[validation]))))
            except (ValueError, TypeError, FloatingPointError):
                continue
        if not reg_scores:
            raise ValueError("No reaction regression candidate could be fitted")
        best_reg = min(reg_scores, key=reg_scores.get)
        self.regression_model = Pipeline([("preprocess", make_preprocessor(self.numeric_features, self.categorical_features, best_reg in {"linear", "ridge"})), ("model", _reaction_regressors(self.seed)[best_reg])])
        self.regression_model.fit(data, y)
        binary = (y > 0).astype(int)
        class_scores: dict[str, float] = {}
        if binary.nunique() > 1:
            for name, estimator in _classifiers(self.seed).items():
                pipeline = Pipeline([("preprocess", make_preprocessor(self.numeric_features, self.categorical_features, name == "logistic")), ("model", estimator)])
                try:
                    pipeline.fit(data.loc[train], binary.loc[train])
                    probability = pipeline.predict_proba(data.loc[validation])[:, 1]
                    class_scores[name] = float(log_loss(binary.loc[validation], probability, labels=[0, 1]))
                except (ValueError, TypeError, FloatingPointError):
                    continue
            if class_scores:
                best_class = min(class_scores, key=class_scores.get)
                self.classification_model = Pipeline([("preprocess", make_preprocessor(self.numeric_features, self.categorical_features, best_class == "logistic")), ("model", _classifiers(self.seed)[best_class])])
                self.classification_model.fit(data, binary)
                self.selected_models = {"regression": best_reg, "classification": best_class}
            else:
                self.selected_models = {"regression": best_reg, "classification": "unavailable_in_fold"}
        else:
            self.selected_models = {"regression": best_reg, "classification": "unavailable_single_class"}
        fit_prediction = self.regression_model.predict(data)
        self.validation_metrics = {
            "regression": {"selected": best_reg, "candidate_rmse": reg_scores, "mae_fit": float(mean_absolute_error(y, fit_prediction)), "rmse_fit": float(np.sqrt(mean_squared_error(y, fit_prediction))), "r2_fit": float(r2_score(y, fit_prediction))},
            "classification": {"candidate_log_loss": class_scores},
        }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.regression_model is None:
            raise RuntimeError("Reaction model is not fitted")
        predicted = self.regression_model.predict(frame)
        probability = self.classification_model.predict_proba(frame)[:, 1] if self.classification_model else np.repeat(0.5, len(frame))
        return pd.DataFrame({"predicted_abnormal_return_3d": predicted, "probability_up": probability}, index=frame.index)
