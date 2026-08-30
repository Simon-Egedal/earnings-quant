from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.financial_model import FinancialForecaster
from src.models.reaction_model import ReactionForecaster
from src.models.training import _remove_stale_evaluation_artifacts


def test_two_stage_models_fit_and_predict():
    rng = np.random.default_rng(42)
    n = 48
    frame = pd.DataFrame({
        "event_year": np.repeat([2020, 2021, 2022, 2023], 12),
        "lag_revenue": rng.normal(100, 10, n), "revenue_yoy": rng.normal(.1, .05, n),
        "predicted_eps_surprise": rng.normal(0, .05, n), "sector": np.tile(["Tech", "Health"], n // 2),
    })
    frame["actual_revenue"] = frame["lag_revenue"] * (1 + frame["revenue_yoy"]) + rng.normal(0, 1, n)
    frame["abnormal_return_3d"] = .3 * frame["predicted_eps_surprise"] + rng.normal(0, .01, n)
    financial = FinancialForecaster(["actual_revenue"], ["lag_revenue", "revenue_yoy"], ["sector"], seed=42).fit(frame)
    prediction = financial.predict(frame.tail(3))
    assert prediction["predicted_revenue"].notna().all()
    reaction = ReactionForecaster("abnormal_return_3d", ["predicted_eps_surprise", "revenue_yoy"], ["sector"], seed=42).fit(frame)
    scored = reaction.predict(frame.tail(3))
    assert scored["probability_up"].between(0, 1).all()
    assert "baseline_mean" in reaction.validation_metrics["regression"]["candidate_rmse"]
    assert "baseline_prior" in reaction.validation_metrics["classification"]["candidate_log_loss"]


def test_revenue_forecast_is_anchored_to_company_scale():
    rng = np.random.default_rng(7)
    n = 48
    lag_revenue = rng.uniform(1_000, 10_000, n)
    frame = pd.DataFrame({
        "event_year": np.repeat([2020, 2021, 2022, 2023], 12),
        "lag_revenue": lag_revenue,
        "revenue_yoy": 0.10,
        "actual_revenue": lag_revenue * 1.10,
    })
    model = FinancialForecaster(
        ["actual_revenue"], ["lag_revenue", "revenue_yoy"], seed=7
    ).fit(frame)

    prediction = model.predict(pd.DataFrame({"lag_revenue": [10.0], "revenue_yoy": [0.10]}))

    assert prediction.loc[0, "predicted_revenue"] == pytest.approx(11.0, rel=0.10)


def test_financial_model_treats_infinite_features_as_missing():
    frame = pd.DataFrame({
        "event_year": np.repeat([2020, 2021, 2022, 2023], 12),
        "lag_revenue": np.linspace(100.0, 200.0, 48),
        "revenue_yoy": np.linspace(0.01, 0.20, 48),
    })
    frame["actual_revenue"] = frame["lag_revenue"] * 1.1
    frame.loc[0, "revenue_yoy"] = np.inf
    model = FinancialForecaster(
        ["actual_revenue"], ["lag_revenue", "revenue_yoy"], seed=42
    ).fit(frame)

    prediction = model.predict(pd.DataFrame({"lag_revenue": [150.0], "revenue_yoy": [np.inf]}))

    assert np.isfinite(prediction.loc[0, "predicted_revenue"])


def test_retraining_invalidates_old_evaluation_artifacts(tmp_path) -> None:
    stale = [
        tmp_path / "holdout_predictions.parquet",
        tmp_path / "walk_forward_predictions.parquet",
        tmp_path / "holdout_metrics.json",
    ]
    for path in stale:
        path.write_text("stale")

    _remove_stale_evaluation_artifacts(tmp_path)

    assert not any(path.exists() for path in stale)
