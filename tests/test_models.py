from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.financial_model import FinancialForecaster
from src.models.reaction_model import ReactionForecaster


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

