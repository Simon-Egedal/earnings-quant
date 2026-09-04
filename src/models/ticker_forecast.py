from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logging_utils import log
from src.providers import AlphaVantageProvider, SECCompanyFactsProvider, YahooFinanceProvider


TARGETS = ("eps_diluted", "revenue", "operating_margin", "free_cash_flow")
TARGET_LABELS = {
    "eps_diluted": "EPS",
    "revenue": "Revenue",
    "operating_margin": "Operating margin",
    "free_cash_flow": "Free cash flow",
}


@dataclass
class CandidateResult:
    name: str
    score: float
    target_scores: dict[str, float]
    prediction_count: int


@dataclass
class TickerForecastResult:
    ticker: str
    company: str
    data_source: str
    currency: str
    history_start: str
    history_end: str
    history_periods: int
    expected_fiscal_period: str
    expected_earnings_date: str | None
    score: float
    threshold: float
    qualified: bool
    selected_model: str
    attempts: list[CandidateResult] = field(default_factory=list)
    target_scores: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, float | None] = field(default_factory=dict)
    latest_reported: dict[str, float | None] = field(default_factory=dict)
    consensus: dict[str, float | None] = field(default_factory=dict)
    warning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _quarter_number(value: object) -> int:
    text = "" if value is None or pd.isna(value) else str(value).upper()
    if text.startswith("Q") and text[1:2].isdigit():
        return int(text[1])
    if text == "FY":
        return 4
    return 0


def _next_period(value: object) -> str:
    quarter = _quarter_number(value)
    return f"Q{quarter % 4 + 1}" if quarter else "Next quarter"


def prepare_quarterly_history(
    fundamentals: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    years: int = 10,
) -> pd.DataFrame:
    """Return one point-in-time-safe quarterly observation per fiscal period."""
    if fundamentals.empty:
        return pd.DataFrame()
    cutoff = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz="UTC"))
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    frame = fundamentals.copy()
    filed = pd.to_datetime(frame.get("filed_at"), utc=True, errors="coerce")
    periods = pd.to_datetime(frame.get("period_end"), errors="coerce")
    cadence = frame.get("statement_type", pd.Series("quarterly", index=frame.index))
    frame = frame.loc[(filed < cutoff) & cadence.eq("quarterly") & periods.notna()].copy()
    if frame.empty:
        return frame
    frame["filed_at"] = filed.loc[frame.index]
    frame["period_end"] = periods.loc[frame.index]
    earliest = cutoff.tz_localize(None) - pd.DateOffset(years=years)
    frame = frame.loc[frame["period_end"] >= earliest]
    if frame.empty:
        return frame

    # Amendments and later comparative disclosures can duplicate a period. Use
    # the last value available before as_of for each column without allowing a
    # sparse amendment to erase an earlier reported value.
    frame = frame.sort_values(["period_end", "filed_at"])
    grouped = frame.groupby("period_end", sort=True)
    history = grouped.last().reset_index()
    revenue = pd.to_numeric(
        history.get("revenue", pd.Series(np.nan, index=history.index)), errors="coerce"
    )
    operating_income = pd.to_numeric(
        history.get("operating_income", pd.Series(np.nan, index=history.index)), errors="coerce"
    )
    calculated_margin = operating_income / revenue.replace(0, np.nan)
    if "operating_margin" in history:
        calculated_margin = calculated_margin.combine_first(
            pd.to_numeric(history["operating_margin"], errors="coerce")
        )
    history["operating_margin"] = calculated_margin
    for target in TARGETS:
        if target not in history:
            history[target] = np.nan
        history[target] = pd.to_numeric(history[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return history.sort_values("period_end").reset_index(drop=True)


def _feature_row(history: pd.DataFrame, position: int) -> dict[str, float]:
    """Build a row using only observations strictly before ``position``."""
    prior = history.iloc[:position]
    row: dict[str, float] = {}
    for target in TARGETS:
        values = pd.to_numeric(prior[target], errors="coerce")
        for lag in (1, 2, 4):
            row[f"{target}_lag{lag}"] = float(values.iloc[-lag]) if len(values) >= lag else np.nan
        recent = values.tail(4)
        row[f"{target}_mean4"] = float(recent.mean()) if recent.notna().any() else np.nan
        clean = recent.dropna()
        row[f"{target}_trend4"] = (
            float(np.polyfit(np.arange(len(clean)), clean.to_numpy(dtype=float), 1)[0])
            if len(clean) >= 2 else np.nan
        )
    next_quarter = _quarter_number(history.iloc[position].get("fiscal_period")) if position < len(history) else 0
    if not next_quarter and len(prior):
        next_quarter = _quarter_number(prior.iloc[-1].get("fiscal_period")) % 4 + 1
    angle = 2 * math.pi * (next_quarter or 1) / 4
    row["quarter_sin"] = math.sin(angle)
    row["quarter_cos"] = math.cos(angle)
    row["time_index"] = float(position)
    return row


def build_supervised_history(history: pd.DataFrame, minimum_lags: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(history) <= minimum_lags:
        return pd.DataFrame(), pd.DataFrame()
    features = pd.DataFrame([_feature_row(history, position) for position in range(minimum_lags, len(history))])
    targets = history.loc[minimum_lags:, list(TARGETS)].reset_index(drop=True)
    return features, targets


def accuracy_score_01(actual: pd.Series, predicted: pd.Series) -> float:
    """Financial forecast accuracy: 1 - weighted absolute percentage error.

    The result is clipped to [0, 1], so 1 means exact predictions and 0 means
    total absolute error was at least as large as the values being predicted.
    """
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    if not valid.any():
        return float("nan")
    actual_values, predicted_values = actual_values[valid], predicted_values[valid]
    scale = float(np.nanmedian(np.abs(actual_values)))
    floor = max(scale * 0.05, 1e-9)
    denominator = np.maximum(np.abs(actual_values), floor).sum()
    error = np.abs(actual_values - predicted_values).sum()
    return float(np.clip(1.0 - error / denominator, 0.0, 1.0))


def _candidate_factories(seed: int) -> list[tuple[str, Callable[[], object] | None]]:
    return [
        ("seasonal_naive", None),
        ("linear_trend", lambda: Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("model", LinearRegression())])),
        ("ridge_1", lambda: Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("ridge_10", lambda: Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])),
        ("random_forest", lambda: Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", RandomForestRegressor(n_estimators=160, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=seed))])),
        ("extra_trees", lambda: Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=seed))])),
        ("hist_gradient_boosting", lambda: Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", HistGradientBoostingRegressor(max_iter=150, l2_regularization=1.0, min_samples_leaf=5, random_state=seed))])),
    ]


def _predict_one(
    name: str,
    factory: Callable[[], object] | None,
    target: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> float:
    if name == "seasonal_naive":
        value = x_test.iloc[0].get(f"{target}_lag4")
        if pd.isna(value):
            value = x_test.iloc[0].get(f"{target}_lag1")
        return _plausible_value(target, float(value))
    model = factory()
    model.fit(x_train, y_train)
    return _plausible_value(target, float(model.predict(x_test)[0]))


def _plausible_value(target: str, value: float) -> float:
    if target == "revenue":
        return max(0.0, value)
    if target == "operating_margin":
        return float(np.clip(value, -1.0, 1.0))
    return value


def backtest_candidate(
    name: str,
    factory: Callable[[], object] | None,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    minimum_train_rows: int,
    backtest_rows: int,
) -> tuple[CandidateResult, dict[str, list[float]]]:
    start = max(minimum_train_rows, len(features) - backtest_rows)
    actual: dict[str, list[float]] = {target: [] for target in TARGETS}
    predicted: dict[str, list[float]] = {target: [] for target in TARGETS}
    for row_index in range(start, len(features)):
        for target in TARGETS:
            train_mask = targets.loc[: row_index - 1, target].notna()
            y_train = targets.loc[: row_index - 1, target].loc[train_mask]
            current = targets.at[row_index, target]
            if len(y_train) < minimum_train_rows or pd.isna(current):
                continue
            try:
                prediction = _predict_one(
                    name,
                    factory,
                    target,
                    features.loc[: row_index - 1].loc[train_mask],
                    y_train,
                    features.iloc[[row_index]],
                )
            except (TypeError, ValueError, FloatingPointError):
                continue
            if np.isfinite(prediction):
                actual[target].append(float(current))
                predicted[target].append(prediction)
    # A one-off reported value is not enough evidence for a metric score.
    scores = {
        target: accuracy_score_01(pd.Series(actual[target]), pd.Series(predicted[target]))
        for target in TARGETS
        if len(actual[target]) >= 3
    }
    finite_scores = [score for score in scores.values() if np.isfinite(score)]
    score = float(np.mean(finite_scores)) if len(finite_scores) >= 2 else 0.0
    count = sum(len(values) for values in actual.values())
    return CandidateResult(name, score, scores, count), predicted


def select_model(
    history: pd.DataFrame,
    *,
    threshold: float = 0.8,
    minimum_train_rows: int = 8,
    backtest_rows: int = 8,
    seed: int = 42,
) -> tuple[CandidateResult, list[CandidateResult], dict[str, object]]:
    features, targets = build_supervised_history(history)
    if len(features) < minimum_train_rows + 2:
        raise ValueError(
            f"Only {len(history)} usable quarterly periods were found; at least "
            f"{minimum_train_rows + 6} are required for chronological training and backtesting"
        )
    attempts: list[CandidateResult] = []
    factories = dict(_candidate_factories(seed))
    selected: CandidateResult | None = None
    for name, factory in _candidate_factories(seed):
        result, _ = backtest_candidate(
            name, factory, features, targets,
            minimum_train_rows=minimum_train_rows,
            backtest_rows=backtest_rows,
        )
        attempts.append(result)
        log("MODEL", "%s ticker-model backtest accuracy %.3f", name, result.score)
        if selected is None or result.score > selected.score:
            selected = result
        if result.score >= threshold:
            selected = result
            break
    assert selected is not None

    models: dict[str, object] = {}
    next_features = pd.DataFrame([_feature_row(history, len(history))])
    for target in TARGETS:
        valid = targets[target].notna()
        if valid.sum() < minimum_train_rows:
            continue
        factory = factories[selected.name]
        if selected.name == "seasonal_naive":
            models[target] = float(_predict_one(selected.name, factory, target, features.loc[valid], targets.loc[valid, target], next_features))
        else:
            model = factory()
            model.fit(features.loc[valid], targets.loc[valid, target])
            models[target] = model
    return selected, attempts, {"models": models, "next_features": next_features}


def forecast_with_model(selected_name: str, fitted: dict[str, object]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    next_features = fitted["next_features"]
    for target in TARGETS:
        model = fitted["models"].get(target)
        if model is None:
            output[target] = None
        elif selected_name == "seasonal_naive":
            output[target] = float(model)
        else:
            value = _plausible_value(target, float(model.predict(next_features)[0]))
            output[target] = value if np.isfinite(value) else None
    return output


def load_ticker_fundamentals(ticker: str, config: dict) -> tuple[pd.DataFrame, str]:
    """Use SEC for covered filers and Alpha Vantage as the global fallback."""
    data_dir = Path(config["project"]["data_dir"])
    sec = SECCompanyFactsProvider(
        data_dir / "cache",
        config["sec"]["user_agent"],
        config["sec"]["requests_per_second"],
        config["sec"]["cache_days"],
    )
    try:
        fundamentals = sec.fundamentals(ticker)
    except KeyError:
        fundamentals = pd.DataFrame()
    if not fundamentals.empty:
        return fundamentals, "SEC"

    alpha_config = config.get("alpha_vantage", {})
    environment_name = str(alpha_config.get("api_key_env", "ALPHA_VANTAGE_API_KEY"))
    api_key = _load_api_key(environment_name, Path(config["project"].get("root", Path.cwd())))
    if not api_key:
        raise ValueError(
            f"{ticker} is not covered by SEC Company Facts. Set the free Alpha Vantage key in "
            f"{environment_name}; see README.md for setup instructions"
        )
    alpha = AlphaVantageProvider(
        data_dir / "cache", api_key, int(alpha_config.get("cache_days", 7))
    )
    return alpha.fundamentals(ticker), "Alpha Vantage"


def _load_api_key(environment_name: str, project_root: Path) -> str:
    """Read the key from the process environment or an ignored local .env file."""
    configured = os.environ.get(environment_name, "").strip()
    if configured:
        return configured
    env_path = Path(project_root) / ".env"
    if not env_path.exists():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == environment_name:
            return value.strip().strip('"').strip("'")
    return ""


def run_ticker_forecast(ticker: str, config: dict) -> TickerForecastResult:
    symbol = str(ticker).strip().upper()
    if not symbol or len(symbol) > 10 or not all(character.isalnum() or character in ".-" for character in symbol):
        raise ValueError("Enter one valid ticker symbol, for example AAPL")
    ticker_config = config.get("ticker_model", {})
    years = int(ticker_config.get("history_years", 10))
    threshold = float(ticker_config.get("minimum_accuracy", 0.8))
    minimum_train = int(ticker_config.get("minimum_train_quarters", 8))
    backtest_rows = int(ticker_config.get("backtest_quarters", 8))
    seed = int(config.get("project", {}).get("random_seed", 42))
    data_dir = Path(config["project"]["data_dir"])
    fundamentals, data_source = load_ticker_fundamentals(symbol, config)
    history = prepare_quarterly_history(fundamentals, years=years)
    if history.empty:
        raise ValueError(f"No quarterly SEC fundamentals were found for {symbol}")
    selected, attempts, fitted = select_model(
        history,
        threshold=threshold,
        minimum_train_rows=minimum_train,
        backtest_rows=backtest_rows,
        seed=seed,
    )
    predictions = forecast_with_model(selected.name, fitted)

    yahoo = YahooFinanceProvider()
    try:
        metadata = yahoo.metadata(symbol)
    except Exception:
        metadata = {"company": symbol}
    try:
        analyst = yahoo.analyst_snapshot(symbol)
    except Exception:
        analyst = {}
    try:
        event = yahoo.next_earnings_event(symbol)
    except Exception:
        event = None

    latest = history.iloc[-1]
    currency = "USD" if data_source == "SEC" else _text_or_default(latest.get("currency"), "Unknown")
    result = TickerForecastResult(
        ticker=symbol,
        company=str(metadata.get("company") or symbol),
        data_source=data_source,
        currency=currency,
        history_start=str(pd.Timestamp(history["period_end"].min()).date()),
        history_end=str(pd.Timestamp(history["period_end"].max()).date()),
        history_periods=len(history),
        expected_fiscal_period=_next_period(latest.get("fiscal_period")),
        expected_earnings_date=str(pd.Timestamp(event).date()) if event is not None and pd.notna(event) else None,
        score=selected.score,
        threshold=threshold,
        qualified=selected.score >= threshold,
        selected_model=selected.name,
        attempts=attempts,
        target_scores=selected.target_scores,
        predictions={key: (float(value) if value is not None else None) for key, value in predictions.items()},
        latest_reported={key: (float(latest[key]) if pd.notna(latest.get(key)) else None) for key in TARGETS},
        consensus={
            "eps_diluted": _finite_or_none(analyst.get("consensus_eps")),
            "revenue": _finite_or_none(analyst.get("consensus_revenue")),
        },
        warning=(
            "No candidate reached the required accuracy. The best forecast is shown for research, "
            "but it is not qualified."
            if selected.score < threshold else ""
        ),
    )
    artifact_dir = data_dir / "models" / "tickers" / symbol
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted | {"selected_model": selected.name}, artifact_dir / "model.joblib")
    (artifact_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _text_or_default(value: object, default: str) -> str:
    return default if value is None or pd.isna(value) or not str(value).strip() else str(value)
