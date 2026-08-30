from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(frame: pd.DataFrame, return_column: str = "strategy_return") -> dict[str, float]:
    traded = frame.loc[frame["signal"] != "NO_TRADE"].copy()
    returns = pd.to_numeric(traded.get(return_column, pd.Series(dtype=float)), errors="coerce").dropna()
    if returns.empty:
        return {"number_of_signals": 0, "number_of_long_signals": 0, "number_of_short_signals": 0}
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    downside = returns.loc[returns < 0]
    gross_profit = returns.loc[returns > 0].sum()
    gross_loss = -returns.loc[returns < 0].sum()
    dates = pd.to_datetime(traded.loc[returns.index, "earnings_date"], utc=True, errors="coerce")
    years = max((dates.max() - dates.min()).days / 365.25, 1 / 12) if len(dates) > 1 else np.nan
    annualized = equity.iloc[-1] ** (1 / years) - 1 if pd.notna(years) and equity.iloc[-1] > 0 else np.nan
    direction_actual = np.sign(pd.to_numeric(traded.loc[returns.index, "abnormal_return_3d"], errors="coerce"))
    direction_signal = traded.loc[returns.index, "signal"].map({"LONG": 1, "SHORT": -1})
    return {
        "number_of_signals": int(len(returns)),
        "number_of_long_signals": int((traded.loc[returns.index, "signal"] == "LONG").sum()),
        "number_of_short_signals": int((traded.loc[returns.index, "signal"] == "SHORT").sum()),
        "win_rate": float((returns > 0).mean()), "average_return": float(returns.mean()),
        "median_return": float(returns.median()), "cumulative_return": float(equity.iloc[-1] - 1),
        "annualized_return": float(annualized),
        "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(len(returns))) if returns.std() > 0 else np.nan,
        "sortino_ratio": float(returns.mean() / downside.std() * np.sqrt(len(returns))) if downside.std() > 0 else np.nan,
        "maximum_drawdown": float(drawdown.min()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "directional_accuracy": float((direction_actual == direction_signal).mean()),
    }


def grouped_metrics(frame: pd.DataFrame) -> dict[str, dict]:
    output = {"overall": calculate_metrics(frame)}
    for column in ("signal", "sector", "market_cap_group", "confidence_group"):
        if column not in frame:
            continue
        for value, group in frame.groupby(column, dropna=False, observed=True):
            output[f"{column}:{value}"] = calculate_metrics(group)
    return output
