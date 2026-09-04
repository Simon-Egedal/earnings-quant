from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.backtest.charts import generate_charts, generate_feature_importance
from src.backtest.engine import run_backtest
from src.config import ensure_directories, load_config
from src.data.collector import collect_data
from src.data.dataset import build_dataset
from src.logging_utils import configure_logging, log
from src.models.training import evaluate_project, train_project
from src.models.ticker_forecast import run_ticker_forecast
from src.scanner import scan_upcoming


def _dataset(config: dict) -> pd.DataFrame:
    path = Path(config["project"]["data_dir"]) / "processed" / "earnings_events.parquet"
    if not path.exists():
        raise FileNotFoundError("No processed dataset; run build-dataset first")
    return pd.read_parquet(path)


def _print_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("No rows found.")
        return
    display = frame.copy()
    for column in display.columns:
        if any(word in column for word in ("return", "surprise", "probability", "growth", "margin")):
            display[column] = display[column].map(lambda value: f"{value:+.2%}" if pd.notna(value) else "-")
    print(display.to_string(index=False))


def status(config: dict) -> None:
    data_dir = Path(config["project"]["data_dir"])
    report: dict[str, object] = {}
    for name in ("fundamentals", "earnings", "prices", "metadata"):
        path = data_dir / "raw" / f"{name}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            report[name] = {"rows": len(frame), "tickers": int(frame["ticker"].nunique()) if "ticker" in frame else None}
            if "statement_type" in frame:
                report[name]["statement_types"] = {
                    str(name): int(count) for name, count in frame["statement_type"].value_counts().items()
                }
            missing_pct = frame.isna().mean().sort_values(ascending=False)
            report[name]["most_missing_pct"] = {
                column: round(float(rate) * 100, 1) for column, rate in missing_pct.head(5).items() if rate > 0
            }
            date_columns = [column for column in ("filed_at", "earnings_date", "date") if column in frame]
            if date_columns:
                report[name]["coverage"] = [str(frame[date_columns[0]].min()), str(frame[date_columns[0]].max())]
        else:
            report[name] = "missing"
    dataset_path = data_dir / "processed" / "earnings_events.parquet"
    report["earnings_events"] = len(pd.read_parquet(dataset_path)) if dataset_path.exists() else "missing"
    report["trained_models"] = (data_dir / "models" / "model_bundle.joblib").exists()
    print(json.dumps(report, indent=2, default=str))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Earnings Quant research CLI")
    root.add_argument("--config", default=None, help="Path to YAML configuration")
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Collect SEC, Yahoo earnings, metadata, and prices")
    collect.add_argument("--tickers", nargs="*", help="Optional explicit ticker list")
    collect.add_argument("--limit", type=int, help="Limit universe size for a development run")
    collect.add_argument("--refresh", action="store_true")
    commands.add_parser("build-dataset", help="Build point-in-time event dataset")
    commands.add_parser("train", help="Train two-stage models without the final test year")
    commands.add_parser("evaluate", help="Evaluate the untouched final test year")
    commands.add_parser("backtest", help="Backtest saved holdout predictions")
    scan = commands.add_parser("scan", help="Rank upcoming earnings events")
    scan.add_argument("--days", type=int, default=14)
    scan.add_argument("--top", type=int, default=20)
    forecast = commands.add_parser(
        "forecast", help="Train, backtest, and forecast the next statement for one ticker"
    )
    forecast.add_argument("ticker", help="US ticker symbol, for example AAPL")
    commands.add_parser("tui", help="Open the interactive terminal application")
    commands.add_parser("status", help="Show local data and model status")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command != "tui":
        configure_logging(args.verbose)
    config = load_config(args.config)
    ensure_directories(config)
    data_dir = Path(config["project"]["data_dir"])
    if args.command == "collect":
        collect_data(config, args.tickers, args.limit, args.refresh)
    elif args.command == "build-dataset":
        build_dataset(config)
    elif args.command == "train":
        train_project(_dataset(config), config, data_dir / "models")
    elif args.command == "evaluate":
        _, metrics = evaluate_project(_dataset(config), config, data_dir / "models")
        print(json.dumps(metrics, indent=2, default=str))
    elif args.command == "backtest":
        predictions_path = data_dir / "models" / "walk_forward_predictions.parquet"
        if not predictions_path.exists():
            predictions_path = data_dir / "models" / "holdout_predictions.parquet"
        if not predictions_path.exists():
            raise FileNotFoundError("No holdout predictions; run evaluate first")
        results, metrics = run_backtest(pd.read_parquet(predictions_path), config)
        results_dir = Path(config["project"]["results_dir"])
        results.to_parquet(results_dir / "backtest.parquet", index=False)
        (results_dir / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        generate_charts(results, results_dir / "charts")
        import joblib
        generate_feature_importance(joblib.load(data_dir / "models" / "model_bundle.joblib"), results_dir / "charts")
        print(json.dumps(metrics, indent=2, default=str))
    elif args.command == "scan":
        _print_frame(scan_upcoming(config, args.days, args.top))
    elif args.command == "forecast":
        print(json.dumps(run_ticker_forecast(args.ticker, config).to_dict(), indent=2))
    elif args.command == "status":
        status(config)
    elif args.command == "tui":
        from src.tui import EarningsQuantApp
        logger = logging.getLogger("earnings_quant")
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        EarningsQuantApp(args.config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
