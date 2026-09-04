from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static

from src.config import ensure_directories, load_config
from src.models.ticker_forecast import TARGETS, TARGET_LABELS, TickerForecastResult, run_ticker_forecast


def _as_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def format_number(value: object, decimals: int = 2) -> str:
    number = _as_float(value)
    return "-" if number is None else f"{number:,.{decimals}f}"


def format_large_number(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return "-"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= scale:
            return f"{number / scale:,.2f}{suffix}"
    return f"{number:,.0f}"


def format_percent(value: object) -> str:
    number = _as_float(value)
    return "-" if number is None else f"{number:+.1%}"


def format_score(value: object) -> str:
    number = _as_float(value)
    return "-" if number is None else f"{number:.3f} ({number:.1%})"


def parse_tickers(value: str) -> list[str]:
    normalized = value.replace(",", " ").split()
    return list(dict.fromkeys(ticker.strip().upper() for ticker in normalized if ticker.strip()))


def financial_comparison_rows(row: pd.Series) -> list[tuple[str, str, str]]:
    """Retained as a small public formatting helper for existing callers."""
    return [
        ("EPS", format_number(row.get("lag_eps_diluted")), format_number(row.get("predicted_eps"))),
        ("Revenue", format_large_number(row.get("lag_revenue")), format_large_number(row.get("predicted_revenue"))),
        ("Operating margin", format_percent(row.get("lag_operating_margin")), format_percent(row.get("predicted_operating_margin"))),
        ("Free cash flow", format_large_number(row.get("lag_free_cash_flow")), format_large_number(row.get("predicted_fcf"))),
    ]


def _format_metric(metric: str, value: object) -> str:
    if metric in {"revenue", "free_cash_flow"}:
        return format_large_number(value)
    if metric == "operating_margin":
        return format_percent(value)
    return format_number(value)


class EarningsQuantApp(App[None]):
    """Ticker-first model search and next-statement forecasting UI."""

    TITLE = "Earnings Quant"
    SUB_TITLE = "10-year ticker model and next-statement forecast"
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+s", "analyze", "Analyze")]
    CSS = """
    Screen { background: #07111f; color: #d7e3f4; }
    Header, Footer { background: #0b1e33; color: #f4c95d; }
    #body { padding: 1 2; }
    #intro { color: #8ba6c6; margin-bottom: 1; }
    #controls { height: 3; margin-bottom: 1; }
    #ticker-input { width: 32; margin-right: 1; }
    #run-model { min-width: 24; }
    #status { height: 3; color: #8ba6c6; }
    #model-summary { height: 4; padding: 0 1; border: round #2d8b71; }
    .section-title { color: #f4c95d; text-style: bold; margin-top: 1; }
    #attempts-table { height: 10; min-height: 7; border: round #24496b; }
    #forecast-table { height: 9; min-height: 7; border: round #2d8b71; }
    #detail { min-height: 3; padding: 0 1; color: #a9bed5; }
    #disclaimer { height: 2; color: #6f87a3; text-style: italic; }
    DataTable > .datatable--cursor { background: #1c4568; color: white; }
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        self.config = load_config(config_path)
        ensure_directories(self.config)
        self.result: TickerForecastResult | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="body"):
            yield Static(
                "Enter one ticker. The app fetches up to 10 years of SEC or Alpha Vantage fundamentals, "
                "tries models against unseen historical quarters, and forecasts the next statement.",
                id="intro",
            )
            with Horizontal(id="controls"):
                yield Input(placeholder="Ticker (for example AAPL)", id="ticker-input")
                yield Button("Build, backtest & forecast", variant="success", id="run-model")
            yield Static("Ready — enter a ticker to begin.", id="status")
            yield Static("No model has been run yet.", id="model-summary")
            yield Label("MODEL ATTEMPTS", classes="section-title")
            yield DataTable(id="attempts-table")
            yield Label("NEXT FINANCIAL STATEMENT", classes="section-title")
            yield DataTable(id="forecast-table")
            yield Static("A forecast is qualified only when its chronological score is at least 0.800.", id="detail")
            yield Static("Research output only — model accuracy is not a guarantee of future results.", id="disclaimer")
        yield Footer()

    def on_mount(self) -> None:
        attempts = self.query_one("#attempts-table", DataTable)
        attempts.zebra_stripes = True
        attempts.add_columns("Attempt", "Model", "Backtest score", "Result")
        forecast = self.query_one("#forecast-table", DataTable)
        forecast.zebra_stripes = True
        forecast.add_columns("Metric", "Latest reported", "Model forecast", "Street consensus", "Metric score")
        self.query_one("#ticker-input", Input).focus()

    def _set_status(self, message: str, *, error: bool = False) -> None:
        style = "bold #ff7b72" if error else "#8ba6c6"
        self.query_one("#status", Static).update(Text(message, style=style))

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#attempts-table", DataTable).loading = busy
        self.query_one("#forecast-table", DataTable).loading = busy
        self.query_one("#run-model", Button).disabled = busy
        self.query_one("#ticker-input", Input).disabled = busy

    @on(Button.Pressed, "#run-model")
    def _run_pressed(self) -> None:
        self.action_analyze()

    @on(Input.Submitted, "#ticker-input")
    def _ticker_submitted(self) -> None:
        self.action_analyze()

    def action_analyze(self) -> None:
        tickers = parse_tickers(self.query_one("#ticker-input", Input).value)
        if len(tickers) != 1:
            self.notify("Enter exactly one ticker.", severity="warning")
            return
        ticker = tickers[0]
        self._set_busy(True)
        self.query_one("#attempts-table", DataTable).clear()
        self.query_one("#forecast-table", DataTable).clear()
        self.query_one("#model-summary", Static).update(f"{ticker}: building chronological model candidates…")
        self._set_status(f"Loading {ticker} SEC fundamentals and backtesting model candidates…")
        self._run_ticker(ticker)

    @work(thread=True, exclusive=True, group="ticker-model", exit_on_error=False)
    def _run_ticker(self, ticker: str) -> None:
        try:
            result = run_ticker_forecast(ticker, self.config)
        except Exception as exc:
            self.call_from_thread(self._show_failure, str(exc))
        else:
            self.call_from_thread(self._show_result, result)

    def _show_failure(self, message: str) -> None:
        self._set_busy(False)
        self.query_one("#model-summary", Static).update("No forecast was created.")
        self._set_status(f"Analysis failed: {message}", error=True)
        self.query_one("#ticker-input", Input).focus()

    def _show_result(self, result: TickerForecastResult) -> None:
        self.result = result
        attempts_table = self.query_one("#attempts-table", DataTable)
        for index, attempt in enumerate(result.attempts, start=1):
            reached = attempt.score >= result.threshold
            attempts_table.add_row(
                str(index), attempt.name, format_score(attempt.score),
                Text(
                    "QUALIFIED" if reached else f"BELOW {result.threshold:.3f}",
                    style="bold green" if reached else "yellow",
                ),
            )
        forecast_table = self.query_one("#forecast-table", DataTable)
        for metric in TARGETS:
            forecast_table.add_row(
                TARGET_LABELS[metric],
                _format_metric(metric, result.latest_reported.get(metric)),
                _format_metric(metric, result.predictions.get(metric)),
                _format_metric(metric, result.consensus.get(metric)),
                format_score(result.target_scores.get(metric)),
            )
        event_text = result.expected_earnings_date or "date unavailable"
        qualification = "QUALIFIED" if result.qualified else "NOT QUALIFIED"
        summary_style = "bold green" if result.qualified else "bold #ff7b72"
        self.query_one("#model-summary", Static).update(Text(
            f"{result.ticker} — {result.company}\n"
            f"Source: {result.data_source}  |  Currency: {result.currency}  |  "
            f"Model: {result.selected_model}  |  "
            f"Accuracy: {format_score(result.score)}  |  "
            f"Required: {result.threshold:.3f}  |  {qualification}",
            style=summary_style,
        ))
        detail = (
            f"History checked: {result.history_start} to {result.history_end} "
            f"({result.history_periods} quarters). Next expected statement: "
            f"{result.expected_fiscal_period}, {event_text}."
        )
        if result.warning:
            detail += f" {result.warning}"
        self.query_one("#detail", Static).update(detail)
        self._set_busy(False)
        self._set_status(
            f"Finished {len(result.attempts)} model attempt(s). Best score: {result.score:.3f}.",
            error=not result.qualified,
        )
        self.query_one("#ticker-input", Input).focus()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Earnings Quant terminal UI")
    parser.add_argument("--config", default=None, help="Path to YAML configuration")
    args = parser.parse_args(argv)
    logger = logging.getLogger("earnings_quant")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    EarningsQuantApp(args.config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
