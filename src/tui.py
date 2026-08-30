from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Select, Static

from src.config import ensure_directories, load_config
from src.scanner import get_upcoming_calendar, scan_events


DAY_OPTIONS = (("Next 7 days", 7), ("Next 14 days", 14), ("Next 30 days", 30), ("Next 60 days", 60))


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


def parse_tickers(value: str) -> list[str]:
    normalized = value.replace(",", " ").split()
    return list(dict.fromkeys(ticker.strip().upper() for ticker in normalized if ticker.strip()))


def financial_comparison_rows(row: pd.Series) -> list[tuple[str, str, str]]:
    """Format the latest matching statement beside the model forecast."""
    return [
        ("EPS", format_number(row.get("lag_eps_diluted")), format_number(row.get("predicted_eps"))),
        (
            "Revenue",
            format_large_number(row.get("lag_revenue")),
            format_large_number(row.get("predicted_revenue")),
        ),
        (
            "Operating margin",
            format_percent(row.get("lag_operating_margin")),
            format_percent(row.get("predicted_operating_margin")),
        ),
        (
            "Free cash flow",
            format_large_number(row.get("lag_free_cash_flow")),
            format_large_number(row.get("predicted_fcf")),
        ),
    ]


class EarningsQuantApp(App[None]):
    """Keyboard-driven UI for selecting and scoring upcoming earnings events."""

    TITLE = "Earnings Quant"
    SUB_TITLE = "Upcoming statement growth and reaction scanner"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_calendar", "Refresh"),
        ("ctrl+s", "scan_selected", "Analyze"),
    ]
    CSS = """
    Screen {
        background: #07111f;
        color: #d7e3f4;
    }
    Header {
        background: #0b1e33;
        color: #f4c95d;
    }
    #body {
        padding: 1 2;
    }
    #intro {
        color: #8ba6c6;
        margin-bottom: 1;
    }
    #controls {
        height: 3;
        margin-bottom: 1;
    }
    #ticker-input {
        width: 28;
        margin-right: 1;
    }
    #days-select {
        width: 22;
        margin-right: 1;
    }
    Button {
        margin-right: 1;
        min-width: 16;
    }
    #status {
        height: 2;
        color: #8ba6c6;
    }
    .section-title {
        color: #f4c95d;
        text-style: bold;
        margin-top: 1;
    }
    #calendar-table {
        height: 1fr;
        min-height: 8;
        border: round #24496b;
    }
    #results-table {
        height: 12;
        min-height: 8;
        border: round #2d8b71;
    }
    #financial-comparison {
        height: 7;
        min-height: 7;
        border: round #24496b;
    }
    #detail {
        min-height: 2;
        padding: 0 1;
        color: #a9bed5;
    }
    #disclaimer {
        height: 2;
        color: #6f87a3;
        text-style: italic;
    }
    DataTable > .datatable--cursor {
        background: #1c4568;
        color: white;
    }
    Footer {
        background: #0b1e33;
    }
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        self.config = load_config(config_path)
        ensure_directories(self.config)
        self.calendar = pd.DataFrame()
        self.results = pd.DataFrame()
        self.model_warning = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="body"):
            yield Static(
                "Choose an upcoming company, then run the existing two-stage financial and price-reaction models.",
                id="intro",
            )
            with Horizontal(id="controls"):
                yield Input(placeholder="Ticker (for example NVDA)", id="ticker-input")
                yield Select(DAY_OPTIONS, value=14, allow_blank=False, id="days-select")
                yield Button("Refresh calendar", id="refresh-calendar")
                yield Button("Analyze selected", variant="success", id="run-scan")
            yield Static("Starting…", id="status")
            yield Label("UPCOMING EARNINGS — select a row with Enter", classes="section-title")
            yield DataTable(id="calendar-table")
            yield Label("FORECAST AND COMPARISON", classes="section-title")
            yield DataTable(id="results-table")
            yield Label("LATEST REPORTED VS MODEL FORECAST", classes="section-title")
            yield DataTable(id="financial-comparison")
            yield Static("Select a ticker to see forecast details.", id="detail")
            yield Static("Research output only — this application does not place trades or provide financial advice.", id="disclaimer")
        yield Footer()

    def on_mount(self) -> None:
        calendar_table = self.query_one("#calendar-table", DataTable)
        calendar_table.cursor_type = "row"
        calendar_table.zebra_stripes = True
        calendar_table.add_columns("Ticker", "Company", "Earnings", "Timing", "Street EPS")
        results_table = self.query_one("#results-table", DataTable)
        results_table.cursor_type = "row"
        results_table.zebra_stripes = True
        results_table.add_columns(
            "Ticker", "Statement", "Signal", "Pred EPS", "Street EPS", "EPS YoY", "Pred revenue",
            "Street revenue", "Revenue YoY", "Expected 3D", "P(up)", "Confidence",
        )
        comparison_table = self.query_one("#financial-comparison", DataTable)
        comparison_table.zebra_stripes = True
        comparison_table.add_columns("Metric", "Latest reported", "Model forecast")
        model_path = Path(self.config["project"]["data_dir"]) / "models" / "model_bundle.joblib"
        if model_path.exists():
            metadata_path = model_path.with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            ticker_count = int(metadata.get("training_ticker_count", 0))
            minimum = int(self.config.get("scanner", {}).get("minimum_training_tickers", 20))
            if ticker_count < minimum:
                self.model_warning = f"Model coverage is too small ({ticker_count}/{minimum} training tickers). Signals will be rejected."
                self._set_status(self.model_warning, error=True)
            else:
                self._set_status("Model ready. Loading the upcoming earnings calendar…")
        else:
            self._set_status("Model missing. The calendar will load, but analysis requires: python -m src.cli train", error=True)
        self.action_refresh_calendar()

    def _days(self) -> int:
        value = self.query_one("#days-select", Select).value
        return int(value) if isinstance(value, int) else 14

    def _set_status(self, message: str, *, error: bool = False) -> None:
        style = "bold #ff7b72" if error else "#8ba6c6"
        self.query_one("#status", Static).update(Text(message, style=style))

    def _set_calendar_busy(self, busy: bool) -> None:
        self.query_one("#calendar-table", DataTable).loading = busy
        self.query_one("#refresh-calendar", Button).disabled = busy

    def _set_scan_busy(self, busy: bool) -> None:
        self.query_one("#results-table", DataTable).loading = busy
        self.query_one("#financial-comparison", DataTable).loading = busy
        self.query_one("#run-scan", Button).disabled = busy

    def action_refresh_calendar(self) -> None:
        self._set_calendar_busy(True)
        self._set_status(f"Loading companies reporting in the next {self._days()} days…")
        self._load_calendar(self._days())

    @work(thread=True, exclusive=True, group="calendar", exit_on_error=False)
    def _load_calendar(self, days: int) -> None:
        try:
            calendar = get_upcoming_calendar(self.config, days)
        except Exception as exc:
            self.call_from_thread(self._calendar_failed, str(exc))
        else:
            self.call_from_thread(self._show_calendar, calendar)

    def _calendar_failed(self, message: str) -> None:
        self._set_calendar_busy(False)
        self._set_status(f"Could not load the earnings calendar: {message}", error=True)

    def _show_calendar(self, calendar: pd.DataFrame) -> None:
        self.calendar = calendar.reset_index(drop=True)
        table = self.query_one("#calendar-table", DataTable)
        table.clear()
        for index, row in self.calendar.iterrows():
            date = pd.to_datetime(row.get("earnings_date"), errors="coerce")
            date_text = date.strftime("%Y-%m-%d %H:%M") if pd.notna(date) else "-"
            table.add_row(
                str(row.get("ticker", "-")),
                str(row.get("company", "-")),
                date_text,
                str(row.get("timing", "-") or "-"),
                format_number(row.get("consensus_eps")),
                key=str(index),
            )
        self._set_calendar_busy(False)
        if self.calendar.empty:
            self._set_status("No qualifying earnings events were found for this period.")
        else:
            message = f"Loaded {len(self.calendar)} upcoming events. Select a row or type a ticker."
            if self.model_warning:
                message = f"{message} {self.model_warning}"
            self._set_status(message, error=bool(self.model_warning))

    @on(DataTable.RowSelected, "#calendar-table")
    def _calendar_row_selected(self, event: DataTable.RowSelected) -> None:
        if not 0 <= event.cursor_row < len(self.calendar):
            return
        ticker = str(self.calendar.iloc[event.cursor_row]["ticker"]).upper()
        self.query_one("#ticker-input", Input).value = ticker
        self._set_status(f"Selected {ticker}. Press Analyze selected or Ctrl+S.")

    @on(Button.Pressed, "#refresh-calendar")
    def _refresh_pressed(self) -> None:
        self.action_refresh_calendar()

    @on(Button.Pressed, "#run-scan")
    def _scan_pressed(self) -> None:
        self.action_scan_selected()

    @on(Input.Submitted, "#ticker-input")
    def _ticker_submitted(self) -> None:
        self.action_scan_selected()

    def action_scan_selected(self) -> None:
        tickers = parse_tickers(self.query_one("#ticker-input", Input).value)
        if not tickers:
            self.notify("Select a calendar row or enter a ticker first.", severity="warning")
            return
        if self.calendar.empty:
            self.notify("Load the upcoming earnings calendar first.", severity="warning")
            return
        symbols = self.calendar["ticker"].astype(str).str.upper()
        events = self.calendar.loc[symbols.isin(tickers)].copy()
        missing = sorted(set(tickers) - set(symbols))
        if events.empty:
            self._set_status(
                f"{', '.join(missing)} has no qualifying event in the selected {self._days()}-day window.",
                error=True,
            )
            return
        self._set_scan_busy(True)
        self._set_status(f"Running SEC, analyst, market, financial, and reaction scans for {', '.join(tickers)}…")
        self._scan_events(events, len(events))

    @work(thread=True, exclusive=True, group="scan", exit_on_error=False)
    def _scan_events(self, events: pd.DataFrame, top: int) -> None:
        try:
            results = scan_events(self.config, events, top)
        except Exception as exc:
            self.call_from_thread(self._scan_failed, str(exc))
        else:
            self.call_from_thread(self._show_results, results)

    def _scan_failed(self, message: str) -> None:
        self._set_scan_busy(False)
        self._set_status(f"Analysis failed: {message}", error=True)

    def _show_results(self, results: pd.DataFrame) -> None:
        self.results = results.reset_index(drop=True)
        table = self.query_one("#results-table", DataTable)
        table.clear()
        for index, row in self.results.iterrows():
            signal = str(row.get("signal", "NO_TRADE"))
            signal_style = {
                "LONG": "bold green", "SHORT": "bold red", "INSUFFICIENT_DATA": "bold #ff7b72",
            }.get(signal, "bold yellow")
            table.add_row(
                str(row.get("ticker", "-")),
                str(row.get("statement_type", "-")).title(),
                Text(signal, style=signal_style),
                format_number(row.get("predicted_eps")),
                format_number(row.get("consensus_eps")),
                format_percent(row.get("predicted_eps_growth_yoy")),
                format_large_number(row.get("predicted_revenue")),
                format_large_number(row.get("consensus_revenue")),
                format_percent(row.get("predicted_revenue_growth_yoy")),
                format_percent(row.get("predicted_abnormal_return_3d")),
                format_percent(row.get("probability_up")),
                str(row.get("confidence", "-")),
                key=str(index),
            )
        self._set_scan_busy(False)
        if self.results.empty:
            self.query_one("#financial-comparison", DataTable).clear()
            self.query_one("#detail", Static).update("No ticker could be scored. Check the warnings and source data.")
            self._set_status("No forecast was produced for the selection.", error=True)
            return
        row = self.results.iloc[0]
        self._populate_financial_comparison(row)
        if row.get("data_quality") != "OK":
            self.query_one("#detail", Static).update(
                f"Forecast rejected: {row.get('quality_reason', 'insufficient or implausible source data')}"
            )
            self._set_status("Analysis finished, but no reliable signal could be produced.", error=True)
            return
        detail = (
            f"{row.get('ticker', '-')} {str(row.get('statement_type', '-')).title()}"
            f" ({row.get('expected_fiscal_period', '-')}):"
            f"  •  EPS vs street {format_percent(row.get('predicted_eps_surprise'))}"
            f"  •  revenue vs street {format_percent(row.get('predicted_revenue_surprise'))}"
        )
        self.query_one("#detail", Static).update(detail)
        self._set_status(f"Analysis complete for {len(results)} event(s).")

    @on(DataTable.RowSelected, "#results-table")
    def _result_row_selected(self, event: DataTable.RowSelected) -> None:
        if not 0 <= event.cursor_row < len(self.results):
            return
        row = self.results.iloc[event.cursor_row]
        self._populate_financial_comparison(row)
        if row.get("data_quality") != "OK":
            self.query_one("#detail", Static).update(
                f"Forecast rejected: {row.get('quality_reason', 'insufficient or implausible source data')}"
            )
            return
        self.query_one("#detail", Static).update(
            f"{row.get('ticker', '-')} {str(row.get('statement_type', '-')).title()}"
            f" ({row.get('expected_fiscal_period', '-')}):"
            f"  •  EPS vs street {format_percent(row.get('predicted_eps_surprise'))}"
            f"  •  revenue vs street {format_percent(row.get('predicted_revenue_surprise'))}"
        )

    def _populate_financial_comparison(self, row: pd.Series) -> None:
        comparison = self.query_one("#financial-comparison", DataTable)
        comparison.clear()
        for metric, reported, forecast in financial_comparison_rows(row):
            comparison.add_row(metric, reported, forecast)


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
