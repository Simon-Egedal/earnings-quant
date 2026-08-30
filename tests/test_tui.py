import pandas as pd

from src.tui import financial_comparison_rows, format_large_number, format_percent, parse_tickers


def test_parse_tickers_normalizes_and_deduplicates() -> None:
    assert parse_tickers("nvda, AAPL nvda") == ["NVDA", "AAPL"]


def test_tui_financial_formatting() -> None:
    assert format_large_number(1_250_000_000) == "1.25B"
    assert format_percent(0.125) == "+12.5%"
    assert format_percent(None) == "-"


def test_financial_comparison_rows_show_latest_and_forecast() -> None:
    row = pd.Series({
        "lag_eps_diluted": 0.36,
        "predicted_eps": 1.09,
        "lag_revenue": 4_800_000_000,
        "predicted_revenue": 6_340_000_000,
        "lag_operating_margin": 0.061,
        "predicted_operating_margin": 0.088,
        "lag_free_cash_flow": 150_000_000,
        "predicted_fcf": 189_310_000,
    })

    assert financial_comparison_rows(row) == [
        ("EPS", "0.36", "1.09"),
        ("Revenue", "4.80B", "6.34B"),
        ("Operating margin", "+6.1%", "+8.8%"),
        ("Free cash flow", "150.00M", "189.31M"),
    ]
