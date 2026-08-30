from src.tui import format_large_number, format_percent, parse_tickers


def test_parse_tickers_normalizes_and_deduplicates() -> None:
    assert parse_tickers("nvda, AAPL nvda") == ["NVDA", "AAPL"]


def test_tui_financial_formatting() -> None:
    assert format_large_number(1_250_000_000) == "1.25B"
    assert format_percent(0.125) == "+12.5%"
    assert format_percent(None) == "-"
