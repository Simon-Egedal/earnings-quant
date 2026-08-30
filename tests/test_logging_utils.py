import logging

from src.logging_utils import log


def test_log_formats_stage_and_message_arguments(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="earnings_quant"):
        log("DATA", "%d quarterly SEC reports loaded for %s", 437, "NVDA")

    assert caplog.messages == ["[DATA] 437 quarterly SEC reports loaded for NVDA"]


def test_log_accepts_message_without_arguments(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="earnings_quant"):
        log("TRAIN", "Generating forecasts...")

    assert caplog.messages == ["[TRAIN] Generating forecasts..."]
