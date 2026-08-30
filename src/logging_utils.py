from __future__ import annotations

import logging
import sys


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_logger(stage: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logging.getLogger("earnings_quant"), {"stage": stage})


def log(stage: str, message: str, *args: object) -> None:
    logging.getLogger("earnings_quant").info("[%s] " + message, stage, *args)
