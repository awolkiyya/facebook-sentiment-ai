from __future__ import annotations

import logging
import sys
from typing import Final


DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


def configure_logging(
    level: str = "INFO",
) -> None:
    """
    Configure application-wide logging.

    This function should be called once during application startup.
    """

    log_level = getattr(
        logging,
        level.upper(),
        logging.INFO,
    )

    logging.basicConfig(
        level=log_level,
        format=DEFAULT_LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def get_logger(
    name: str,
) -> logging.Logger:
    """
    Return a reusable logger for the given module.

    Example:

        logger = get_logger(__name__)
    """

    return logging.getLogger(name)