"""Logging configuration for VectorForge"""

import logging
import sys

from vectorforge.api.config import APIConfig


def configure_logging() -> None:
    """Configure logging based on LOG_LEVEL environment variable.

    Sets up basic logging to stdout (required for Docker log collection).
    Respects LOG_LEVEL env var: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    log_level: str = APIConfig.LOG_LEVEL

    valid_levels: list[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        log_level = "INFO"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level {log_level}")
