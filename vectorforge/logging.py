"""Centralized logging configuration for VectorForge.

This module provides:
- Dual-handler setup (console + rotating file)
- JSON formatting for file logs (machine-parseable)
- Human-readable formatting for console logs
- Lazy evaluation support via % formatter
- Sensitive data sanitization
- Environment variable configuration
- Log rotation (10MB files, 5 backups = 50MB total)
"""

import logging
import logging.handlers
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

from vectorforge.config import VFGConfig


def _sanitize_text_for_logging(
    text: str, max_len: int = VFGConfig.LOG_MAX_TEXT_LEN
) -> str:
    """Truncate text for logging to prevent sensitive data exposure and log bloat.

    Args:
        text: Input text to sanitize.
        max_len: Maximum characters to include (default: 100).

    Returns:
        Truncated text with ellipsis if needed.
    """
    if len(text) <= max_len:
        return text

    return text[:max_len] + "..."


def configure_logging(
    log_level: int = VFGConfig.LOG_LEVEL,
    log_file: str = VFGConfig.LOG_FILE,
    json_console: bool = VFGConfig.LOG_JSON_CONSOLE,
    max_bytes: int = VFGConfig.LOG_MAX_BYTES,
    backup_count: int = VFGConfig.LOG_BACKUP_COUNT,
) -> None:
    """Configure application-wide logging with dual handlers.

    Sets up:
    - Console handler at INFO level (human-readable by default)
    - File handler at DEBUG level (JSON format)
    - Rotating file handler (prevents disk space issues)
    - Module-specific logger namespace

    Args:
        log_level: Minimum logging level for the application (DEBUG, INFO, etc.).
            Defaults to VFGConfig.LOG_LEVEL if not specified.
        log_file: Path to log file (relative or absolute).
            Defaults to VFGConfig.LOG_FILE if not specified.
        json_console: Whether to output JSON to console (useful for production).
            Defaults to VFGConfig.LOG_JSON_CONSOLE if not specified.
        max_bytes: Maximum size per log file before rotation in bytes.
            Defaults to VFGConfig.LOG_MAX_BYTES if not specified.
        backup_count: Number of backup files to keep.
            Defaults to VFGConfig.LOG_BACKUP_COUNT if not specified.
    """
    json_formatter = JsonFormatter(
        "%(timestamp)s %(level)s %(name)s %(funcName)s %(lineno)d %(message)s",
        rename_fields={"levelname": "level", "asctime": "timestamp"},
        timestamp=True,
    )
    if log_level == logging.DEBUG:
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(json_formatter if json_console else console_formatter)

    log_path = Path(log_file).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)

    app_logger = logging.getLogger("vectorforge")
    app_logger.setLevel(log_level)
    app_logger.handlers.clear()
    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)
    app_logger.propagate = False

    app_logger.debug(
        "Logging configured: level=%s, file=%s, json_console=%s",
        logging.getLevelName(log_level),
        log_path,
        json_console,
    )
