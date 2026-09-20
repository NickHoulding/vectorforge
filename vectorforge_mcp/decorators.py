"""Error-handling decorator for VectorForge MCP tool functions."""

import functools
import inspect
import logging
from typing import Any, Callable, cast

import requests

from .utils import build_error_response

logger = logging.getLogger(__name__)


def _build_tool_error_response(exc: Exception, func_name: str) -> dict[str, Any]:
    """Translate a caught exception into a standardised error response dict.

    Args:
      exc: The exception caught by a tool wrapper.
      func_name: Name of the wrapped tool function, for logging.

    Returns:
      A ``{"success": False, ...}`` response dict per ``handle_tool_errors``'s
      docstring.
    """
    if isinstance(exc, requests.ConnectionError):
        logger.error("Connection error in %s: %s", func_name, str(exc), exc_info=True)
        return build_error_response(
            Exception("VectorForge API is not available"),
            details="Connection refused - check if API is running",
        )

    if isinstance(exc, requests.Timeout):
        logger.error("Timeout error in %s: %s", func_name, str(exc), exc_info=True)
        return build_error_response(
            Exception("Request timeout"),
            details="VectorForge API request timed out",
        )

    if isinstance(exc, requests.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else None

        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)

        logger.error(
            "HTTP error in %s: status=%s, detail=%s",
            func_name,
            status_code,
            detail,
            exc_info=True,
        )
        return build_error_response(Exception(detail), details=status_code)

    logger.error("Unexpected error in %s: %s", func_name, str(exc), exc_info=True)
    return build_error_response(Exception("Operation failed"), details=str(exc))


def handle_tool_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool function with standardised error handling.

    Catches HTTP, connection, timeout, and generic errors and converts
    them into a consistent ``{"success": False, ...}`` response dict so
    MCP clients always receive a structured response. Supports both sync
    and async tool functions.

    Args:
      func: The MCP tool function to wrap.

    Returns:
      Wrapped function that returns a success or error response dict.
    """

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            logger.debug("Calling tool function: %s", func.__name__)
            result = cast(dict[str, Any], func(*args, **kwargs))
            logger.debug("Tool function %s completed successfully", func.__name__)
            return result
        except Exception as e:
            return _build_tool_error_response(e, func.__name__)

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            logger.debug("Calling async tool function: %s", func.__name__)
            result = cast(dict[str, Any], await func(*args, **kwargs))
            logger.debug("Async tool function %s completed successfully", func.__name__)
            return result
        except Exception as e:
            return _build_tool_error_response(e, func.__name__)

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
