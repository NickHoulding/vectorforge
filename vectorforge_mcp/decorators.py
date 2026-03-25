"""Error-handling decorator for VectorForge MCP tool functions."""

import functools
import inspect
import logging
from typing import Any, Callable, cast

import requests

from .utils import build_error_response

logger = logging.getLogger(__name__)


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

        except requests.ConnectionError as e:
            logger.error(
                "Connection error in %s: %s",
                func.__name__,
                str(e),
                exc_info=True,
            )
            return build_error_response(
                Exception("VectorForge API is not available"),
                details="Connection refused - check if API is running",
            )
        except requests.Timeout as e:
            logger.error(
                "Timeout error in %s: %s",
                func.__name__,
                str(e),
                exc_info=True,
            )
            return build_error_response(
                Exception("Request timeout"),
                details="VectorForge API request timed out",
            )
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None

            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)

            logger.error(
                "HTTP error in %s: status=%s, detail=%s",
                func.__name__,
                status_code,
                detail,
                exc_info=True,
            )
            return build_error_response(Exception(detail), details=status_code)
        except Exception as e:
            logger.error(
                "Unexpected error in %s: %s",
                func.__name__,
                str(e),
                exc_info=True,
            )
            return build_error_response(Exception("Operation failed"), details=str(e))

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            logger.debug("Calling async tool function: %s", func.__name__)
            result = cast(dict[str, Any], await func(*args, **kwargs))
            logger.debug("Async tool function %s completed successfully", func.__name__)
            return result

        except requests.ConnectionError as e:
            logger.error(
                "Connection error in %s: %s",
                func.__name__,
                str(e),
                exc_info=True,
            )
            return build_error_response(
                Exception("VectorForge API is not available"),
                details="Connection refused - check if API is running",
            )
        except requests.Timeout as e:
            logger.error(
                "Timeout error in %s: %s",
                func.__name__,
                str(e),
                exc_info=True,
            )
            return build_error_response(
                Exception("Request timeout"),
                details="VectorForge API request timed out",
            )
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None

            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)

            logger.error(
                "HTTP error in %s: status=%s, detail=%s",
                func.__name__,
                status_code,
                detail,
                exc_info=True,
            )
            return build_error_response(Exception(detail), details=status_code)
        except Exception as e:
            logger.error(
                "Unexpected error in %s: %s",
                func.__name__,
                str(e),
                exc_info=True,
            )
            return build_error_response(Exception("Operation failed"), details=str(e))

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
