"""Decorators for VectorForge API endpoints."""

import functools
import inspect
import logging
from typing import Any, Callable, NoReturn, TypeVar, cast

from fastapi import HTTPException

from vectorforge.api import manager

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def _raise_as_http_exception(exc: Exception, func_name: str) -> NoReturn:
    """Translate a caught exception into the appropriate HTTPException and raise it.

    Args:
        exc: The exception caught by an endpoint wrapper.
        func_name: Name of the wrapped endpoint function, for logging.

    Raises:
        HTTPException: Always. Mapped from the exception type per
            ``handle_api_errors``'s docstring.
    """
    if isinstance(exc, HTTPException):
        raise exc

    if isinstance(exc, FileNotFoundError):
        logger.warning("FileNotFoundError in %s: %s", func_name, exc)
        raise HTTPException(status_code=404, detail=f"Resource not found: {str(exc)}")

    if isinstance(exc, ValueError):
        logger.warning("ValueError in %s: %s", func_name, exc)
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(exc)}")

    if isinstance(exc, TypeError):
        logger.warning("TypeError in %s: %s", func_name, exc)
        raise HTTPException(status_code=422, detail=f"Invalid type: {str(exc)}")

    if isinstance(exc, RuntimeError):
        error_msg = str(exc)
        if "already in progress" in error_msg.lower():
            raise HTTPException(status_code=503, detail=error_msg)
        logger.error("RuntimeError in %s: %s", func_name, error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

    logger.error("Unexpected error in %s: %s", func_name, exc, exc_info=True)
    raise HTTPException(status_code=500, detail="Internal server error")


def handle_api_errors(func: F) -> F:
    """Decorator to handle common API errors consistently.

    Catches and converts errors to appropriate HTTP responses:
    - ValueError -> 400 Bad Request
    - FileNotFoundError -> 404 Not Found
    - RuntimeError -> 503 (migration already in progress) or 500 (other runtime errors)
    - HTTPException -> re-raised as-is
    - Generic Exception -> 500 Internal Server Error

    Logs all errors for debugging and monitoring.

    Args:
        func: The endpoint function to wrap with error handling.

    Returns:
        Wrapped function with comprehensive error handling.
    """

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _raise_as_http_exception(e, func.__name__)

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _raise_as_http_exception(e, func.__name__)

    if inspect.iscoroutinefunction(func):
        return cast(F, async_wrapper)
    else:
        return cast(F, sync_wrapper)


def _check_collection_exists(collection_name: str | None) -> None:
    """Raise an HTTPException if the named collection does not exist.

    Args:
        collection_name: Collection name to check, or ``None`` to skip the check.

    Raises:
        HTTPException: 500 if existence check fails, 404 if the collection is
            missing.
    """
    if not collection_name:
        return

    try:
        exists = manager.collection_exists(collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking collection existence: {str(e)}",
        )
    if not exists:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found",
        )


def require_collection(func: F) -> F:
    """Decorator to check if a collection exists before executing the endpoint.

    Looks for 'collection_name' in function kwargs and validates the collection exists.
    Raises 404 HTTPException if collection not found.

    Args:
        func: The endpoint function to wrap with collection validation.

    Returns:
        Wrapped function that validates collection existence.
    """

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        _check_collection_exists(kwargs.get("collection_name"))
        return func(*args, **kwargs)

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        _check_collection_exists(kwargs.get("collection_name"))
        return await func(*args, **kwargs)

    if inspect.iscoroutinefunction(func):
        return cast(F, async_wrapper)
    else:
        return cast(F, sync_wrapper)
