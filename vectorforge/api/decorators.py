"""Decorators for VectorForge API endpoints."""

import functools
import inspect
import logging
from typing import Any, Callable, TypeVar, cast

from fastapi import HTTPException

from vectorforge.api import manager

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def handle_api_errors(func: F) -> F:
    """Decorator to handle common API errors consistently.

    Catches and converts errors to appropriate HTTP responses:
    - ValueError -> 400 Bad Request
    - FileNotFoundError -> 404 Not Found
    - RuntimeError -> 500 (migration in progress) or 503 (service unavailable)
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

        except HTTPException:
            raise

        except FileNotFoundError as e:
            logger.warning(f"FileNotFoundError: {e}")
            raise HTTPException(status_code=404, detail=f"Resource not found: {str(e)}")

        except ValueError as e:
            logger.warning(f"ValueError: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

        except RuntimeError as e:
            error_msg = str(e)
            if "already in progress" in error_msg.lower():
                raise HTTPException(status_code=503, detail=error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)

        except HTTPException:
            raise

        except FileNotFoundError as e:
            logger.warning(f"FileNotFoundError: {e}")
            raise HTTPException(status_code=404, detail=f"Resource not found: {str(e)}")

        except ValueError as e:
            logger.warning(f"ValueError: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

        except RuntimeError as e:
            error_msg = str(e)
            if "already in progress" in error_msg.lower():
                raise HTTPException(status_code=503, detail=error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    if inspect.iscoroutinefunction(func):
        return cast(F, async_wrapper)
    else:
        return cast(F, sync_wrapper)


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
        collection_name = kwargs.get("collection_name")
        if collection_name:
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
        return func(*args, **kwargs)

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        collection_name = kwargs.get("collection_name")
        if collection_name:
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
        return await func(*args, **kwargs)

    if inspect.iscoroutinefunction(func):
        return cast(F, async_wrapper)
    else:
        return cast(F, sync_wrapper)
