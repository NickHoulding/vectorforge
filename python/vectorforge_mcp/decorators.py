"""Decorators for VectorForge MCP tools."""
import functools
import inspect

from typing import Any

import httpx

from fastapi import HTTPException

from .utils import build_error_response


def handle_api_errors(func):
    """Decorator to handle common VectorForge API errors consistently.
    
    Catches and formats errors for:
    - Connection issues (refused, network errors)
    - Timeouts
    - API version mismatches (422 status)
    - Generic exceptions
    
    Args:
        func: The tool function to wrap with error handling.
        
    Returns:
        Wrapped function with comprehensive error handling.
    """
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> dict[str, Any]:
        try:
            return func(*args, **kwargs)
            
        except ConnectionRefusedError:
            return build_error_response(
                Exception("VectorForge API is not available"),
                details="Connection refused - check if API is running"
            )
        except (ConnectionError, httpx.ConnectError):
            return build_error_response(
                Exception("Network error"),
                details="Unable to connect to VectorForge API"
            )
        except (TimeoutError, httpx.TimeoutException):
            return build_error_response(
                Exception("Request timeout"),
                details="VectorForge API request timed out"
            )
        except HTTPException as e:
            if e.status_code == 422:
                return build_error_response(
                    Exception("API version mismatch"),
                    details="Request format incompatible with API version"
                )
            return build_error_response(
                e, 
                details=str(e.status_code)
            )
        except Exception as e:
            return build_error_response(
                Exception("Operation failed"), 
                details=str(e)
            )
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> dict[str, Any]:
        try:
            return await func(*args, **kwargs)
            
        except ConnectionRefusedError:
            return build_error_response(
                Exception("VectorForge API is not available"),
                details="Connection refused - check if API is running"
            )
        except (ConnectionError, httpx.ConnectError):
            return build_error_response(
                Exception("Network error"),
                details="Unable to connect to VectorForge API"
            )
        except (TimeoutError, httpx.TimeoutException):
            return build_error_response(
                Exception("Request timeout"),
                details="VectorForge API request timed out"
            )
        except HTTPException as e:
            if e.status_code == 422:
                return build_error_response(
                    Exception("API version mismatch"),
                    details="Request format incompatible with API version"
                )
            return build_error_response(
                e, 
                details=str(e.status_code)
            )
        except Exception as e:
            return build_error_response(
                Exception("Operation failed"), 
                details=str(e)
            )

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
