"""Decorators for VectorForge API endpoints."""
import functools
import inspect

from typing import Any, Callable, TypeVar, cast

from fastapi import HTTPException


F = TypeVar('F', bound=Callable[..., Any])


def handle_api_errors(func: F) -> F:
    """Decorator to handle common API errors consistently.
    
    Catches and converts errors to appropriate HTTP responses:
    - ValueError -> 400 Bad Request
    - FileNotFoundError -> 404 Not Found  
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
            # Re-raise FastAPI exceptions as-is (already formatted)
            raise
            
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Resource not found: {str(e)}"
            )
            
        except ValueError as e:
            print(f"ValueError: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input: {str(e)}"
            )
            
        except Exception as e:
            print(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )
    
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
            
        except HTTPException:
            # Re-raise FastAPI exceptions as-is (already formatted)
            raise
            
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Resource not found: {str(e)}"
            )
            
        except ValueError as e:
            print(f"ValueError: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input: {str(e)}"
            )
            
        except Exception as e:
            print(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )

    if inspect.iscoroutinefunction(func):
        return cast(F, async_wrapper)
    else:
        return cast(F, sync_wrapper)
