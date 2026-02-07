from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel


def build_success_response(response: BaseModel, **extra_fields: Any) -> dict[str, Any]:
    """Build a standardized success response from an API response model.

    Converts a Pydantic response model to a dictionary and adds a success flag
    along with any additional fields specified.

    Args:
        response: Pydantic model instance from the API
        **extra_fields: Additional key-value pairs to include in the response

    Returns:
        Dictionary with 'success': True and all response/extra fields

    Example:
        >>> response = api.get_index_stats()
        >>> return build_success_response(response)
        {'success': True, 'total_documents': 100, ...}
    """
    return {"success": True, **response.model_dump(), **extra_fields}


def build_error_response(error: Exception, details: Any = None) -> dict[str, Any]:
    """Build a standardized error response.

    Creates a consistent error response structure for MCP tools to return
    when operations fail.

    Args:
        error: The exception that was caught
        details: Optional additional details (e.g., status code)

    Returns:
        Dictionary with 'success': False, error message, and optional details

    Example:
        >>> return build_error_response(e, details=e.status_code)
        {'success': False, 'error': 'Not found', 'details': 404}
    """
    response: dict[str, Any] = {
        "success": False,
        "error": (error.detail if isinstance(error, HTTPException) else str(error)),
    }

    if details is not None:
        response["details"] = details

    return response
