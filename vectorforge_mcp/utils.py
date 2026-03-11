"""Shared response-building helpers for MCP tool handlers."""

from typing import Any


def build_success_response(data: dict[str, Any], **extra_fields: Any) -> dict[str, Any]:
    """Build a standardised success response from an API response dictionary.

    Args:
      data: JSON response dict returned from the VectorForge REST API.
      **extra_fields: Additional key-value pairs to merge into the response.

    Returns:
      Dict with ``success: True`` merged with all data and extra fields.
    """
    return {"success": True, **data, **extra_fields}


def build_error_response(error: Exception, details: Any = None) -> dict[str, Any]:
    """Build a standardised error response.

    Args:
      error: The exception that was caught.
      details: Optional extra context, such as an HTTP status code.

    Returns:
      Dict with ``success: False``, the error message, and optional details.
    """
    response: dict[str, Any] = {
        "success": False,
        "error": str(error),
    }

    if details is not None:
        response["details"] = details

    return response
