import httpx

from fastapi import HTTPException

from vectorforge import api

from ..instance import mcp
from ..utils import build_error_response, build_success_response


@mcp.tool
def get_metrics() -> dict:
    """Get comprehensive system metrics.
    
    Returns:
        Dictionary with detailed performance, usage, memory, timestamp, and system metrics.
    """
    try:
        response = api.get_metrics()
        return build_success_response(response)
    
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
        return build_error_response(e, details=e.status_code)
    except Exception as e:
        return build_error_response(
            Exception("Operation failed"), 
            details=str(e)
        )


@mcp.tool
def check_health() -> dict:
    """Check VectorForge API health and connectivity."""
    try:
        response = api.check_health()
        return {
            "success": True,
            "data": response
        }

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
            Exception("Health check timeout"),
            details="VectorForge API health check timed out"
        )
    except HTTPException as e:
        if e.status_code == 422:
            return build_error_response(
                Exception("API version mismatch"),
                details="Request format incompatible with API version"
            )
        return build_error_response(e, details=e.status_code)
    except Exception as e:
        return build_error_response(e)
