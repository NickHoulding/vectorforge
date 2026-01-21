from vectorforge import api

from ..decorators import handle_api_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Get comprehensive metrics: performance stats (query times, percentiles), memory usage, operation counts, uptime, and system info."
)
@handle_api_errors
def get_metrics() -> dict:
    """Get comprehensive system metrics.
    
    Returns:
        Dictionary with detailed performance, usage, memory, timestamp, and system metrics.
    """
    response = api.get_metrics()
    return build_success_response(response)


@mcp.tool(
    description="Verify VectorForge API connectivity and status. Returns version and health status. Use for monitoring and troubleshooting."
)
@handle_api_errors
def check_health() -> dict:
    """Check VectorForge API health and connectivity.
    
    Returns:
        Dictionary with health status and version information.
    """
    response = api.check_health()
    return {
        "success": True,
        "data": response
    }
