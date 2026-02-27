from typing import Any

from vectorforge.api import system
from vectorforge.config import VFGConfig
from vectorforge.models.metrics import MetricsResponse

from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Get comprehensive metrics: performance stats (query times, percentiles), memory usage, operation counts, uptime, and system info."
)
@handle_tool_errors
def get_metrics(
    collection_name: str = VFGConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Get comprehensive system metrics.

    Args:
        collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
        Dictionary with detailed performance, usage, memory, timestamp, and system metrics.
    """
    response: MetricsResponse = system.get_collection_metrics(
        collection_name=collection_name
    )
    return build_success_response(response)


@mcp.tool(
    description="Verify VectorForge API connectivity and status. Returns version and health status. Use for monitoring and troubleshooting."
)
@handle_tool_errors
def check_health() -> dict[str, Any]:
    """Check VectorForge API health and connectivity.

    Returns:
        Dictionary with health status and version information.
    """
    response: dict[str, Any] = system.check_health()
    return {"success": True, "data": response}
