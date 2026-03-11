"""MCP tools for system health checks and metrics."""

from typing import Any

from ..client import get
from ..config import MCPConfig
from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Get comprehensive metrics: performance stats (query times, percentiles), memory usage, operation counts, uptime, and system info."
)
@handle_tool_errors
def get_metrics(
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Get comprehensive system metrics.

    Args:
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      Dictionary with detailed performance, usage, memory, timestamp, and system metrics.
    """
    data = get(f"/collections/{collection_name}/metrics")
    return build_success_response(data)


@mcp.tool(
    description="Verify VectorForge API connectivity and status. Returns version and health status. Use for monitoring and troubleshooting."
)
@handle_tool_errors
def check_health() -> dict[str, Any]:
    """Check VectorForge API health and connectivity.

    Returns:
      Dictionary with health status and version information.
    """
    data = get("/health")
    return build_success_response(data)
