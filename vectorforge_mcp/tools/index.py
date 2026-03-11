"""MCP tools for retrieving index statistics."""

from typing import Any

from ..client import get
from ..config import MCPConfig
from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Get index health check: document count, embedding dimension, and HNSW configuration."
)
@handle_tool_errors
def get_index_stats(
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Get quick index statistics.

    Args:
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      Dictionary with index statistics including document count, embedding dimension, and HNSW configuration.
    """
    data = get(f"/collections/{collection_name}/stats")
    return build_success_response(data)
