"""MCP tools for retrieving index statistics."""

import logging
from typing import Any

from ..client import get
from ..config import MCPConfig
from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response

logger = logging.getLogger(__name__)


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
    logger.debug("Getting index stats: collection=%s", collection_name)
    data = get(f"/collections/{collection_name}/stats")
    logger.info("Retrieved index stats for collection %s", collection_name)
    return build_success_response(data)
