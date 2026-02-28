from typing import Any

from vectorforge.api import index
from vectorforge.config import VFGConfig
from vectorforge.models.index import IndexStatsResponse

from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Get index health check: document count, embedding dimension, and HNSW configuration."
)
@handle_tool_errors
def get_index_stats(
    collection_name: str = VFGConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Get quick index statistics.

    Args:
        collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
        Dictionary with index statistics including document count, embedding dimension, and HNSW configuration.
    """
    response: IndexStatsResponse = index.get_collection_stats(
        collection_name=collection_name
    )
    return build_success_response(response)
