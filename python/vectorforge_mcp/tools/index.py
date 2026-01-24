from typing import Any

from vectorforge import api
from vectorforge.config import VFConfig
from vectorforge.models.index import (
    IndexLoadResponse,
    IndexSaveResponse,
    IndexStatsResponse,
)

from ..decorators import handle_api_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Get lightweight index health check: document counts, embedding dimension, deletion ratio, compaction status."
)
@handle_api_errors
def get_index_stats() -> dict[str, Any]:
    """Get quick index statistics.
    
    Returns:
        Dictionary with index statistics including document counts, embedding dimension, and compaction status.
    """
    response: IndexStatsResponse = api.get_index_stats()
    return build_success_response(response)


@mcp.tool(
    description="Rebuild entire vector index from scratch. Regenerates all embeddings. Optimizes search performance but time-intensive."
)
@handle_api_errors
def build_index() -> dict[str, Any]:
    """Build or rebuild the vector index.
    
    Returns:
        Dictionary with updated index statistics after rebuild.
    """
    response: IndexStatsResponse = api.build_index()
    return build_success_response(response)


@mcp.tool(
    description="Persist index to disk (embeddings + metadata). Enables fast recovery and reduces startup time. Returns file sizes and counts."
)
@handle_api_errors
def save_index(directory: str = VFConfig.DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Persist index to disk.
    
    Args:
        directory: Directory path for saving (default: './data').
        
    Returns:
        Dictionary with save confirmation, file sizes, and document counts.
    """
    response: IndexSaveResponse = api.save_index(directory=directory)
    return build_success_response(response)


@mcp.tool(
    description="Restore index from disk. Loads previously saved embeddings and metadata. Faster than rebuilding from documents."
)
@handle_api_errors
def load_index(directory: str = VFConfig.DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Load index from disk.
    
    Args:
        directory: Directory path to load from (default: './data').
        
    Returns:
        Dictionary with load confirmation, counts, and version information.
    """
    response: IndexLoadResponse = api.load_index(directory=directory)
    return build_success_response(response)
