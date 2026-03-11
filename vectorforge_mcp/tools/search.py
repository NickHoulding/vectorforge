"""MCP tools for semantic search across indexed documents."""

from typing import Any

from ..client import post
from ..config import MCPConfig
from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Semantic search across indexed documents using embeddings. Returns top-k most similar results with scores and metadata. Optionally filter by source_file and/or chunk_index."
)
@handle_tool_errors
def search_documents(
    query: str,
    top_k: int = MCPConfig.DEFAULT_TOP_K,
    source_file: str | None = None,
    chunk_index: int | None = None,
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Perform semantic search on indexed documents with optional filtering.

    When both source_file and chunk_index are provided, they are combined with AND logic
    using exact equality matching (case-sensitive).

    Args:
      query: Search query string (natural language).
      top_k: Number of top results to return (default: 10, max: 100).
      source_file: Optional filter by source filename (case-sensitive).
      chunk_index: Optional filter by chunk index (must be >= 0).
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      List of search results with document IDs, content, similarity scores, and metadata.
    """
    filters: dict[str, Any] | None = None
    if source_file is not None or chunk_index is not None:
        filters = {}
        if source_file is not None:
            filters["source_file"] = source_file
        if chunk_index is not None:
            filters["chunk_index"] = chunk_index

    body: dict[str, Any] = {"query": query, "top_k": top_k}
    if filters is not None:
        body["filters"] = filters

    data = post(f"/collections/{collection_name}/search", json=body)
    return build_success_response(data)
