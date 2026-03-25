"""MCP tools for semantic search across indexed documents."""

import logging
from typing import Any

from ..client import post
from ..config import MCPConfig
from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response

logger = logging.getLogger(__name__)


@mcp.tool(
    description="Semantic search across indexed documents using embeddings. Returns top-k most similar results with scores and metadata. Optionally filter by metadata fields using the 'where' parameter (dict of key-value conditions with AND logic). Supports exact matching and operator expressions ($gte, $lte, $ne, $in)."
)
@handle_tool_errors
def search_documents(
    query: str,
    top_k: int = MCPConfig.DEFAULT_TOP_K,
    where: dict[str, Any] | None = None,
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Perform semantic search on indexed documents with optional metadata filtering.

    The 'where' parameter accepts metadata filters as key-value pairs. All conditions
    are combined with AND logic (all must match). Supports exact equality matching and
    operator expressions for advanced filtering.

    Args:
      query: Search query string (natural language).
      top_k: Number of top results to return (default: 10, max: 100).
      where: Optional metadata filters as dict. Examples:
          - {"source_file": "textbook.pdf"} - exact match
          - {"year": {"$gte": 2024}} - greater than or equal
          - {"category": {"$in": ["AI", "ML"]}} - value in list
          - {"source_file": "guide.pdf", "chunk_index": 0} - multiple conditions (AND)
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      List of search results with document IDs, content, similarity scores, and metadata.
    """
    logger.debug(
        "Searching documents: query_len=%d, top_k=%d, has_filters=%s, collection=%s",
        len(query),
        top_k,
        where is not None,
        collection_name,
    )
    body: dict[str, Any] = {"query": query, "top_k": top_k}
    if where is not None:
        body["filters"] = where

    data = post(f"/collections/{collection_name}/search", json=body)
    result_count = len(data.get("results", []))
    logger.info(
        "Search completed: collection=%s, results=%d", collection_name, result_count
    )
    return build_success_response(data)
