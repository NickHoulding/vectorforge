from typing import Any

from vectorforge.api import search
from vectorforge.config import VFGConfig
from vectorforge.models import SearchQuery
from vectorforge.models.search import SearchResponse

from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Semantic search across indexed documents using embeddings. Returns top-k most similar results with scores and metadata. Optionally filter by source_file and/or chunk_index."
)
@handle_tool_errors
def search_documents(
    query: str,
    top_k: int = VFGConfig.DEFAULT_TOP_K,
    source_file: str | None = None,
    chunk_index: int | None = None,
    collection_name: str = VFGConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Perform semantic search on indexed documents with optional filtering.

    Args:
        query: Search query string (natural language).
        top_k: Number of top results to return (default: 10, max: 100).
        source_file: Optional filter by source filename (case-sensitive).
        chunk_index: Optional filter by chunk index (must be >= 0).
        collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
        List of search results with document IDs, content, similarity scores, and metadata.

    Note:
        If both source_file and chunk_index are provided, they are combined with AND logic.
        Both filters use exact equality matching (case-sensitive).
    """
    filters: dict[str, Any] | None = None

    if source_file is not None or chunk_index is not None:
        filters = {}

        if source_file is not None:
            filters["source_file"] = source_file
        if chunk_index is not None:
            filters["chunk_index"] = chunk_index

    search_params: SearchQuery = SearchQuery(query=query, top_k=top_k, filters=filters)
    response: SearchResponse = search.search(
        collection_name=collection_name, search_params=search_params
    )

    return build_success_response(response)
