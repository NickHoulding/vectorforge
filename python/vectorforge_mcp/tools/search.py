from typing import Any

from vectorforge.api import search
from vectorforge.config import VFConfig
from vectorforge.models import SearchQuery
from vectorforge.models.search import SearchResponse

from ..decorators import handle_api_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Semantic search across indexed documents using embeddings. Returns top-k most similar results with scores and metadata."
)
@handle_api_errors
def search_documents(query: str, top_k: int = VFConfig.DEFAULT_TOP_K) -> dict[str, Any]:
    """Perform semantic search on indexed documents.
    
    Args:
        query: Search query string (natural language).
        top_k: Number of top results to return (default: 10, max: 100).
        
    Returns:
        List of search results with document IDs, content, similarity scores, and metadata.
    """
    search_params: SearchQuery = SearchQuery(query=query, top_k=top_k)
    response: SearchResponse = search.search(search_params=search_params)
    return build_success_response(response)
