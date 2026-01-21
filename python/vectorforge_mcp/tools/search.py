from vectorforge import api
from vectorforge.config import Config
from vectorforge.models import SearchQuery

from ..decorators import handle_api_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Semantic search across indexed documents using embeddings. Returns top-k most similar results with scores and metadata."
)
@handle_api_errors
def search_documents(query: str, top_k: int = Config.DEFAULT_TOP_K) -> dict:
    """Perform semantic search on indexed documents.
    
    Args:
        query: Search query string (natural language).
        top_k: Number of top results to return (default: 10, max: 100).
        
    Returns:
        List of search results with document IDs, content, similarity scores, and metadata.
    """
    search_params = SearchQuery(query=query, top_k=top_k)
    response = api.search(search_params=search_params)
    return build_success_response(response)
