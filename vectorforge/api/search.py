"""Search Endpoints

Covers:
    POST /collections/{collection_name}/search
    POST /collections/{collection_name}/search  (with filters)
"""

from fastapi import APIRouter

from vectorforge.api import manager
from vectorforge.api.decorators import handle_api_errors, require_collection
from vectorforge.models import SearchQuery, SearchResponse, SearchResult

router: APIRouter = APIRouter()


@router.post("/collections/{collection_name}/search", response_model=SearchResponse)
@require_collection
@handle_api_errors
def search(collection_name: str, search_params: SearchQuery) -> SearchResponse:
    """
    Perform semantic search within a specific collection

    Searches the collection's vector index using semantic similarity to find
    the most relevant document chunks for the given query. Returns results
    ranked by similarity score. Optionally filter results by metadata fields.

    Args:
        collection_name: Name of the collection to search
        search_params: Query string, number of results (top_k), and optional filters

    Returns:
        SearchResponse: Ranked search results with similarity scores and metadata

    Raises:
        HTTPException: 404 if collection not found
        HTTPException: 500 if search fails

    Example:
        ```
        POST /collections/customer_docs/search
        {
            "query": "What is machine learning?",
            "top_k": 5,
            "filters": {"source_file": "textbook.pdf"}
        }
        ```
    """
    engine = manager.get_engine(collection_name)
    query: str = search_params.query.strip()
    results: list[SearchResult] = engine.search(
        query=query, top_k=search_params.top_k, filters=search_params.filters
    )

    return SearchResponse(query=query, results=results, count=len(results))
