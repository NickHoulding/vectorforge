"""Search Endpoints"""

from fastapi import APIRouter, HTTPException

from vectorforge.api import engine
from vectorforge.api.decorators import handle_api_errors
from vectorforge.models import SearchQuery, SearchResponse, SearchResult

router: APIRouter = APIRouter()


@router.post("/search", response_model=SearchResponse)
@handle_api_errors
def search(search_params: SearchQuery) -> SearchResponse:
    """
    Perform semantic search on indexed documents

    Searches the vector index using semantic similarity to find the most relevant
    document chunks for the given query. Returns results ranked by similarity score.
    Optionally filter results by metadata fields.

    Args:
        search_params (SearchQuery): Query string, number of results (top_k),
                                    and optional metadata filters

    Returns:
        SearchResponse: Ranked search results with similarity scores and metadata

    Raises:
        HTTPException: 500 if search fails

    Example:
        ```json
        {
            "query": "What is machine learning?",
            "top_k": 5,
            "filters": {"source_file": "textbook.pdf"}
        }
        ```
    """
    query: str = search_params.query.strip()
    results: list[SearchResult] = engine.search(
        query=query, top_k=search_params.top_k, filters=search_params.filters
    )

    return SearchResponse(query=query, results=results, count=len(results))
