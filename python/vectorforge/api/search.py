"""Search Endpoints"""

from fastapi import APIRouter, HTTPException

from vectorforge.api import engine
from vectorforge.models import SearchQuery, SearchResponse, SearchResult


router: APIRouter = APIRouter()

@router.post('/search', response_model=SearchResponse)
def search(search_params: SearchQuery) -> SearchResponse:
    """
    Perform semantic search on indexed documents
    
    Searches the vector index using semantic similarity to find the most relevant
    document chunks for the given query. Returns results ranked by similarity score.
    
    Args:
        search_params (SearchQuery): Query string and number of results (top_k)
        
    Returns:
        SearchResponse: Ranked search results with similarity scores and metadata
        
    Raises:
        HTTPException: 500 if search fails
        
    Example:
        ```json
        {
            "query": "What is machine learning?",
            "top_k": 5
        }
        ```
    """
    try:
        query: str = search_params.query.strip()
        results: list[SearchResult] = engine.search(
            query=query, 
            top_k=search_params.top_k
        )

        return SearchResponse(
            query=query,
            results=results,
            count=len(results)
        )
    
    except ValueError as e:
        print(f"ValueError: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Malformed data: {e}"
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
