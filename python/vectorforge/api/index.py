"""Index Management Endpoints"""

from typing import Any

from fastapi import APIRouter

from vectorforge.api import engine
from vectorforge.api.decorators import handle_api_errors
from vectorforge.models import IndexStatsResponse

router: APIRouter = APIRouter()


@router.get("/index/stats", response_model=IndexStatsResponse)
@handle_api_errors
def get_index_stats() -> IndexStatsResponse:
    """
    Get quick index statistics

    Lightweight endpoint for checking index health and size. Returns essential
    metrics including document counts and embedding dimension. For comprehensive
    metrics, use GET /metrics instead.

    Returns:
        IndexStatsResponse: Core index statistics

    Raises:
        HTTPException: 500 if stats retrieval fails
    """
    stats: dict[str, Any] = engine.get_index_stats()

    return IndexStatsResponse(
        status="success",
        total_documents=stats["total_documents"],
        embedding_dimension=stats["embedding_dimension"],
    )
