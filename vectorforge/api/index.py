"""Index statistics endpoints for collections."""

from typing import Any

from fastapi import APIRouter

from vectorforge.api import manager
from vectorforge.api.decorators import handle_api_errors, require_collection
from vectorforge.models import IndexStatsResponse

router: APIRouter = APIRouter()


@router.get("/collections/{collection_name}/stats", response_model=IndexStatsResponse)
@require_collection
@handle_api_errors
def get_index_stats(collection_name: str) -> IndexStatsResponse:
    """
    Get quick index statistics for a specific collection

    Lightweight endpoint for checking index health and size. Returns essential
    metrics including document counts and embedding dimension. For comprehensive
    metrics, use GET /collections/{name}/metrics instead.

    Args:
        collection_name: Name of the collection

    Returns:
        IndexStatsResponse: Core index statistics

    Raises:
        HTTPException: 404 if collection not found
        HTTPException: 500 if stats retrieval fails
    """
    engine = manager.get_engine(collection_name)
    stats: dict[str, Any] = engine.get_index_stats()

    return IndexStatsResponse(
        status="success",
        total_documents=stats["total_documents"],
        embedding_dimension=stats["embedding_dimension"],
    )
