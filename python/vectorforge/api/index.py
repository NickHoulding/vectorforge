"""Index Management Endpoints"""

from typing import Any

from fastapi import APIRouter

from vectorforge.api import engine
from vectorforge.api.decorators import handle_api_errors
from vectorforge.models import HNSWConfig, IndexStatsResponse

router: APIRouter = APIRouter()


@router.get("/index/stats", response_model=IndexStatsResponse)
@handle_api_errors
def get_index_stats() -> IndexStatsResponse:
    """
    Get quick index statistics

    Lightweight endpoint for checking index health and size. Returns essential
    metrics including document counts, embedding dimension, and HNSW index
    configuration. For comprehensive metrics, use GET /metrics instead.

    Returns:
        IndexStatsResponse: Core index statistics and HNSW configuration

    Raises:
        HTTPException: 500 if stats retrieval fails
    """
    stats: dict[str, Any] = engine.get_index_stats()
    hnsw_config_dict: dict[str, Any] = engine.get_hnsw_config()

    hnsw_config = HNSWConfig(
        space=hnsw_config_dict["space"],
        ef_construction=hnsw_config_dict["ef_construction"],
        ef_search=hnsw_config_dict["ef_search"],
        max_neighbors=hnsw_config_dict["max_neighbors"],
        resize_factor=hnsw_config_dict["resize_factor"],
        sync_threshold=hnsw_config_dict["sync_threshold"],
    )

    return IndexStatsResponse(
        status="success",
        total_documents=stats["total_documents"],
        embedding_dimension=stats["embedding_dimension"],
        hnsw_config=hnsw_config,
    )
