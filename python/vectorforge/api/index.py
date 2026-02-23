"""Index Management Endpoints"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from vectorforge.api import engine
from vectorforge.api.decorators import handle_api_errors
from vectorforge.models import (
    HNSWConfig,
    HNSWConfigUpdate,
    HNSWConfigUpdateResponse,
    IndexStatsResponse,
)

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


@router.put("/index/config/hnsw", response_model=HNSWConfigUpdateResponse)
@handle_api_errors
def update_hnsw_config(
    config: HNSWConfigUpdate,
    confirm: Optional[str] = Query(None, description="Must be 'true' to confirm"),
) -> HNSWConfigUpdateResponse:
    """
    Update HNSW index configuration (requires collection recreation)

    This endpoint performs a zero-downtime blue-green migration:
    1. Creates new collection with updated HNSW settings
    2. Migrates all documents with embeddings in batches
    3. Atomically swaps collections
    4. Deletes old collection

    **Warning:** This is a destructive operation that recreates the entire collection.
    Always include `?confirm=true` to proceed.

    Args:
        config: New HNSW configuration (all fields optional, unspecified use defaults)
        confirm: Must be "true" to execute (safety gate)

    Returns:
        HNSWConfigUpdateResponse: Migration statistics and new configuration

    Raises:
        HTTPException: 400 if missing confirmation or invalid configuration
        HTTPException: 503 if migration already in progress
        HTTPException: 500 if migration fails (old collection preserved)

    Example:
        ```
        PUT /index/config/hnsw?confirm=true
        {
            "ef_search": 150,
            "max_neighbors": 32
        }
        ```
    """
    if confirm != "true":
        raise HTTPException(
            status_code=400,
            detail="Must include ?confirm=true to update HNSW config (requires collection recreation)",
        )

    if engine._migration_in_progress:
        raise HTTPException(
            status_code=503,
            detail="HNSW configuration migration already in progress",
        )

    try:
        result = engine.update_hnsw_config(config.model_dump())

    except RuntimeError as e:
        if "already in progress" in str(e):
            raise HTTPException(status_code=503, detail=str(e))

        raise HTTPException(status_code=500, detail=str(e))

    return HNSWConfigUpdateResponse(**result)
