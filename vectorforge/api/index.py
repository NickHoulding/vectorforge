"""Index Management Endpoints"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from vectorforge.api import manager
from vectorforge.api.decorators import handle_api_errors, require_collection
from vectorforge.models import (
    HNSWConfig,
    HNSWConfigUpdate,
    HNSWConfigUpdateResponse,
    IndexStatsResponse,
)

router: APIRouter = APIRouter()


@router.get("/collections/{collection_name}/stats", response_model=IndexStatsResponse)
@require_collection
@handle_api_errors
def get_collection_stats(collection_name: str) -> IndexStatsResponse:
    """
    Get quick index statistics for a specific collection

    Lightweight endpoint for checking index health and size. Returns essential
    metrics including document counts, embedding dimension, and HNSW index
    configuration. For comprehensive metrics, use GET /metrics instead.

    Args:
        collection_name: Name of the collection

    Returns:
        IndexStatsResponse: Core index statistics and HNSW configuration

    Raises:
        HTTPException: 404 if collection not found
        HTTPException: 500 if stats retrieval fails
    """
    engine = manager.get_engine(collection_name)
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


@router.put(
    "/collections/{collection_name}/config/hnsw",
    response_model=HNSWConfigUpdateResponse,
)
@require_collection
@handle_api_errors
def update_collection_hnsw_config(
    collection_name: str,
    config: HNSWConfigUpdate,
    confirm: Optional[bool] = Query(None, description="Must be true to confirm"),
) -> HNSWConfigUpdateResponse:
    """
    Update HNSW index configuration for a specific collection (requires collection recreation)

    This endpoint performs a zero-downtime collection-level migration:
    1. Creates new collection with updated HNSW settings
    2. Migrates all documents with embeddings in batches
    3. Atomically swaps to the new collection
    4. Deletes old collection

    Note: This is collection-level blue-green migration within the same ChromaDB
    database. All collections share the same persistent storage volume.

    **Warning:** This is a destructive operation that recreates the entire collection.
    Migration temporarily requires up to 3x disk space. Always include `?confirm=true`
    to proceed.

    Args:
        collection_name: Name of the collection to update
        config: New HNSW configuration (all fields optional, unspecified use defaults)
        confirm: Must be true to execute (safety gate)

    Returns:
        HNSWConfigUpdateResponse: Migration statistics and new configuration

    Raises:
        HTTPException: 404 if collection not found
        HTTPException: 400 if missing confirmation or invalid configuration
        HTTPException: 503 if migration already in progress
        HTTPException: 500 if migration fails (old collection preserved)

    Example:
        ```
        PUT /collections/my_collection/config/hnsw?confirm=true
        {
            "ef_search": 150,
            "max_neighbors": 32
        }
        ```
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must include ?confirm=true to update HNSW config (requires collection recreation)",
        )

    engine = manager.get_engine(collection_name)

    if engine.migration_in_progress:
        raise HTTPException(
            status_code=503,
            detail="HNSW configuration migration already in progress",
        )

    result = engine.update_hnsw_config(config.model_dump())

    return HNSWConfigUpdateResponse(**result)
