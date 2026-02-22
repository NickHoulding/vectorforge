"""Index Management Endpoints"""

from typing import Any

from fastapi import APIRouter

from vectorforge.api import engine
from vectorforge.api.decorators import handle_api_errors
from vectorforge.config import VFGConfig
from vectorforge.models import IndexLoadResponse, IndexSaveResponse, IndexStatsResponse

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
        total_embeddings=stats["total_embeddings"],
        embedding_dimension=stats["embedding_dimension"],
    )


@router.post("/index/save", response_model=IndexSaveResponse)
@handle_api_errors
def save_index(directory: str = VFGConfig.DEFAULT_DATA_DIR) -> IndexSaveResponse:
    """
    Get persistent storage information

    ChromaDB automatically persists all data to disk. This endpoint provides
    information about the persisted data location and current statistics.

    Args:
        directory: Target directory path for ChromaDB data

    Returns:
        IndexSaveResponse: Storage information with file sizes and document counts

    Raises:
        HTTPException: 500 if operation fails

    Example:
        POST /index/save
    """
    save_metrics: dict[str, Any] = engine.save(directory=directory)

    return IndexSaveResponse(
        status=save_metrics["status"],
        directory=save_metrics["directory"],
        documents_saved=save_metrics["documents_saved"],
        embeddings_saved=save_metrics["embeddings_saved"],
        version=save_metrics["version"],
    )


@router.post("/index/load", response_model=IndexLoadResponse)
@handle_api_errors
def load_index(directory: str = VFGConfig.DEFAULT_DATA_DIR) -> IndexLoadResponse:
    """
    Get current loaded data information

    ChromaDB automatically loads data on initialization. This endpoint returns
    information about the currently loaded data.

    Args:
        directory: Target directory path for ChromaDB data

    Returns:
        IndexLoadResponse: Current data information with counts and version

    Raises:
        HTTPException: 500 if operation fails
    """
    load_metrics: dict[str, Any] = engine.load(directory=directory)

    return IndexLoadResponse(
        status=load_metrics["status"],
        directory=load_metrics["directory"],
        documents_loaded=load_metrics["documents_loaded"],
        embeddings_loaded=load_metrics["embeddings_loaded"],
        version=load_metrics["version"],
    )
