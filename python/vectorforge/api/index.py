"""Index Management Endpoints"""

from typing import Any

from fastapi import APIRouter, HTTPException

from vectorforge.api import engine
from vectorforge.api.decorators import handle_api_errors
from vectorforge.config import VFGConfig
from vectorforge.models import IndexLoadResponse, IndexSaveResponse, IndexStatsResponse


router: APIRouter = APIRouter()

@router.get('/index/stats', response_model=IndexStatsResponse)
@handle_api_errors
def get_index_stats() -> IndexStatsResponse:
    """
    Get quick index statistics
    
    Lightweight endpoint for checking index health and size. Returns essential
    metrics including document counts, embedding dimension, and compaction status.
    For comprehensive metrics, use GET /metrics instead.
    
    Returns:
        IndexStatsResponse: Core index statistics and health indicators
        
    Raises:
        HTTPException: 500 if stats retrieval fails
    """
    stats: dict[str, Any] = engine.get_index_stats()

    return IndexStatsResponse(
        status="success",
        total_documents=stats["total_documents"],
        total_embeddings=stats["total_embeddings"],
        deleted_documents=stats["deleted_documents"],
        deleted_ratio=stats["deleted_ratio"],
        needs_compaction=stats["needs_compaction"],
        embedding_dimension=stats["embedding_dimension"]
    )


@router.post('/index/build', response_model=IndexStatsResponse)
@handle_api_errors
def build_index() -> IndexStatsResponse:
    """
    Build or rebuild the vector index
    
    Reconstructs the entire vector index from scratch. Useful for optimizing
    search performance after many document additions/deletions. This operation
    can be time-consuming for large indexes.
    
    Returns:
        IndexStatsResponse: Updated index statistics after rebuild
        
    Raises:
        HTTPException: 500 if index build fails
    """
    engine.build()
    stats: dict[str, Any] = engine.get_index_stats()

    return IndexStatsResponse(
        status="success",
        total_documents=stats["total_documents"],
        total_embeddings=stats["total_embeddings"],
        deleted_documents=stats["deleted_documents"],
        deleted_ratio=stats["deleted_ratio"],
        needs_compaction=stats["needs_compaction"],
        embedding_dimension=stats["embedding_dimension"]
    )


@router.post('/index/save', response_model=IndexSaveResponse)
@handle_api_errors
def save_index(directory: str = VFGConfig.DEFAULT_DATA_DIR) -> IndexSaveResponse:
    """
    Persist index to disk
    
    Saves the current vector index and all document metadata to the specified
    directory. Creates persistent storage for index recovery and reduces startup time.
    
    Args:
        directory (str, optional): Directory path for saving. Defaults to VFGConfig.DEFAULT_DATA_DIR
        
    Returns:
        IndexSaveResponse: Save confirmation with file sizes and document counts
        
    Raises:
        HTTPException: 500 if save operation fails
        
    Example:
        POST /index/save?directory=/path/to/storage
    """
    save_metrics: dict[str, Any] = engine.save(directory=directory)

    return IndexSaveResponse(
        status=save_metrics["status"],
        directory=save_metrics["directory"],
        metadata_size_mb=save_metrics["metadata_size_mb"],
        embeddings_size_mb=save_metrics["embeddings_size_mb"],
        total_size_mb=save_metrics["total_size_mb"],
        documents_saved=save_metrics["documents_saved"],
        embeddings_saved=save_metrics["embeddings_saved"],
        version=save_metrics["version"]
    )


@router.post('/index/load', response_model=IndexLoadResponse)
@handle_api_errors
def load_index(directory: str = VFGConfig.DEFAULT_DATA_DIR) -> IndexLoadResponse:
    """
    Load index from disk
    
    Restores a previously saved vector index from disk, including all embeddings,
    documents, and metadata. Faster than rebuilding the index from scratch.
    
    Returns:
        IndexLoadResponse: Load confirmation with counts and version information
        
    Raises:
        HTTPException: 404 if index files not found
        HTTPException: 500 if load operation fails
    """
    load_metrics: dict[str, Any] = engine.load(directory=directory)

    return IndexLoadResponse(
        status=load_metrics["status"],
        directory=load_metrics["directory"],
        documents_loaded=load_metrics["documents_loaded"],
        embeddings_loaded=load_metrics["embeddings_loaded"],
        deleted_docs=load_metrics["deleted_docs"],
        version=load_metrics["version"]
    )
