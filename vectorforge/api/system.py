"""Health check and metrics endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException

from vectorforge import __version__
from vectorforge.api import manager
from vectorforge.api.decorators import handle_api_errors, require_collection
from vectorforge.config import VFGConfig
from vectorforge.models import (
    ChromaDBMetrics,
    IndexMetrics,
    MetricsResponse,
    PerformanceMetrics,
    SystemInfo,
    TimestampMetrics,
    UsageMetrics,
)

router: APIRouter = APIRouter()


@router.get("/health")
@handle_api_errors
def check_health() -> dict[str, Any]:
    """Verify the API is running and return basic health information.

    Returns:
        Dictionary with status, version, chromadb_heartbeat, and total_collections.
    """
    default_engine = manager.get_engine(VFGConfig.DEFAULT_COLLECTION_NAME)
    heartbeat = default_engine.chroma_client.heartbeat()
    collections = manager.list_collections()

    return {
        "status": "healthy",
        "version": __version__,
        "chromadb_heartbeat": heartbeat,
        "total_collections": len(collections),
    }


@router.get("/health/ready")
async def readiness_check() -> dict[str, str | int]:
    """Readiness probe for container orchestration.

    Checks if VectorForge is fully initialized and ready to handle requests.

    Returns:
        Dictionary with status, document count, and model name.

    Raises:
        HTTPException: 503 if the service is not yet ready.
    """
    try:
        default_engine = manager.get_engine(VFGConfig.DEFAULT_COLLECTION_NAME)
        doc_count: int = default_engine.collection.count()

        return {
            "status": "ready",
            "documents": doc_count,
            "model": default_engine.model_name,
        }

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@router.get("/health/live")
async def liveness_check() -> dict[str, str]:
    """Liveness probe for container orchestration.

    Simple check used by Docker/Kubernetes to detect hung processes.

    Returns:
        Dictionary with a static alive status.
    """
    return {"status": "alive"}


@router.get("/collections/{collection_name}/metrics", response_model=MetricsResponse)
@require_collection
@handle_api_errors
def get_collection_metrics(collection_name: str) -> MetricsResponse:
    """Return comprehensive metrics for a specific collection.

    Gathers index statistics, query performance, usage counters, system info,
    and ChromaDB storage details for the named collection.

    Args:
        collection_name: Name of the collection to query.

    Returns:
        MetricsResponse with index, performance, usage, timestamps, system,
        and chromadb sub-objects.

    Raises:
        HTTPException: 404 if the collection does not exist.
        HTTPException: 500 if metrics collection fails.
    """
    engine = manager.get_engine(collection_name)
    metrics: dict[str, Any] = engine.get_metrics()
    chromadb_metrics_dict: dict[str, Any] = engine.get_chromadb_metrics()

    index_metrics: IndexMetrics = IndexMetrics(
        total_documents=metrics["total_documents"],
        total_documents_peak=metrics["total_documents_peak"],
    )
    performance_metrics: PerformanceMetrics = PerformanceMetrics(
        total_queries=metrics["total_queries"],
        avg_query_time_ms=metrics["avg_query_time_ms"],
        total_query_time_ms=metrics["total_query_time_ms"],
        min_query_time_ms=metrics["min_query_time_ms"],
        max_query_time_ms=metrics["max_query_time_ms"],
        p50_query_time_ms=metrics["p50_query_time_ms"],
        p95_query_time_ms=metrics["p95_query_time_ms"],
        p99_query_time_ms=metrics["p99_query_time_ms"],
    )
    usage_metrics: UsageMetrics = UsageMetrics(
        documents_added=metrics["docs_added"],
        documents_deleted=metrics["docs_deleted"],
        chunks_created=metrics["chunks_created"],
        files_uploaded=metrics["files_uploaded"],
        total_doc_size_bytes=metrics["total_doc_size_bytes"],
    )
    timestamp_metrics: TimestampMetrics = TimestampMetrics(
        engine_created_at=metrics["lifetime_created_at"],
        last_query_at=metrics["last_query_at"],
        last_document_added_at=metrics["last_doc_added_at"],
        last_file_uploaded_at=metrics["last_file_uploaded_at"],
    )
    system_info: SystemInfo = SystemInfo(
        model_name=metrics["model_name"],
        model_dimension=metrics["model_dimension"],
        uptime_seconds=metrics["uptime_seconds"],
        version=metrics["version"],
    )
    chromadb_metrics: ChromaDBMetrics = ChromaDBMetrics(
        version=chromadb_metrics_dict["version"],
        collection_id=chromadb_metrics_dict["collection_id"],
        collection_name=chromadb_metrics_dict["collection_name"],
        disk_size_bytes=chromadb_metrics_dict["disk_size_bytes"],
        disk_size_mb=chromadb_metrics_dict["disk_size_mb"],
        persist_directory=chromadb_metrics_dict["persist_directory"],
        max_batch_size=chromadb_metrics_dict["max_batch_size"],
    )

    return MetricsResponse(
        index=index_metrics,
        performance=performance_metrics,
        usage=usage_metrics,
        timestamps=timestamp_metrics,
        system=system_info,
        chromadb=chromadb_metrics,
    )
