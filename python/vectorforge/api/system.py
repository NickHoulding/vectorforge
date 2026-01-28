"""Health and Metrics Endpoints"""

from typing import Any

from fastapi import APIRouter

from vectorforge import __version__
from vectorforge.api import engine
from vectorforge.models import (
    IndexMetrics,
    MemoryMetrics,
    MetricsResponse,
    PerformanceMetrics,
    SystemInfo,
    TimestampMetrics,
    UsageMetrics,
)


router: APIRouter = APIRouter()

@router.get('/health')
def check_health() -> dict[str, Any]:
    """
    API health check
    
    Simple endpoint to verify the API is running and responsive. Returns a
    success status for monitoring and load balancer health checks.
    
    Returns:
        dict: Health status object
        
    Example Response:
        ```json
        {
            "status": "healthy",
            "version": "0.9.0"
        }
        ```
    """
    return {
        "status": "healthy",
        "version": __version__
    }


@router.get('/metrics', response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """
    Get comprehensive system metrics
    
    Returns detailed performance, usage, and system statistics including:
    - Index statistics (documents, embeddings, compaction status)
    - Performance metrics (query times, percentiles)
    - Usage statistics (operations performed)
    - Memory consumption
    - System information and uptime
    
    This endpoint provides complete observability into the vector engine's state
    and performance characteristics.
    
    Returns:
        MetricsResponse: Comprehensive metrics across all categories
        
    Raises:
        HTTPException: 500 if metrics collection fails
    """
    metrics: dict[str, Any] = engine.get_metrics()
    
    index_metrics: IndexMetrics = IndexMetrics(
        total_documents=metrics["active_documents"],
        total_embeddings=metrics["total_embeddings"],
        deleted_documents=metrics["docs_deleted"],
        deleted_ratio=metrics["deleted_ratio"],
        needs_compaction=metrics["needs_compaction"],
        compact_threshold=metrics["compact_threshold"]
    )
    performance_metrics: PerformanceMetrics = PerformanceMetrics(
        total_queries=metrics["total_queries"],
        avg_query_time_ms=metrics["avg_query_time_ms"],
        total_query_time_ms=metrics["total_query_time_ms"],
        min_query_time_ms=metrics["min_query_time_ms"],
        max_query_time_ms=metrics["max_query_time_ms"],
        p50_query_time_ms=metrics["p50_query_time_ms"],
        p95_query_time_ms=metrics["p95_query_time_ms"],
        p99_query_time_ms=metrics["p99_query_time_ms"]
    )
    usage_metrics: UsageMetrics = UsageMetrics(
        documents_added=metrics["docs_added"],
        documents_deleted=metrics["docs_deleted"],
        compactions_performed=metrics["compactions_performed"],
        chunks_created=metrics["chunks_created"],
        files_uploaded=metrics["files_uploaded"]
    )
    memory_metrics: MemoryMetrics = MemoryMetrics(
        embeddings_mb=metrics["embeddings_mb"],
        documents_mb=metrics["documents_mb"],
        total_mb=metrics["total_mb"]
    )
    timestamp_metrics: TimestampMetrics = TimestampMetrics(
        engine_created_at=metrics["created_at"],
        last_query_at=metrics["last_query_at"],
        last_document_added_at=metrics["last_doc_added_at"],
        last_compaction_at=metrics["last_compaction_at"],
        last_file_uploaded_at=metrics["last_file_uploaded_at"]
    )
    system_info: SystemInfo = SystemInfo(
        model_name=metrics["model_name"],
        model_dimension=metrics["model_dimension"],
        uptime_seconds=metrics["uptime_seconds"],
        version=metrics["version"]
    )

    return MetricsResponse(
        index=index_metrics,
        performance=performance_metrics,
        usage=usage_metrics,
        memory=memory_metrics,
        timestamps=timestamp_metrics,
        system=system_info
    )
