from typing import Optional

from pydantic import BaseModel, Field


class IndexMetrics(BaseModel):
    """Index health metrics.

    Statistics about the vector index state with ChromaDB backend.

    Attributes:
        total_documents: Active documents currently in the index.
    """

    total_documents: int = Field(..., description="Active documents in index")


class PerformanceMetrics(BaseModel):
    """Query execution performance statistics.

    Comprehensive metrics about search query performance, including average,
    minimum, maximum, and percentile latencies. Useful for monitoring and
    optimizing search performance.

    Attributes:
        total_queries: Cumulative count of search queries executed.
        avg_query_time_ms: Mean query execution time in milliseconds.
        total_query_time_ms: Sum of all query execution times.
        min_query_time_ms: Fastest query execution time (if queries exist).
        max_query_time_ms: Slowest query execution time (if queries exist).
        p50_query_time_ms: Median query time (50th percentile).
        p95_query_time_ms: 95th percentile query time.
        p99_query_time_ms: 99th percentile query time.
    """

    total_queries: int = Field(..., description="Total queries executed")
    avg_query_time_ms: float = Field(..., ge=0, description="Average query latency")
    total_query_time_ms: float = Field(..., ge=0, description="Cumulative query time")
    min_query_time_ms: Optional[float] = Field(None, ge=0, description="Fastest query")
    max_query_time_ms: Optional[float] = Field(None, ge=0, description="Slowest query")
    p50_query_time_ms: Optional[float] = Field(
        None, ge=0, description="Median query time"
    )
    p95_query_time_ms: Optional[float] = Field(
        None, ge=0, description="95th percentile"
    )
    p99_query_time_ms: Optional[float] = Field(
        None, ge=0, description="99th percentile"
    )


class UsageMetrics(BaseModel):
    """System operation and usage statistics.

    Tracks the volume of various operations performed on the vector database.
    With ChromaDB, compaction is handled internally so compaction counts are
    not tracked.

    Attributes:
        documents_added: Total number of documents added to the index.
        documents_deleted: Total number of documents deleted from the index.
        chunks_created: Total document chunks created from file uploads.
        files_uploaded: Total number of files processed and indexed.
    """

    documents_added: int = Field(..., description="Total documents added")
    documents_deleted: int = Field(..., description="Total documents deleted")
    chunks_created: int = Field(0, description="Total chunks from file uploads")
    files_uploaded: int = Field(0, description="Total files uploaded")


class MemoryMetrics(BaseModel):
    """Memory consumption estimates.

    Provides approximate memory usage for the vector database components,
    useful for capacity planning and resource monitoring.

    Attributes:
        embeddings_mb: Estimated memory used by embedding vectors in megabytes.
        documents_mb: Estimated memory used by document content in megabytes.
        total_mb: Combined estimated memory usage in megabytes.
    """

    embeddings_mb: float = Field(..., ge=0, description="Memory used by embeddings")
    documents_mb: float = Field(..., ge=0, description="Memory used by documents")
    total_mb: float = Field(..., ge=0, description="Total memory usage")


class TimestampMetrics(BaseModel):
    """Event timestamps for monitoring and debugging.

    Tracks when key operations occurred in the vector database lifecycle.
    With ChromaDB, compaction is handled internally so compaction timestamps
    are not tracked.

    Attributes:
        engine_created_at: ISO timestamp when the engine was initialized.
        last_query_at: ISO timestamp of most recent search query (if any).
        last_document_added_at: ISO timestamp of most recent document addition (if any).
        last_file_uploaded_at: ISO timestamp of most recent file upload (if any).
    """

    engine_created_at: str = Field(..., description="Engine initialization time")
    last_query_at: Optional[str] = Field(None, description="Most recent query")
    last_document_added_at: Optional[str] = Field(
        None, description="Most recent addition"
    )
    last_file_uploaded_at: Optional[str] = Field(
        None, description="Most recent file upload"
    )


class SystemInfo(BaseModel):
    """System configuration and runtime information.

    Provides details about the vector database configuration including the
    embedding model, dimensionality, version, and uptime.

    Attributes:
        model_name: Name of the sentence transformer model being used.
        model_dimension: Dimensionality of the embedding vectors.
        uptime_seconds: Time elapsed since engine initialization (if available).
        version: VectorForge version number.
    """

    model_name: str = Field(..., description="Embedding model name")
    model_dimension: int = Field(..., description="Embedding dimension")
    uptime_seconds: Optional[float] = Field(None, ge=0, description="Engine uptime")
    version: str = Field(..., description="vectorforge version")


class MetricsResponse(BaseModel):
    """Comprehensive metrics response with all system statistics.

    Complete observability data for the vector database, organized into logical
    categories. This is the primary metrics endpoint providing detailed insights
    into index health, query performance, resource usage, and system state.

    Attributes:
        index: Index health and compaction metrics.
        performance: Query execution performance statistics.
        usage: Operation counts and usage patterns.
        memory: Memory consumption estimates.
        timestamps: Event timing information.
        system: System configuration and version details.
    """

    index: IndexMetrics
    performance: PerformanceMetrics
    usage: UsageMetrics
    memory: MemoryMetrics
    timestamps: TimestampMetrics
    system: SystemInfo

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "index": {
                    "total_documents": 1500,
                },
                "performance": {
                    "total_queries": 5420,
                    "avg_query_time_ms": 12.5,
                    "total_query_time_ms": 67750.0,
                    "min_query_time_ms": 3.2,
                    "max_query_time_ms": 145.8,
                    "p50_query_time_ms": 10.1,
                    "p95_query_time_ms": 28.4,
                    "p99_query_time_ms": 52.7,
                },
                "usage": {
                    "documents_added": 2100,
                    "documents_deleted": 850,
                    "chunks_created": 1875,
                    "files_uploaded": 42,
                },
                "memory": {
                    "embeddings_mb": 2.304,
                    "documents_mb": 0.512,
                    "total_mb": 2.816,
                },
                "timestamps": {
                    "engine_created_at": "2024-01-15T10:30:00Z",
                    "last_query_at": "2024-01-15T16:45:23Z",
                    "last_document_added_at": "2024-01-15T15:20:10Z",
                    "last_file_uploaded_at": "2024-01-15T15:15:42Z",
                },
                "system": {
                    "model_name": "all-MiniLM-L6-v2",
                    "model_dimension": 384,
                    "uptime_seconds": 22523.45,
                    "version": "0.9.0",
                },
            }
        }
