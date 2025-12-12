from pydantic import BaseModel, Field
from typing import Optional


class IndexMetrics(BaseModel):
    """Index statistics"""
    total_documents: int = Field(..., description="Active documents in index")
    total_embeddings: int = Field(..., description="Total embeddings including deleted")
    deleted_documents: int = Field(..., description="Number of deleted documents")
    deleted_ratio: float = Field(..., ge=0, le=1, description="Ratio of deleted to total")
    needs_compaction: bool = Field(..., description="Whether compaction is recommended")
    compact_threshold: float = Field(..., ge=0, le=1, description="Compaction trigger threshold")

class PerformanceMetrics(BaseModel):
    """Query performance statistics"""
    total_queries: int = Field(..., description="Total queries executed")
    avg_query_time_ms: float = Field(..., ge=0, description="Average query latency")
    total_query_time_ms: float = Field(..., ge=0, description="Cumulative query time")
    min_query_time_ms: Optional[float] = Field(None, ge=0, description="Fastest query")
    max_query_time_ms: Optional[float] = Field(None, ge=0, description="Slowest query")
    p50_query_time_ms: Optional[float] = Field(None, ge=0, description="Median query time")
    p95_query_time_ms: Optional[float] = Field(None, ge=0, description="95th percentile")
    p99_query_time_ms: Optional[float] = Field(None, ge=0, description="99th percentile")

class UsageMetrics(BaseModel):
    """System usage statistics"""
    documents_added: int = Field(..., description="Total documents added")
    documents_deleted: int = Field(..., description="Total documents deleted")
    compactions_performed: int = Field(..., description="Number of compactions")
    chunks_created: int = Field(0, description="Total chunks from file uploads")
    files_uploaded: int = Field(0, description="Total files uploaded")

class MemoryMetrics(BaseModel):
    """Memory usage estimates"""
    embeddings_mb: float = Field(..., ge=0, description="Memory used by embeddings")
    documents_mb: float = Field(..., ge=0, description="Memory used by documents")
    total_mb: float = Field(..., ge=0, description="Total memory usage")

class TimestampMetrics(BaseModel):
    """Important timestamps"""
    engine_created_at: str = Field(..., description="Engine initialization time")
    last_query_at: Optional[str] = Field(None, description="Most recent query")
    last_document_added_at: Optional[str] = Field(None, description="Most recent addition")
    last_compaction_at: Optional[str] = Field(None, description="Most recent compaction")
    last_file_uploaded_at: Optional[str] = Field(None, description="Most recent file upload")

class SystemInfo(BaseModel):
    """System configuration"""
    model_name: str = Field(..., description="Embedding model name")
    model_dimension: int = Field(..., description="Embedding dimension")
    uptime_seconds: Optional[float] = Field(None, ge=0, description="Engine uptime")

class MetricsResponse(BaseModel):
    """Complete metrics response"""
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
                    "total_documents": 1250,
                    "total_embeddings": 1500,
                    "deleted_documents": 250,
                    "deleted_ratio": 0.167,
                    "needs_compaction": False,
                    "compact_threshold": 0.3
                },
                "performance": {
                    "total_queries": 5420,
                    "avg_query_time_ms": 12.5,
                    "total_query_time_ms": 67750.0,
                    "min_query_time_ms": 3.2,
                    "max_query_time_ms": 145.8,
                    "p50_query_time_ms": 10.1,
                    "p95_query_time_ms": 28.4,
                    "p99_query_time_ms": 52.7
                },
                "usage": {
                    "documents_added": 2100,
                    "documents_deleted": 850,
                    "compactions_performed": 3,
                    "chunks_created": 1875,
                    "files_uploaded": 42
                },
                "memory": {
                    "embeddings_mb": 2.304,
                    "documents_mb": 0.512,
                    "total_mb": 2.816
                },
                "timestamps": {
                    "engine_created_at": "2024-01-15T10:30:00Z",
                    "last_query_at": "2024-01-15T16:45:23Z",
                    "last_document_added_at": "2024-01-15T15:20:10Z",
                    "last_compaction_at": "2024-01-15T14:00:00Z",
                    "last_file_uploaded_at": "2024-01-15T15:15:42Z"
                },
                "system": {
                    "model_name": "all-MiniLM-L6-v2",
                    "model_dimension": 384,
                    "uptime_seconds": 22523.45
                }
            }
        }
