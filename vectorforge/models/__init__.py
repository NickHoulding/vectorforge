"""
VectorForge API Models

Pydantic models for request/response validation organized by domain.
All models are re-exported here for backward compatibility.
"""

from .documents import DocumentDetail, DocumentInput, DocumentResponse
from .files import FileDeleteResponse, FileListResponse, FileUploadResponse
from .index import (
    HNSWConfig,
    HNSWConfigUpdate,
    HNSWConfigUpdateResponse,
    IndexStatsResponse,
    MigrationInfo,
)
from .metadata import StandardMetadata, create_metadata
from .metrics import (
    ChromaDBMetrics,
    IndexMetrics,
    MetricsResponse,
    PerformanceMetrics,
    SystemInfo,
    TimestampMetrics,
    UsageMetrics,
)
from .search import SearchQuery, SearchResponse, SearchResult

__all__: list[str] = [
    "DocumentInput",
    "DocumentResponse",
    "DocumentDetail",
    # Files
    "FileUploadResponse",
    "FileDeleteResponse",
    "FileListResponse",
    # Search
    "SearchQuery",
    "SearchResult",
    "SearchResponse",
    # Index
    "IndexStatsResponse",
    "HNSWConfig",
    "HNSWConfigUpdate",
    "HNSWConfigUpdateResponse",
    "MigrationInfo",
    # Metadata
    "StandardMetadata",
    "create_metadata",
    # Metrics
    "IndexMetrics",
    "PerformanceMetrics",
    "UsageMetrics",
    "ChromaDBMetrics",
    "TimestampMetrics",
    "SystemInfo",
    "MetricsResponse",
]
