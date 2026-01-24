"""
VectorForge API Models

Pydantic models for request/response validation organized by domain.
All models are re-exported here for backward compatibility.
"""

# Document models
from .documents import DocumentDetail, DocumentInput, DocumentResponse

# File models
from .files import FileDeleteResponse, FileListResponse, FileUploadResponse

# Index models
from .index import IndexLoadResponse, IndexSaveResponse, IndexStatsResponse

# Metrics models
from .metrics import (
    IndexMetrics,
    MemoryMetrics,
    MetricsResponse,
    PerformanceMetrics,
    SystemInfo,
    TimestampMetrics,
    UsageMetrics,
)

# Search models
from .search import SearchQuery, SearchResponse, SearchResult


__all__: list[str] = [
    # Documents
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
    "IndexSaveResponse",
    "IndexLoadResponse",
    # Metrics
    "IndexMetrics",
    "PerformanceMetrics",
    "UsageMetrics",
    "MemoryMetrics",
    "TimestampMetrics",
    "SystemInfo",
    "MetricsResponse",
]
