"""
VectorForge API Models

Pydantic models for request/response validation organized by domain.
All models are re-exported here for backward compatibility.
"""

# Document models
from .documents import (
    DocumentInput,
    DocumentResponse,
    DocumentDetail
)

# File models
from .files import (
    FileUploadResponse,
    FileDeleteResponse,
    FileListResponse
)

# Search models
from .search import (
    SearchQuery,
    SearchResult,
    SearchResponse
)

# Index models
from .index import (
    IndexStatsResponse,
    IndexSaveResponse,
    IndexLoadResponse
)

# Metrics models
from .metrics import (
    IndexMetrics,
    PerformanceMetrics,
    UsageMetrics,
    MemoryMetrics,
    TimestampMetrics,
    SystemInfo,
    MetricsResponse
)

__all__ = [
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
