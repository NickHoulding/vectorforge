"""Pydantic models for request/response validation, organized by domain and re-exported here."""

from .collections import (
    CollectionCreateRequest,
    CollectionCreateResponse,
    CollectionDeleteResponse,
    CollectionInfo,
    CollectionListResponse,
    DocumentListParams,
    DocumentListResponse,
)
from .documents import (
    BatchDeleteInput,
    BatchDocumentInput,
    BatchDocumentResponse,
    DocumentDetail,
    DocumentInput,
)
from .files import FileDeleteResponse, FileListResponse, FileUploadResponse
from .index import IndexStatsResponse
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
    "DocumentDetail",
    "BatchDocumentResponse",
    "BatchDocumentInput",
    "BatchDeleteInput",
    # Collections
    "CollectionCreateRequest",
    "CollectionCreateResponse",
    "CollectionDeleteResponse",
    "CollectionInfo",
    "CollectionListResponse",
    "DocumentListResponse",
    "DocumentListParams",
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
