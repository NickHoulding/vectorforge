from typing import Optional

from pydantic import BaseModel, Field


class HNSWConfig(BaseModel):
    """HNSW index configuration parameters.

    Hierarchical Navigable Small World (HNSW) is the indexing algorithm used
    by ChromaDB for efficient approximate nearest neighbor search.

    Attributes:
        space: Distance metric used for similarity (e.g., 'cosine', 'l2', 'ip').
        ef_construction: Size of the dynamic candidate list for index construction.
                        Higher values improve index quality but slow construction.
        ef_search: Size of the dynamic candidate list for search operations.
                  Higher values improve search accuracy but slow queries.
        max_neighbors: Maximum number of connections per node in the graph.
                      Higher values improve recall but increase memory usage.
        resize_factor: Factor by which to grow the index when capacity is reached.
        sync_threshold: Number of operations before forcing index persistence.
    """

    space: str = Field(..., description="Distance metric (cosine, l2, ip)")
    ef_construction: int = Field(..., description="Index construction parameter")
    ef_search: int = Field(..., description="Search quality parameter")
    max_neighbors: int = Field(..., description="Max connections per node")
    resize_factor: float = Field(..., description="Index growth factor")
    sync_threshold: int = Field(..., description="Persistence threshold")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "space": "cosine",
                "ef_construction": 100,
                "ef_search": 100,
                "max_neighbors": 16,
                "resize_factor": 1.2,
                "sync_threshold": 1000,
            }
        }


class IndexStatsResponse(BaseModel):
    """Lightweight index health and statistics.

    Provides essential metrics about the current state of the vector index.

    Attributes:
        total_documents: Number of active documents in the index.
        embedding_dimension: Dimensionality of the embedding vectors.
        hnsw_config: HNSW index configuration parameters.
    """

    status: str = Field(..., description="Operation status")
    total_documents: int = Field(..., description="Active documents")
    embedding_dimension: int = Field(..., description="Embedding size")
    hnsw_config: HNSWConfig = Field(..., description="HNSW index configuration")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "success",
                "total_documents": 1250,
                "embedding_dimension": 384,
                "hnsw_config": {
                    "space": "cosine",
                    "ef_construction": 100,
                    "ef_search": 100,
                    "max_neighbors": 16,
                    "resize_factor": 1.2,
                    "sync_threshold": 1000,
                },
            }
        }


class HNSWConfigUpdate(BaseModel):
    """Request model for updating HNSW configuration.

    All fields are optional. Unspecified fields use ChromaDB defaults.
    Note: Updating HNSW config requires full collection recreation with
    zero-downtime collection-level migration. All collections share the
    same ChromaDB database and persistent storage volume.

    Attributes:
        space: Distance metric (default: "cosine")
        ef_construction: Build-time search depth (default: 100)
        ef_search: Query-time search depth (default: 100)
        max_neighbors: Maximum connections per node (default: 16)
        resize_factor: Dynamic index growth factor (default: 1.2)
        sync_threshold: Batch size for persistence (default: 1000)
    """

    space: Optional[str] = Field(
        default="cosine", description="Distance metric (cosine, l2, ip)"
    )
    ef_construction: Optional[int] = Field(
        default=100, description="Build-time search depth"
    )
    ef_search: Optional[int] = Field(default=100, description="Query-time search depth")
    max_neighbors: Optional[int] = Field(
        default=16, description="Maximum connections per node (M)"
    )
    resize_factor: Optional[float] = Field(
        default=1.2, description="Dynamic index growth factor"
    )
    sync_threshold: Optional[int] = Field(
        default=1000, description="Batch size for persistence"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "ef_search": 150,
                "max_neighbors": 32,
            }
        }


class MigrationInfo(BaseModel):
    """Migration statistics for HNSW config update.

    Attributes:
        documents_migrated: Number of documents migrated to new collection
        time_taken_seconds: Total migration time in seconds
        old_collection_deleted: Whether old collection was cleaned up
    """

    documents_migrated: int = Field(
        ..., description="Number of documents migrated", ge=0
    )
    time_taken_seconds: float = Field(..., description="Total migration time", ge=0.0)
    old_collection_deleted: bool = Field(
        ..., description="Whether old collection was cleaned up"
    )


class HNSWConfigUpdateResponse(BaseModel):
    """Response from HNSW configuration update.

    Attributes:
        status: Operation status (always "success" for 200 response)
        message: Human-readable success message
        migration: Migration statistics
        config: New HNSW configuration after update
    """

    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Human-readable message")
    migration: MigrationInfo = Field(..., description="Migration statistics")
    config: HNSWConfig = Field(..., description="New HNSW configuration")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "HNSW configuration updated successfully",
                "migration": {
                    "documents_migrated": 1500,
                    "time_taken_seconds": 12.34,
                    "old_collection_deleted": True,
                },
                "config": {
                    "space": "cosine",
                    "ef_construction": 200,
                    "ef_search": 150,
                    "max_neighbors": 32,
                    "resize_factor": 1.5,
                    "sync_threshold": 2000,
                },
            }
        }
