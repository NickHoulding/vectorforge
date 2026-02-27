"""Pydantic models for collection management."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from vectorforge.config import VFGConfig

from .index import HNSWConfig, HNSWConfigUpdate


class CollectionCreateRequest(BaseModel):
    """Request to create a new collection.

    Attributes:
        name: Collection name (alphanumeric, underscores, hyphens only)
        description: Optional human-readable description
        hnsw_config: Optional HNSW index configuration (uses defaults if not provided)
        metadata: Optional custom metadata (up to 20 key-value pairs)
    """

    name: str = Field(
        ...,
        min_length=VFGConfig.MIN_COLLECTION_NAME_LENGTH,
        max_length=VFGConfig.MAX_COLLECTION_NAME_LENGTH,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Collection name (alphanumeric, _, - only)",
    )
    description: Optional[str] = Field(
        None,
        max_length=VFGConfig.MAX_DESCRIPTION_LENGTH,
        description="Human-readable description",
    )
    hnsw_config: Optional[HNSWConfigUpdate] = Field(
        None, description="HNSW index configuration"
    )
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Custom metadata key-value pairs"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "customer_docs",
                "description": "Customer documentation collection",
                "hnsw_config": {"ef_search": 150, "max_neighbors": 32},
                "metadata": {"tenant": "acme_corp", "version": "1.0"},
            }
        }
    )


class CollectionInfo(BaseModel):
    """Collection information and metadata.

    Attributes:
        name: Collection name
        id: ChromaDB collection UUID
        document_count: Number of documents in the collection
        created_at: ISO timestamp when collection was created
        description: Optional human-readable description
        hnsw_config: HNSW index configuration
        metadata: Custom metadata key-value pairs
    """

    collection_name: str = Field(..., description="Collection name")
    id: str = Field(..., description="ChromaDB collection UUID")
    document_count: int = Field(..., description="Number of documents", ge=0)
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    description: Optional[str] = Field(None, description="Human-readable description")
    hnsw_config: HNSWConfig = Field(..., description="HNSW index configuration")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection_name": "customer_docs",
                "id": "7029ba34-0788-4798-b82c-512bf3037c99",
                "document_count": 1500,
                "created_at": "2026-02-26T12:00:00Z",
                "description": "Customer documentation collection",
                "hnsw_config": {
                    "space": "cosine",
                    "ef_construction": 100,
                    "ef_search": 150,
                    "max_neighbors": 32,
                    "resize_factor": 1.2,
                    "sync_threshold": 1000,
                },
                "metadata": {"tenant": "acme_corp", "version": "1.0"},
            }
        }
    )


class CollectionListResponse(BaseModel):
    """Response containing list of all collections.

    Attributes:
        status: Operation status (always "success" for 200 response)
        collections: List of collection information objects
        total: Total number of collections
    """

    status: str = Field(..., description="Operation status")
    collections: list[CollectionInfo] = Field(..., description="List of collections")
    total: int = Field(..., description="Total number of collections", ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "collections": [
                    {
                        "collection_name": "vectorforge",
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_count": 1250,
                        "created_at": "2026-01-15T10:00:00Z",
                        "description": "Default collection",
                        "hnsw_config": {
                            "space": "cosine",
                            "ef_construction": 100,
                            "ef_search": 100,
                            "max_neighbors": 16,
                            "resize_factor": 1.2,
                            "sync_threshold": 1000,
                        },
                        "metadata": {},
                    },
                    {
                        "collection_name": "customer_docs",
                        "id": "7029ba34-0788-4798-b82c-512bf3037c99",
                        "document_count": 1500,
                        "created_at": "2026-02-26T12:00:00Z",
                        "description": "Customer documentation",
                        "hnsw_config": {
                            "space": "cosine",
                            "ef_construction": 100,
                            "ef_search": 150,
                            "max_neighbors": 32,
                            "resize_factor": 1.2,
                            "sync_threshold": 1000,
                        },
                        "metadata": {"tenant": "acme_corp"},
                    },
                ],
                "total": 2,
            }
        }
    )


class CollectionCreateResponse(BaseModel):
    """Response from collection creation.

    Attributes:
        status: Operation status (always "success" for 201 response)
        message: Human-readable success message
        collection: Information about the created collection
    """

    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Success message")
    collection: CollectionInfo = Field(..., description="Created collection info")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Collection 'customer_docs' created successfully",
                "collection": {
                    "collection_name": "customer_docs",
                    "id": "7029ba34-0788-4798-b82c-512bf3037c99",
                    "document_count": 0,
                    "created_at": "2026-02-26T12:00:00Z",
                    "description": "Customer documentation",
                    "hnsw_config": {
                        "space": "cosine",
                        "ef_construction": 100,
                        "ef_search": 150,
                        "max_neighbors": 32,
                        "resize_factor": 1.2,
                        "sync_threshold": 1000,
                    },
                    "metadata": {"tenant": "acme_corp"},
                },
            }
        }
    )


class CollectionDeleteResponse(BaseModel):
    """Response from collection deletion.

    Attributes:
        status: Operation status (always "success" for 200 response)
        message: Human-readable success message
        collection_name: Name of the deleted collection
    """

    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Success message")
    collection_name: str = Field(..., description="Deleted collection name")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Collection 'customer_docs' deleted successfully",
                "collection_name": "customer_docs",
            }
        }
    )
