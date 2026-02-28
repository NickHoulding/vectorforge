"""Collection Management Endpoints

Provides REST API for creating, listing, and deleting collections.
Enables multi-tenancy and multi-index use cases.
"""

from fastapi import APIRouter, HTTPException, Query, status

from vectorforge.api import manager
from vectorforge.api.decorators import handle_api_errors
from vectorforge.config import VFGConfig
from vectorforge.models import (
    CollectionCreateRequest,
    CollectionCreateResponse,
    CollectionDeleteResponse,
    CollectionInfo,
    CollectionListResponse,
)

router: APIRouter = APIRouter()


@router.post(
    "/collections",
    response_model=CollectionCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
@handle_api_errors
def create_collection(request: CollectionCreateRequest) -> CollectionCreateResponse:
    """
    Create a new collection with custom HNSW configuration

    Creates a new isolated collection for storing documents and embeddings.
    Each collection has its own HNSW index configuration and metadata.

    **Use Cases:**
    - Multi-tenancy: Separate collections per customer/user
    - Domain separation: Different collections for different document types
    - Versioning: Test configurations without affecting production data

    Args:
        request: Collection creation parameters (name, description, HNSW config, metadata)

    Returns:
        CollectionCreateResponse: Information about the created collection

    Raises:
        HTTPException: 400 if name is invalid or collection limit reached
        HTTPException: 409 if collection already exists

    Example:
        ```
        POST /collections
        {
            "name": "customer_docs",
            "description": "Customer documentation",
            "hnsw_config": {"ef_search": 150},
            "metadata": {"tenant": "acme_corp"}
        }
        ```
    """
    try:
        hnsw_config = {}
        if request.hnsw_config:
            hnsw_config = request.hnsw_config.model_dump(exclude_none=True)

        collection_info = manager.create_collection(
            name=request.name,
            hnsw_config=hnsw_config,
            description=request.description,
            metadata=request.metadata or {},
        )

        return CollectionCreateResponse(
            status="success",
            message=f"Collection '{request.name}' created successfully",
            collection=collection_info,
        )

    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=409, detail=str(e))

        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/collections", response_model=CollectionListResponse)
@handle_api_errors
def list_collections() -> CollectionListResponse:
    """
    List all collections with metadata and stats

    Returns comprehensive information about all collections including
    document counts, HNSW configuration, and custom metadata.

    Returns:
        CollectionListResponse: List of all collections with details

    Example:
        ```
        GET /collections
        ```
    """
    collections = manager.list_collections()

    return CollectionListResponse(
        status="success", collections=collections, total=len(collections)
    )


@router.get("/collections/{collection_name}", response_model=CollectionInfo)
@handle_api_errors
def get_collection(collection_name: str) -> CollectionInfo:
    """
    Get detailed information about a specific collection

    Returns comprehensive details including document count, HNSW config,
    creation timestamp, description, and custom metadata.

    Args:
        collection_name: Name of the collection

    Returns:
        CollectionInfo: Detailed collection information

    Raises:
        HTTPException: 404 if collection doesn't exist

    Example:
        ```
        GET /collections/customer_docs
        ```
    """
    try:
        return manager.get_collection_info(collection_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete(
    "/collections/{collection_name}",
    response_model=CollectionDeleteResponse,
)
@handle_api_errors
def delete_collection(
    collection_name: str,
    confirm: bool | None = Query(None, description="Must be 'true' to confirm"),
) -> CollectionDeleteResponse:
    """
    Delete a collection and all its documents

    **Warning:** This is a destructive operation that permanently deletes
    the collection and all documents within it. Always include `?confirm=true`
    to proceed.

    **Default Collection Protection:** Cannot delete the default collection
    ('vectorforge') to prevent accidental data loss.

    Args:
        collection_name: Name of the collection to delete
        confirm: Must be "true" to execute (safety gate)

    Returns:
        CollectionDeleteResponse: Confirmation message

    Raises:
        HTTPException: 400 if confirmation missing
        HTTPException: 404 if collection doesn't exist
        HTTPException: 409 if trying to delete default collection

    Example:
        ```
        DELETE /collections/customer_docs?confirm=true
        ```
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must include ?confirm=true parameter to delete collection",
        )

    if collection_name == VFGConfig.DEFAULT_COLLECTION_NAME:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete default collection '{VFGConfig.DEFAULT_COLLECTION_NAME}'",
        )

    try:
        manager.delete_collection(collection_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return CollectionDeleteResponse(
        status="success",
        message=f"Collection '{collection_name}' deleted successfully",
        collection_name=collection_name,
    )
