"""FastAPI router for collection management endpoints.

Handles create, list, get, and delete operations for named collections.
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from vectorforge.api import manager
from vectorforge.api.decorators import handle_api_errors, require_collection
from vectorforge.config import VFGConfig
from vectorforge.models import (
    CollectionCreateRequest,
    CollectionCreateResponse,
    CollectionDeleteResponse,
    CollectionInfo,
    CollectionListResponse,
    DocumentDetail,
    DocumentListParams,
    DocumentListResponse,
)
from vectorforge.vector_engine import VectorEngine

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


def _parse_document_filters(filters_raw: str | None) -> dict[str, Any] | None:
    """Parse and validate a JSON-encoded metadata filters query parameter.

    Args:
        filters_raw: Raw JSON string from the query parameter, or ``None``.

    Returns:
        Parsed filters dict, or ``None`` if no filter was provided.

    Raises:
        HTTPException: 422 if the string is not valid JSON, not a JSON object,
            uses an unrecognised operator, or passes a value of the wrong type
            to an operator.
    """
    if filters_raw is None:
        return None

    try:
        parsed = json.loads(filters_raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["query", "filters"],
                    "msg": "filters must be a valid JSON string",
                    "type": "value_error",
                }
            ],
        )

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["query", "filters"],
                    "msg": "filters must be a JSON object",
                    "type": "value_error",
                }
            ],
        )

    valid_scalar_types = VFGConfig.VALID_SCALAR_TYPES
    valid_operators = VFGConfig.VALID_FILTER_OPERATORS

    for field_name, filter_value in parsed.items():
        if not isinstance(filter_value, dict):
            continue

        operators_used = set(filter_value.keys())

        for op in operators_used:
            if op not in valid_operators:
                raise HTTPException(
                    status_code=422,
                    detail=[
                        {
                            "loc": ["query", "filters", field_name],
                            "msg": (
                                f"filters[{field_name!r}] contains unknown operator {op!r}. "
                                f"Allowed operators: {sorted(valid_operators)}"
                            ),
                            "type": "value_error",
                        }
                    ],
                )

        if "$in" in operators_used:
            in_operand = filter_value["$in"]
            if not isinstance(in_operand, list):
                raise HTTPException(
                    status_code=422,
                    detail=[
                        {
                            "loc": ["query", "filters", field_name],
                            "msg": f"filters[{field_name!r}]['$in'] must be a list, got {type(in_operand).__name__!r}",
                            "type": "value_error",
                        }
                    ],
                )
            invalid_items = [
                item for item in in_operand if type(item) not in valid_scalar_types
            ]
            if invalid_items:
                bad_types = {type(item).__name__ for item in invalid_items}
                raise HTTPException(
                    status_code=422,
                    detail=[
                        {
                            "loc": ["query", "filters", field_name],
                            "msg": (
                                f"filters[{field_name!r}]['$in'] contains unsupported value types: "
                                f"{sorted(bad_types)}. Allowed types: str, int, float, bool"
                            ),
                            "type": "value_error",
                        }
                    ],
                )

        for scalar_op in ("$gte", "$lte", "$ne"):
            scalar_operand = filter_value.get(scalar_op)
            if (
                scalar_operand is not None
                and type(scalar_operand) not in valid_scalar_types
            ):
                raise HTTPException(
                    status_code=422,
                    detail=[
                        {
                            "loc": ["query", "filters", field_name],
                            "msg": (
                                f"filters[{field_name!r}][{scalar_op!r}] must be a str, int, float, or bool, "
                                f"got {type(scalar_operand).__name__!r}"
                            ),
                            "type": "value_error",
                        }
                    ],
                )

    return parsed


@router.get(
    "/collections/{collection_name}/documents",
    response_model=DocumentListResponse,
)
@handle_api_errors
@require_collection
def list_documents(
    collection_name: str, params: DocumentListParams = Depends()
) -> DocumentListResponse:
    """List documents in a collection with pagination and optional filtering.

    Returns a paginated list of documents stored in the specified collection,
    including their content and metadata. Supports metadata filtering via a
    JSON-encoded query parameter.

    Args:
        collection_name: Name of the collection to list documents from.
        params: Pagination and filter parameters (limit, offset, filters).

    Returns:
        DocumentListResponse: Paginated list of documents with total count.

    Raises:
        HTTPException: 404 if the collection does not exist.
        HTTPException: 422 if filters is not valid JSON or uses invalid operators.

    Example:
        ```
        GET /collections/customer_docs/documents?limit=50&offset=0&filters={"source":"notes.txt"}
        ```
    """
    filters = _parse_document_filters(params.filters)
    engine: VectorEngine = manager.get_engine(collection_name)
    results: dict[str, Any] = engine.list_documents(
        limit=params.limit, offset=params.offset, filters=filters
    )
    documents: list[DocumentDetail] = [
        DocumentDetail(
            id=results["ids"][i],
            content=results["documents"][i],
            metadata=results["metadatas"][i],
        )
        for i in range(len(results["ids"]))
    ]

    return DocumentListResponse(
        status="success", documents=documents, total=len(results["ids"])
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
