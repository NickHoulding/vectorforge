"""MCP tools for managing VectorForge collections."""

from typing import Any

from vectorforge.api import collections
from vectorforge.models.collections import (
    CollectionCreateRequest,
    CollectionCreateResponse,
    CollectionDeleteResponse,
    CollectionInfo,
    CollectionListResponse,
)
from vectorforge.models.index import HNSWConfigUpdate

from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="List all collections in VectorForge. Returns collection names, IDs, document counts, and metadata."
)
@handle_tool_errors
def list_collections() -> dict[str, Any]:
    """List all available collections.

    Returns:
        Dictionary with list of collections and their details (name, id, document_count, created_at).
    """
    response: CollectionListResponse = collections.list_collections()
    return build_success_response(response)


@mcp.tool(
    description="Get detailed information about a specific collection including document count, HNSW config, and custom metadata."
)
@handle_tool_errors
def get_collection(collection_name: str) -> dict[str, Any]:
    """Retrieve information about a specific collection.

    Args:
        name: Name of the collection to retrieve.

    Returns:
        Dictionary with collection details (name, id, document_count, created_at, hnsw_config, metadata).
    """
    response: CollectionInfo = collections.get_collection(
        collection_name=collection_name
    )
    return build_success_response(response)


@mcp.tool(
    description="Create a new collection for multi-tenancy or domain separation. Optionally configure HNSW parameters and add custom metadata."
)
@handle_tool_errors
def create_collection(
    collection_name: str,
    description: str | None = None,
    hnsw_space: str | None = None,
    hnsw_ef_construction: int | None = None,
    hnsw_ef_search: int | None = None,
    hnsw_max_neighbors: int | None = None,
    hnsw_resize_factor: float | None = None,
    hnsw_sync_threshold: int | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create a new collection with optional HNSW configuration and metadata.

    Args:
        name: Collection name (alphanumeric, underscores, hyphens only).
        description: Optional collection description.
        hnsw_space: Distance metric ("cosine", "l2", or "ip"). Default: "cosine".
        hnsw_ef_construction: HNSW construction parameter. Default: 100.
        hnsw_ef_search: HNSW search parameter. Default: 100.
        hnsw_max_neighbors: Max neighbors in HNSW graph. Default: 16.
        hnsw_resize_factor: Dynamic index growth factor. Default: 1.2.
        hnsw_sync_threshold: Batch size for persistence. Default: 1000.
        metadata: Optional custom metadata dictionary (max 20 key-value pairs).

    Returns:
        Dictionary with created collection details.
    """
    hnsw_config = None
    if any(
        [
            hnsw_space,
            hnsw_ef_construction,
            hnsw_ef_search,
            hnsw_max_neighbors,
            hnsw_resize_factor,
            hnsw_sync_threshold,
        ]
    ):
        hnsw_config = HNSWConfigUpdate(
            space=hnsw_space,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
            max_neighbors=hnsw_max_neighbors,
            resize_factor=hnsw_resize_factor,
            sync_threshold=hnsw_sync_threshold,
        )

    request = CollectionCreateRequest(
        name=collection_name,
        description=description,
        hnsw_config=hnsw_config,
        metadata=metadata,
    )

    response: CollectionCreateResponse = collections.create_collection(request=request)
    return build_success_response(response)


@mcp.tool(
    description="Permanently delete a collection and all its documents. Requires confirmation. Cannot delete the default 'vectorforge' collection."
)
@handle_tool_errors
def delete_collection(collection_name: str, confirm: bool = False) -> dict[str, Any]:
    """Delete a collection and all its documents.

    Args:
        collection_name: Name of the collection to delete.
        confirm: Must be True to confirm deletion (safety check).

    Returns:
        Dictionary with deletion status and message.
    """
    response: CollectionDeleteResponse = collections.delete_collection(
        collection_name=collection_name, confirm=confirm
    )
    return build_success_response(response)
