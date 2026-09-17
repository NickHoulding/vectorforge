"""MCP tools for managing VectorForge collections."""

import logging
from typing import Any

from ..client import delete, get, post
from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response

logger = logging.getLogger(__name__)


@mcp.tool(
    description="List all collections in VectorForge. Returns collection names, IDs, document counts, and metadata."
)
@handle_tool_errors
def list_collections() -> dict[str, Any]:
    """List all available collections.

    Returns:
      Dictionary with list of collections and their details (name, id, document_count, created_at).
    """
    logger.debug("Listing all collections")
    data = get("/collections")
    collection_count = len(data.get("collections", []))
    logger.info("Listed %d collections", collection_count)
    return build_success_response(data)


@mcp.tool(
    description="Get detailed information about a specific collection including document count and custom metadata."
)
@handle_tool_errors
def get_collection(collection_name: str) -> dict[str, Any]:
    """Retrieve information about a specific collection.

    Args:
      collection_name: Name of the collection to retrieve.

    Returns:
      Dictionary with collection details (name, id, document_count, created_at, metadata).
    """
    logger.debug("Getting collection: name=%s", collection_name)
    data = get(f"/collections/{collection_name}")
    logger.info("Retrieved collection %s", collection_name)
    return build_success_response(data)


@mcp.tool(
    description="Create a new collection for multi-tenancy or domain separation. Optionally add custom metadata."
)
@handle_tool_errors
def create_collection(
    collection_name: str,
    description: str | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create a new collection with optional metadata.

    Args:
      collection_name: Collection name (alphanumeric, underscores, hyphens only).
      description: Optional collection description.
      metadata: Optional custom metadata dictionary (max 20 key-value pairs).

    Returns:
      Dictionary with created collection details.
    """
    logger.debug(
        "Creating collection: name=%s, has_metadata=%s",
        collection_name,
        metadata is not None,
    )

    body: dict[str, Any] = {"name": collection_name}
    if description is not None:
        body["description"] = description
    if metadata is not None:
        body["metadata"] = metadata

    data = post("/collections", json=body)
    logger.info("Created collection %s", collection_name)
    return build_success_response(data)


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
    logger.debug("Deleting collection: name=%s, confirmed=%s", collection_name, confirm)
    data = delete(
        f"/collections/{collection_name}",
        params={"confirm": confirm},
    )
    logger.info("Deleted collection %s", collection_name)
    return build_success_response(data)
