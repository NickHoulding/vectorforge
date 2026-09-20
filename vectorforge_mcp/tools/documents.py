"""MCP tools for managing VectorForge documents."""

import logging
from typing import Any

from ..client import delete, get, post
from ..config import MCPConfig
from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response

logger = logging.getLogger(__name__)


@mcp.tool(
    description="Fetch document content and metadata by ID. Use to verify stored content, inspect search results, or retrieve metadata."
)
@handle_tool_errors
def get_document(
    doc_id: str,
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Retrieve a single document by ID.

    Args:
      doc_id: Unique document identifier (UUID).
      collection_name: Name of the collection. Defaults to
        ``MCPConfig.DEFAULT_COLLECTION_NAME``.

    Returns:
      Dictionary with document ID, content, and metadata.
    """
    logger.debug("Getting document: doc_id=%s, collection=%s", doc_id, collection_name)
    data = get(f"/collections/{collection_name}/documents/{doc_id}")
    logger.info("Retrieved document %s from collection %s", doc_id, collection_name)
    return build_success_response(data)


@mcp.tool(
    description="Index one or more text documents for semantic search. Generates embeddings automatically. All documents are embedded and persisted atomically."
)
@handle_tool_errors
def add_documents(
    documents: list[dict[str, Any]],
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Add one or more documents to the index in one request.

    Each entry in documents must have a 'content' key (str). An optional
    'metadata' key (dict) may also be provided per document.

    Args:
      documents: List of document objects, each with 'content' and optional 'metadata'.
        Pass a single-item list to add just one document.
      collection_name: Name of the collection. Defaults to
        ``MCPConfig.DEFAULT_COLLECTION_NAME``.

    Returns:
      Dictionary with list of created document IDs and status.
    """
    logger.debug(
        "Adding documents: count=%d, collection=%s",
        len(documents),
        collection_name,
    )
    data = post(
        f"/collections/{collection_name}/documents",
        json={"documents": documents},
    )
    logger.info("Added %d documents to collection %s", len(documents), collection_name)
    return build_success_response(data)


@mcp.tool(
    description="Permanently remove one or more documents and their embeddings from the index in a single request. IDs that do not exist are silently ignored. Cannot be undone."
)
@handle_tool_errors
def delete_documents(
    doc_ids: list[str],
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Delete one or more documents by ID in one request.

    Args:
      doc_ids: List of document UUIDs to permanently delete. Pass a
        single-item list to delete just one document.
      collection_name: Name of the collection. Defaults to
        ``MCPConfig.DEFAULT_COLLECTION_NAME``.

    Returns:
      Dictionary with list of deleted document IDs and status.
    """
    logger.debug(
        "Deleting documents: count=%d, collection=%s",
        len(doc_ids),
        collection_name,
    )
    data = delete(
        f"/collections/{collection_name}/documents",
        json={"ids": doc_ids},
    )
    logger.info(
        "Deleted %d documents from collection %s", len(doc_ids), collection_name
    )
    return build_success_response(data)
