"""MCP tools for managing VectorForge documents."""

from typing import Any

from ..client import delete, get, post
from ..config import MCPConfig
from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_success_response


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
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      Dictionary with document ID, content, and metadata.
    """
    data = get(f"/collections/{collection_name}/documents/{doc_id}")
    return build_success_response(data)


@mcp.tool(
    description="Index text content for semantic search. Generates embeddings automatically. Optionally add metadata for organization and filtering."
)
@handle_tool_errors
def add_document(
    content: str,
    metadata: dict[str, Any] | None = None,
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Add a single document to the index.

    Args:
      content: The document text content to index (required, non-empty).
      metadata: Optional metadata dictionary (e.g., {"source": "email", "date": "2026-01-20"}).
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      Dictionary with created document ID and status.
    """
    body: dict[str, Any] = {"content": content}
    if metadata is not None:
        body["metadata"] = metadata

    data = post(f"/collections/{collection_name}/documents", json=body)
    return build_success_response(data)


@mcp.tool(
    description="Index multiple documents in a single batch request. More efficient than adding one at a time. All documents are embedded and persisted atomically."
)
@handle_tool_errors
def batch_add_documents(
    documents: list[dict[str, Any]],
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Add multiple documents to the index in one request.

    Each entry in documents must have a 'content' key (str). An optional
    'metadata' key (dict) may also be provided per document.

    Args:
      documents: List of document objects, each with 'content' and optional 'metadata'.
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      Dictionary with list of created document IDs and status.
    """
    data = post(
        f"/collections/{collection_name}/documents/batch",
        json={"documents": documents},
    )
    return build_success_response(data)


@mcp.tool(
    description="Permanently remove a document and its embeddings from the index. Cannot be undone."
)
@handle_tool_errors
def delete_document(
    doc_id: str,
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Delete a single document by ID.

    Args:
      doc_id: Unique document identifier (UUID) to permanently delete.
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      Dictionary with document ID and deletion status.
    """
    data = delete(f"/collections/{collection_name}/documents/{doc_id}")
    return build_success_response(data)


@mcp.tool(
    description="Permanently remove multiple documents and their embeddings in a single request. IDs that do not exist are silently ignored. Cannot be undone."
)
@handle_tool_errors
def batch_delete_documents(
    doc_ids: list[str],
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Delete multiple documents by ID in one request.

    Args:
      doc_ids: List of document UUIDs to permanently delete.
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      Dictionary with list of deleted document IDs and status.
    """
    data = delete(
        f"/collections/{collection_name}/documents",
        json={"ids": doc_ids},
    )
    return build_success_response(data)
