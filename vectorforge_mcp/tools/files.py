"""MCP tools for uploading, listing, and deleting indexed files."""

import logging
import os
from typing import Any

from ..client import delete, get, post
from ..config import MCPConfig
from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_error_response, build_success_response

logger = logging.getLogger(__name__)


@mcp.tool(
    description="Get all filenames that have been uploaded and chunked into the vector index."
)
@handle_tool_errors
def list_files(
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """List all indexed files in the vector store.

    Args:
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      List of filenames that have been uploaded and indexed.
    """
    logger.debug("Listing files: collection=%s", collection_name)
    data = get(f"/collections/{collection_name}/files/list")
    file_count = len(data.get("files", []))
    logger.info("Listed %d files from collection %s", file_count, collection_name)
    return build_success_response(data)


@mcp.tool(
    description="Upload and index a file (PDF, TXT). Extracts text, chunks it, generates embeddings. Returns chunk count and document IDs."
)
@handle_tool_errors
def upload_file(
    file_path: str,
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    """Upload and index a file.

    Args:
      file_path: Absolute path to the file to upload (supports .pdf, .txt).
      collection_name: Name of the collection (defaults to 'vectorforge').
      chunk_size: Maximum characters per chunk (default: 500).
      chunk_overlap: Overlapping characters between chunks (default: 50).

    Returns:
      Dictionary with upload status, filename, chunks created, and document IDs.
    """
    logger.debug(
        "Uploading file: path=%s, collection=%s, chunk_size=%s, chunk_overlap=%s",
        file_path,
        collection_name,
        chunk_size,
        chunk_overlap,
    )

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        return build_error_response(FileNotFoundError(f"File not found: {file_path}"))

    params: dict[str, Any] = {}
    if chunk_size is not None:
        params["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        params["chunk_overlap"] = chunk_overlap

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = post(
            f"/collections/{collection_name}/files/upload",
            params=params if params else None,
            files=files,
        )

    chunk_count = data.get("chunks_created", 0)
    logger.info(
        "Uploaded file %s to collection %s: %d chunks created",
        os.path.basename(file_path),
        collection_name,
        chunk_count,
    )
    return build_success_response(data)


@mcp.tool(
    description="Delete all document chunks from a specific uploaded file. Removes all associated embeddings and metadata."
)
@handle_tool_errors
def delete_file(
    filename: str,
    collection_name: str = MCPConfig.DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Delete all chunks associated with an indexed file.

    Args:
      filename: Name of the source file to delete (exact match).
      collection_name: Name of the collection (defaults to 'vectorforge').

    Returns:
      Dictionary with deletion status, filename, chunks deleted, and document IDs.
    """
    logger.debug("Deleting file: filename=%s, collection=%s", filename, collection_name)
    data = delete(f"/collections/{collection_name}/files/{filename}")
    chunks_deleted = data.get("chunks_deleted", 0)
    logger.info(
        "Deleted file %s from collection %s: %d chunks removed",
        filename,
        collection_name,
        chunks_deleted,
    )
    return build_success_response(data)
