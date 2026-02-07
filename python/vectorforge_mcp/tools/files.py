import os
from typing import Any

from fastapi import UploadFile

from vectorforge.api import files
from vectorforge.models.files import (
    FileDeleteResponse,
    FileListResponse,
    FileUploadResponse,
)

from ..decorators import handle_tool_errors
from ..instance import mcp
from ..utils import build_error_response, build_success_response


@mcp.tool(
    description="Get all filenames that have been uploaded and chunked into the vector index."
)
@handle_tool_errors
def list_files() -> dict[str, Any]:
    """List all indexed files in the vector store.

    Returns:
        List of filenames that have been uploaded and indexed.
    """
    response: FileListResponse = files.list_files()
    return build_success_response(response)


@mcp.tool(
    description="Upload and index a file (PDF, TXT). Extracts text, chunks it, generates embeddings. Returns chunk count and document IDs."
)
@handle_tool_errors
async def upload_file(file_path: str) -> dict[str, Any]:
    """Upload and index a file.

    Args:
        file_path: Absolute path to the file to upload (supports .pdf, .txt).

    Returns:
        Dictionary with upload status, filename, chunks created, and document IDs.
    """
    if not os.path.exists(file_path):
        return build_error_response(FileNotFoundError(f"File not found: {file_path}"))

    with open(file_path, "rb") as f:
        file: UploadFile = UploadFile(filename=os.path.basename(file_path), file=f)
        response: FileUploadResponse = await files.upload_file(file=file)

    return build_success_response(response)


@mcp.tool(
    description="Delete all document chunks from a specific uploaded file. Removes all associated embeddings and metadata."
)
@handle_tool_errors
def delete_file(filename: str) -> dict[str, Any]:
    """Delete all chunks associated with an indexed file.

    Args:
        filename: Name of the source file to delete (exact match).

    Returns:
        Dictionary with deletion status, filename, chunks deleted, and document IDs.
    """
    response: FileDeleteResponse = files.delete_file(filename=filename)
    return build_success_response(response)
