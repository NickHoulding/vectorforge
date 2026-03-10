"""Files feature handlers for the VectorForge demo."""

import os
from typing import Any

from demo import client


def upload() -> None:
    """Prompt for a local file path and optional chunking settings, then upload and index the file."""
    print("\n-- Upload File --")
    collection_name = client.prompt_collection()
    path = client.prompt("Local file path (.pdf or .txt)")
    path = os.path.expanduser(path)

    if not os.path.isfile(path):
        print(f"  File not found: {path}")
        return

    chunk_size = client.prompt_int("Chunk size (chars)")
    chunk_overlap = client.prompt_int("Chunk overlap (chars)")

    data: dict[str, Any] = {}
    if chunk_size is not None:
        data["chunk_size"] = str(chunk_size)
    if chunk_overlap is not None:
        data["chunk_overlap"] = str(chunk_overlap)

    filename = os.path.basename(path)
    with open(path, "rb") as fh:
        file_field = (filename, fh, _mime_type(filename))
        resp = client.post_file(
            f"/collections/{collection_name}/files/upload",
            file_field=file_field,
            data=data,
        )

    client.print_response(resp)


def list_all() -> None:
    """Fetch and display the names of all indexed files in a collection."""
    print("\n-- List Files --")
    collection_name = client.prompt_collection()
    resp = client.get(f"/collections/{collection_name}/files/list")
    client.print_response(resp)


def delete() -> None:
    """Prompt for a filename and delete all of its indexed chunks from a collection."""
    print("\n-- Delete File --")
    collection_name = client.prompt_collection()
    filename = client.prompt("Filename (as stored in the collection)")
    resp = client.delete(f"/collections/{collection_name}/files/{filename}")
    client.print_response(resp)


def _mime_type(filename: str) -> str:
    """Return the MIME type for a supported file extension.

    Args:
      filename: The filename to inspect (only the extension is examined).

    Returns:
      "application/pdf" for .pdf files, "text/plain" for everything else.
    """
    if filename.lower().endswith(".pdf"):
        return "application/pdf"

    return "text/plain"
