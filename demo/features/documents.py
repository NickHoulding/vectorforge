"""Documents feature handlers for the VectorForge demo."""

from typing import Any

from demo import client


def add() -> None:
    """Prompt for content and metadata, then add a single document to a collection."""
    print("\n-- Add Document --")
    collection_name = client.prompt_collection()
    content = client.prompt("Content (text to embed)")
    metadata = client.prompt_json("Metadata")

    body: dict[str, Any] = {"content": content}
    if metadata:
        body["metadata"] = metadata

    resp = client.post(f"/collections/{collection_name}/documents", body=body)
    client.print_response(resp)


def get() -> None:
    """Prompt for a document ID and display the matching document."""
    print("\n-- Get Document --")
    collection_name = client.prompt_collection()
    doc_id = client.prompt("Document ID")
    resp = client.get(f"/collections/{collection_name}/documents/{doc_id}")
    client.print_response(resp)


def batch_add() -> None:
    """Interactively collect multiple documents and add them in a single batch request."""
    print("\n-- Batch Add Documents --")
    collection_name = client.prompt_collection()
    print("  Enter documents one at a time. Leave content blank to stop.")
    documents = []
    idx = 1

    while True:
        content = client.prompt_optional(f"Document {idx} content")

        if not content:
            break

        metadata = client.prompt_json(f"Document {idx} metadata")
        doc: dict[str, Any] = {"content": content}

        if metadata:
            doc["metadata"] = metadata

        documents.append(doc)
        idx += 1

    if not documents:
        print("  No documents entered, aborting.")
        return

    resp = client.post(
        f"/collections/{collection_name}/documents/batch", body={"documents": documents}
    )
    client.print_response(resp)


def delete() -> None:
    """Prompt for a document ID and delete it from a collection."""
    print("\n-- Delete Document --")
    collection_name = client.prompt_collection()
    doc_id = client.prompt("Document ID")
    resp = client.delete(f"/collections/{collection_name}/documents/{doc_id}")
    client.print_response(resp)


def batch_delete() -> None:
    """Prompt for a comma-separated list of IDs and delete all matching documents."""
    print("\n-- Batch Delete Documents --")
    collection_name = client.prompt_collection()
    raw = client.prompt("Document IDs (comma-separated)")
    ids = [i.strip() for i in raw.split(",") if i.strip()]

    if not ids:
        print("  No IDs provided, aborting.")
        return

    resp = client.delete(f"/collections/{collection_name}/documents", body={"ids": ids})
    client.print_response(resp)
