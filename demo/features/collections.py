"""Collection management feature handlers for the VectorForge demo."""

from typing import Any

from demo import client


def create() -> None:
    """Prompt for collection details and create a new collection."""
    print("\n-- Create Collection --")
    name = client.prompt("Collection name")
    description = client.prompt_optional("Description")
    hnsw_config = client.prompt_json("HNSW config")
    metadata = client.prompt_json("Metadata")

    body: dict[str, Any] = {"name": name}
    if description:
        body["description"] = description
    if hnsw_config:
        body["hnsw_config"] = hnsw_config
    if metadata:
        body["metadata"] = metadata

    resp = client.post("/collections", body=body)
    client.print_response(resp)


def list_all() -> None:
    """Fetch and display all collections."""
    print("\n-- List Collections --")
    resp = client.get("/collections")
    client.print_response(resp)


def get() -> None:
    """Prompt for a collection name and display its details."""
    print("\n-- Get Collection --")
    collection_name = client.prompt("Collection name")
    resp = client.get(f"/collections/{collection_name}")
    client.print_response(resp)


def delete() -> None:
    """Prompt for a collection name and delete it after confirmation."""
    print("\n-- Delete Collection --")
    collection_name = client.prompt("Collection name")
    confirm = client.prompt_bool("Confirm deletion?", default=False)
    params = {"confirm": str(confirm).lower()}
    resp = client.delete(f"/collections/{collection_name}", params=params)
    client.print_response(resp)
