"""Search feature handler for the VectorForge demo."""

from typing import Any

from demo import client


def search() -> None:
    """POST /collections/{name}/search — semantic similarity search."""
    print("\n-- Search --")
    collection_name = client.prompt("Collection name", default="vectorforge")
    query = client.prompt("Query text")
    top_k = client.prompt_int("top_k (number of results)", default=5)
    filters = client.prompt_json("Metadata filter (where)")
    document_filter = client.prompt_json("Document filter (where_document)")

    body: dict[str, Any] = {"query": query}
    if top_k is not None:
        body["top_k"] = top_k
    if filters:
        body["filters"] = filters
    if document_filter:
        body["document_filter"] = document_filter

    resp = client.post(f"/collections/{collection_name}/search", body=body)
    client.print_response(resp)
