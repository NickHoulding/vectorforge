"""Index feature handlers for the VectorForge demo."""

from demo import client


def stats() -> None:
    """Fetch and display index statistics for a collection."""
    print("\n-- Index Stats --")
    collection_name = client.prompt_collection()
    resp = client.get(f"/collections/{collection_name}/stats")
    client.print_response(resp)
