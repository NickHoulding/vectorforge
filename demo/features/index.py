"""Index / HNSW config feature handlers for the VectorForge demo."""

from typing import Any

from demo import client


def stats() -> None:
    """Fetch and display index statistics and HNSW configuration for a collection."""
    print("\n-- Index Stats --")
    collection_name = client.prompt_collection()
    resp = client.get(f"/collections/{collection_name}/stats")
    client.print_response(resp)


def update_hnsw() -> None:
    """Prompt for new HNSW parameters and trigger a full collection migration after confirmation."""
    print("\n-- Update HNSW Config --")
    print("  WARNING: This triggers a full blue-green collection migration")
    print("           and may temporarily require up to 3x disk space.\n")

    collection_name = client.prompt_collection()
    ef_search = client.prompt_int("ef_search")
    max_neighbors = client.prompt_int("max_neighbors")

    config: dict[str, Any] = {}
    if ef_search is not None:
        config["ef_search"] = ef_search
    if max_neighbors is not None:
        config["max_neighbors"] = max_neighbors

    if not config:
        print("  No config values provided, aborting.")
        return

    confirm = client.prompt_bool("Confirm migration?", default=False)
    params = {"confirm": str(confirm).lower()}

    resp = client.put(
        f"/collections/{collection_name}/config/hnsw", body=config, params=params
    )
    client.print_response(resp)
