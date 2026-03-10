"""Feature registry mapping user-facing keys to their handler functions.

Each key follows the pattern <group>:<action> and corresponds to one API endpoint.
"""

from typing import Any, Callable

from demo.features import (
    collections,
    documents,
    files,
    index,
    search,
    system,
)

FEATURES: dict[str, tuple[Callable[..., Any], str]] = {
    # key: (handler, one-line description)
    "collections:create": (collections.create, "Create a new collection"),
    "collections:list": (collections.list_all, "List all collections"),
    "collections:get": (collections.get, "Get details for one collection"),
    "collections:delete": (collections.delete, "Delete a collection"),
    "documents:add": (documents.add, "Add a single document"),
    "documents:get": (documents.get, "Fetch a document by ID"),
    "documents:batch_add": (documents.batch_add, "Add multiple documents"),
    "documents:delete": (documents.delete, "Delete a document by ID"),
    "documents:batch_delete": (
        documents.batch_delete,
        "Batch-delete documents by ID list",
    ),
    "files:upload": (files.upload, "Upload and index a .pdf or .txt file"),
    "files:list": (files.list_all, "List indexed files in a collection"),
    "files:delete": (files.delete, "Delete all chunks for a file"),
    "search": (search.search, "Semantic similarity search"),
    "index:stats": (index.stats, "Get index statistics and HNSW config"),
    "index:update_hnsw": (index.update_hnsw, "Migrate HNSW index configuration"),
    "system:health": (system.health, "Basic health check"),
    "system:health_ready": (system.health_ready, "Readiness probe"),
    "system:health_live": (system.health_live, "Liveness probe"),
    "system:metrics": (system.metrics, "Comprehensive collection metrics"),
}
