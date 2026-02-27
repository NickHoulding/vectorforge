"""VectorForge API Initialization"""

from fastapi import FastAPI

from vectorforge import __version__
from vectorforge.collection_manager import CollectionManager
from vectorforge.logging_config import configure_logging

configure_logging()

app: FastAPI = FastAPI(
    title="VectorForge API",
    version=__version__,
    description="High-performance vector database with semantic search and multi-collection support",
)

manager: CollectionManager = CollectionManager()

from vectorforge.api import collections, documents, files, index, search, system

app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(files.router)
app.include_router(index.router)
app.include_router(search.router)
app.include_router(system.router)

__all__: list[str] = ["app", "manager"]
