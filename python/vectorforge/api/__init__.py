"""VectorForge API Initialization"""

from fastapi import FastAPI

from vectorforge import __version__
from vectorforge.logging_config import configure_logging
from vectorforge.vector_engine import VectorEngine

configure_logging()

app: FastAPI = FastAPI(
    title="VectorForge API",
    version=__version__,
    description="High-performance in-memory vector database with semantic search",
)
engine: VectorEngine = VectorEngine()

from vectorforge.api import documents, files, index, search, system

app.include_router(documents.router)
app.include_router(files.router)
app.include_router(index.router)
app.include_router(search.router)
app.include_router(system.router)

__all__: list[str] = ["app", "engine"]
