"""VectorForge - High-performance in-memory vector database."""

try:
    from importlib.metadata import version

    __version__: str = version("vectorforge")
except Exception:
    __version__ = "1.0.0"

__all__: list[str] = ["__version__", "VectorEngine"]

from vectorforge.vector_engine import VectorEngine
