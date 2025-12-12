"""VectorForge - High-performance in-memory vector database."""

try:
    from importlib.metadata import version
    __version__ = version("vectorforge")
except Exception:
    __version__ = "0.9.0"

__all__ = ["__version__"]
