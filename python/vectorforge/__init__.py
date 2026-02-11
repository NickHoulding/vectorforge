"""VectorForge - High-performance in-memory vector database."""

try:
    from importlib.metadata import version

    __version__: str = version("vectorforge")
except Exception:
    __version__ = "0.9.0"

__all__: list[str] = ["__version__", "VectorEngine"]

from vectorforge.vector_engine import VectorEngine

try:
    from vectorforge import vectorforge_cpp

    __all__.append("vectorforge_cpp")
except ImportError:
    pass
