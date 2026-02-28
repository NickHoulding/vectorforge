import os


class VFGConfig:
    """Central configuration for VectorForge engine.

    Contains all configurable constants for the vector database including
    model settings, validation limits, storage paths, and performance tuning.
    """

    # =============================================================================
    # Model Configuration
    # =============================================================================

    MODEL_NAME: str = "all-MiniLM-L6-v2"
    """Sentence transformer model for generating embeddings."""

    EMBEDDING_DIMENSION: int = 384
    """Dimension of the embedding vectors (specific to all-MiniLM-L6-v2)."""

    # =============================================================================
    # Storage Configuration
    # =============================================================================

    MAX_PATH_LEN: int = 4096
    """Maximum valid path length for save/load functionality"""

    MAX_FILENAME_LENGTH: int = 255
    """Maximum filename length (including extension)."""

    # =============================================================================
    # Content Validation
    # =============================================================================

    MIN_CONTENT_LENGTH: int = 1
    """Minimum character length for document content."""

    MAX_CONTENT_LENGTH: int = 10_000
    """Maximum character length for document content."""

    MIN_QUERY_LENGTH: int = 1
    """Minimum character length for search queries."""

    MAX_QUERY_LENGTH: int = 2_000
    """Maximum character length for search queries."""

    # =============================================================================
    # Document Processing
    # =============================================================================

    DEFAULT_CHUNK_SIZE: int = 500
    """Default number of characters per chunk when splitting documents."""

    DEFAULT_CHUNK_OVERLAP: int = 50
    """Default number of overlapping characters between consecutive chunks."""

    # =============================================================================
    # Search Configuration
    # =============================================================================

    DEFAULT_TOP_K: int = 10
    """Default number of search results to return."""

    MIN_TOP_K: int = 1
    """Minimum value for top_k parameter."""

    MAX_TOP_K: int = 100
    """Maximum value for top_k parameter."""

    # =============================================================================
    # Metrics Configuration
    # =============================================================================

    MAX_QUERY_HISTORY: int = 1_000
    """Maximum number of query times to retain for percentile calculations."""

    # =============================================================================
    # Supported File Types
    # =============================================================================

    SUPPORTED_FILE_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")
    """Tuple of supported file extensions for upload."""

    # =============================================================================
    # ChromaDB Configuration
    # =============================================================================

    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_DATA_DIR",
        "/data/chroma" if os.path.exists("/data") else "./data/chroma",
    )
    """Directory for ChromaDB persistent storage. Configurable via CHROMA_DATA_DIR env var."""

    MODEL_CACHE_DIR: str = os.getenv(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")
    )
    """Directory for HuggingFace model cache. Configurable via HF_HOME env var."""

    MIGRATION_BATCH_SIZE: int = 1000
    """Batch size for HNSW configuration migration (collection-level recreation)."""

    # =============================================================================
    # Collection Management Configuration
    # =============================================================================

    MAX_COLLECTIONS: int = int(os.getenv("MAX_COLLECTIONS", "100"))
    """Maximum number of collections allowed per instance."""

    DEFAULT_COLLECTION_NAME: str = "vectorforge"
    """Default collection when none specified (backward compatibility)."""

    COLLECTION_CACHE_SIZE: int = int(os.getenv("COLLECTION_CACHE_SIZE", "50"))
    """FIFO cache size for collection engines."""

    MAX_COLLECTION_NAME_LENGTH: int = 64
    """Maximum length for collection names."""

    MIN_COLLECTION_NAME_LENGTH: int = 3
    """Minimum length for collection names."""

    MAX_DESCRIPTION_LENGTH: int = 500
    """Maximum character length of a collection's description."""

    COLLECTION_NAME_PATTERN: str = r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$"
    """Regex pattern for valid collection names."""

    MAX_METADATA_PAIRS: int = 20
    """Maximum number of key-value pairs for collection metadata."""

    # =============================================================================
    # Configuration Class Validator
    # =============================================================================

    @classmethod
    def validate(cls) -> None:
        """Validate configuration values."""
        if not isinstance(cls.MODEL_NAME, str):
            raise ValueError("MODEL_NAME must be a string")
        if len(cls.MODEL_NAME) == 0:
            raise ValueError("MODEL_NAME cannot be empty")

        if not isinstance(cls.EMBEDDING_DIMENSION, int):
            raise ValueError("EMBEDDING_DIMENSION must be an int")
        if cls.EMBEDDING_DIMENSION <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be > 0")

        if not isinstance(cls.MAX_PATH_LEN, int):
            raise ValueError("MAX_PATH_LEN must be an int")
        if cls.MAX_PATH_LEN <= 0:
            raise ValueError("MAX_PATH_LEN must be > 0")

        if not isinstance(cls.MAX_FILENAME_LENGTH, int):
            raise ValueError("MAX_FILENAME_LENGTH must be an int")
        if cls.MAX_FILENAME_LENGTH <= 0:
            raise ValueError("MAX_FILENAME_LENGTH must be > 0")
        if cls.MAX_FILENAME_LENGTH > cls.MAX_PATH_LEN:
            raise ValueError("MAX_FILENAME_LENGTH must be <= MAX_PATH_LEN")

        if not isinstance(cls.MIN_CONTENT_LENGTH, int):
            raise ValueError("MIN_CONTENT_LENGTH must be an int")
        if cls.MIN_CONTENT_LENGTH <= 0:
            raise ValueError("MIN_CONTENT_LENGTH must be > 0")

        if not isinstance(cls.MAX_CONTENT_LENGTH, int):
            raise ValueError("MAX_CONTENT_LENGTH must be an int")
        if cls.MAX_CONTENT_LENGTH <= 0:
            raise ValueError("MAX_CONTENT_LENGTH must be > 0")
        if cls.MAX_CONTENT_LENGTH < cls.MIN_CONTENT_LENGTH:
            raise ValueError("MAX_CONTENT_LENGTH must be >= MIN_CONTENT_LENGTH")

        if not isinstance(cls.MIN_QUERY_LENGTH, int):
            raise ValueError("MIN_QUERY_LENGTH must be an int")
        if cls.MIN_QUERY_LENGTH <= 0:
            raise ValueError("MIN_QUERY_LENGTH must be > 0")

        if not isinstance(cls.MAX_QUERY_LENGTH, int):
            raise ValueError("MAX_QUERY_LENGTH must be an int")
        if cls.MAX_QUERY_LENGTH <= 0:
            raise ValueError("MAX_QUERY_LENGTH must be > 0")
        if cls.MAX_QUERY_LENGTH < cls.MIN_QUERY_LENGTH:
            raise ValueError("MAX_QUERY_LENGTH must be >= MIN_QUERY_LENGTH")

        if not isinstance(cls.DEFAULT_CHUNK_SIZE, int):
            raise ValueError("DEFAULT_CHUNK_SIZE must be an int")
        if cls.DEFAULT_CHUNK_SIZE <= 0:
            raise ValueError("DEFAULT_CHUNK_SIZE must be > 0")

        if not isinstance(cls.DEFAULT_CHUNK_OVERLAP, int):
            raise ValueError("DEFAULT_CHUNK_OVERLAP must be an int")
        if cls.DEFAULT_CHUNK_OVERLAP < 0:
            raise ValueError("DEFAULT_CHUNK_OVERLAP must be >= 0")
        if cls.DEFAULT_CHUNK_OVERLAP > cls.DEFAULT_CHUNK_SIZE:
            raise ValueError("DEFAULT_CHUNK_OVERLAP must be <= DEFAULT_CHUNK_SIZE")

        if not isinstance(cls.DEFAULT_TOP_K, int):
            raise ValueError("DEFAULT_TOP_K must be an int")
        if cls.DEFAULT_TOP_K <= 0:
            raise ValueError("DEFAULT_TOP_K must be > 0")
        if not (cls.MIN_TOP_K <= cls.DEFAULT_TOP_K <= cls.MAX_TOP_K):
            raise ValueError("DEFAULT_TOP_K must be between MIN_TOP_K and MAX_TOP_K")

        if not isinstance(cls.MIN_TOP_K, int):
            raise ValueError("MIN_TOP_K must be an int")
        if cls.MIN_TOP_K <= 0:
            raise ValueError("MIN_TOP_K must be > 0")

        if not isinstance(cls.MAX_TOP_K, int):
            raise ValueError("MAX_TOP_K must be an int")
        if cls.MAX_TOP_K <= 0:
            raise ValueError("MAX_TOP_K must be > 0")
        if cls.MAX_TOP_K < cls.MIN_TOP_K:
            raise ValueError("MAX_TOP_K must be >= MIN_TOP_K")

        if not isinstance(cls.MAX_QUERY_HISTORY, int):
            raise ValueError("MAX_QUERY_HISTORY must be an int")
        if cls.MAX_QUERY_HISTORY < 0:
            raise ValueError("MAX_QUERY_HISTORY must be >= 0")

        if not isinstance(cls.SUPPORTED_FILE_EXTENSIONS, tuple):
            raise ValueError("SUPPORTED_FILE_EXTENSIONS must be a tuple")
        if len(cls.SUPPORTED_FILE_EXTENSIONS) == 0:
            raise ValueError("SUPPORTED_FILE_EXTENSIONS cannot be empty")
        if not all(
            isinstance(ext, str) and ext.startswith(".") and len(ext) > 1
            for ext in cls.SUPPORTED_FILE_EXTENSIONS
        ):
            raise ValueError(
                "Each entry in SUPPORTED_FILE_EXTENSIONS must be a string starting with '.'"
            )
