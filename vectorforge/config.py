"""Central configuration for VectorForge, loaded from environment variables."""

import os


class VFGConfig:
    """Central configuration for VectorForge engine.

    Contains all configurable constants for the vector database including
    model settings, validation limits, storage paths, and performance tuning.
    """

    # =============================================================================
    # Models Configuration
    # =============================================================================

    EMBEDDING_MODEL_NAME: str = os.getenv("VF_MODEL_NAME", "all-MiniLM-L6-v2")
    """Sentence transformer model for generating embeddings."""

    EMBEDDING_DIMENSION: int = int(os.getenv("VF_EMBEDDING_DIMENSION", "384"))
    """Dimension of the embedding vectors (specific to all-MiniLM-L6-v2)."""

    RERANKING_MODEL_NAME: str = os.getenv(
        "VF_RERANKING_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    """Cross-encoder model used to re-score and reorder search results."""

    SHOULD_RERANK: bool = bool(os.getenv("VF_SHOULD_RERANK", "True"))
    """Whether to apply cross-encoder reranking to search results by default."""

    # =============================================================================
    # Storage Configuration
    # =============================================================================

    MAX_PATH_LEN: int = int(os.getenv("VF_MAX_PATH_LEN", "4096"))
    """Maximum valid path length for save/load functionality."""

    MAX_FILENAME_LENGTH: int = int(os.getenv("VF_MAX_FILENAME_LENGTH", "255"))
    """Maximum filename length (including extension)."""

    # =============================================================================
    # Content Validation
    # =============================================================================

    MIN_CONTENT_LENGTH: int = int(os.getenv("VF_MIN_CONTENT_LENGTH", "1"))
    """Minimum character length for document content."""

    MAX_CONTENT_LENGTH: int = int(os.getenv("VF_MAX_CONTENT_LENGTH", "10000"))
    """Maximum character length for document content."""

    MAX_BATCH_SIZE: int = int(os.getenv("VF_MAX_BATCH_SIZE", "100"))
    """Maximum number of documents per batch add or batch delete request."""

    MIN_QUERY_LENGTH: int = int(os.getenv("VF_MIN_QUERY_LENGTH", "1"))
    """Minimum character length for search queries."""

    MAX_QUERY_LENGTH: int = int(os.getenv("VF_MAX_QUERY_LENGTH", "2000"))
    """Maximum character length for search queries."""

    # =============================================================================
    # Document Processing
    # =============================================================================

    DEFAULT_CHUNK_SIZE: int = int(os.getenv("VF_DEFAULT_CHUNK_SIZE", "500"))
    """Default number of characters per chunk when splitting documents."""

    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("VF_DEFAULT_CHUNK_OVERLAP", "50"))
    """Default number of overlapping characters between consecutive chunks."""

    # =============================================================================
    # Search Configuration
    # =============================================================================

    DEFAULT_TOP_K: int = int(os.getenv("VF_DEFAULT_TOP_K", "10"))
    """Default number of search results to return."""

    MIN_TOP_K: int = int(os.getenv("VF_MIN_TOP_K", "1"))
    """Minimum value for top_k parameter."""

    MAX_TOP_K: int = int(os.getenv("VF_MAX_TOP_K", "100"))
    """Maximum value for top_k parameter."""

    DEFAULT_TOP_N: int = int(os.getenv("VF_DEFAULT_TOP_N", "5"))
    """Default number of reranked results to return after cross-encoder scoring."""

    MIN_TOP_N: int = int(os.getenv("VF_MIN_TOP_N", "1"))
    """Minimum value for the top_n reranking parameter."""

    MAX_TOP_N: int = int(os.getenv("VF_MAX_TOP_N", "100"))
    """Maximum value for the top_n reranking parameter."""

    VALID_FILTER_OPERATORS: frozenset[str] = frozenset(["$gte", "$lte", "$in", "$ne"])
    """Permitted operators for metadata search filters."""

    VALID_DOCUMENT_FILTER_OPERATORS: frozenset[str] = frozenset(
        ["$contains", "$not_contains"]
    )
    """Permitted operators for document-text search filters (maps to ChromaDB where_document)."""

    # =============================================================================
    # Metrics Configuration
    # =============================================================================

    MAX_QUERY_HISTORY: int = int(os.getenv("VF_MAX_QUERY_HISTORY", "1000"))
    """Maximum number of query times to retain for percentile calculations."""

    DISK_SIZE_TTL_MINS: int = int(os.getenv("VF_DISK_SIZE_TTL_MINS", "5"))
    """TTL in minutes for the cached ChromaDB disk size calculation."""

    # =============================================================================
    # Supported File Types
    # =============================================================================

    SUPPORTED_FILE_EXTENSIONS: tuple[str, ...] = tuple(
        os.getenv("VF_SUPPORTED_FILE_EXTENSIONS", ".pdf,.txt").split(",")
    )
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

    MIGRATION_BATCH_SIZE: int = int(os.getenv("VF_MIGRATION_BATCH_SIZE", "1000"))
    """Batch size for HNSW configuration migration (collection-level recreation)."""

    # =============================================================================
    # Collection Management Configuration
    # =============================================================================

    MAX_COLLECTIONS: int = int(os.getenv("VF_MAX_COLLECTIONS", "100"))
    """Maximum number of collections allowed per instance."""

    DEFAULT_COLLECTION_NAME: str = os.getenv(
        "VF_DEFAULT_COLLECTION_NAME", "vectorforge"
    )
    """Default collection when none specified."""

    COLLECTION_CACHE_SIZE: int = int(os.getenv("VF_COLLECTION_CACHE_SIZE", "50"))
    """FIFO cache size for collection engines."""

    MAX_COLLECTION_NAME_LENGTH: int = int(
        os.getenv("VF_MAX_COLLECTION_NAME_LENGTH", "64")
    )
    """Maximum length for collection names."""

    MIN_COLLECTION_NAME_LENGTH: int = int(
        os.getenv("VF_MIN_COLLECTION_NAME_LENGTH", "3")
    )
    """Minimum length for collection names."""

    MAX_DESCRIPTION_LENGTH: int = int(os.getenv("VF_MAX_DESCRIPTION_LENGTH", "500"))
    """Maximum character length of a collection's description."""

    COLLECTION_NAME_PATTERN: str = os.getenv(
        "VF_COLLECTION_NAME_PATTERN", r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$"
    )
    """Regex pattern for valid collection names."""

    MAX_METADATA_PAIRS: int = int(os.getenv("VF_MAX_METADATA_PAIRS", "20"))
    """Maximum number of key-value pairs for collection metadata."""

    VALID_SCALAR_TYPES: frozenset[type] = frozenset([str, int, float, bool])
    """Permitted Python types for document metadata and filter values.

    ChromaDB rejects any metadata value that is not one of these four scalar
    types. ``None`` and nested structures (lists, dicts) are not allowed.
    """

    # =============================================================================
    # Logging Configuration
    # =============================================================================

    LOG_LEVEL: int = getattr(
        __import__("logging"), os.getenv("VF_LOG_LEVEL", "INFO").upper(), 20
    )
    """Logging level (DEBUG=10, INFO=20, WARNING=30, ERROR=40). Configurable via VF_LOG_LEVEL."""

    LOG_FILE: str = os.getenv("VF_LOG_FILE", ".logs/vectorforge.log")
    """Path to rotating log file. Configurable via VF_LOG_FILE."""

    LOG_JSON_CONSOLE: bool = os.getenv("VF_LOG_JSON_CONSOLE", "false").lower() == "true"
    """Output JSON format to console instead of human-readable. Configurable via VF_LOG_JSON_CONSOLE."""

    LOG_MAX_TEXT_LEN: int = int(os.getenv("VF_LOG_MAX_TEXT_LEN", "100"))
    """Maximum characters for text values logged (truncation limit). Configurable via VF_LOG_MAX_TEXT_LEN."""

    LOG_MAX_BYTES: int = int(os.getenv("VF_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    """Maximum size per log file before rotation in bytes. Configurable via VF_LOG_MAX_BYTES."""

    LOG_BACKUP_COUNT: int = int(os.getenv("VF_LOG_BACKUP_COUNT", "5"))
    """Number of rotated log file backups to keep. Configurable via VF_LOG_BACKUP_COUNT."""

    # =============================================================================
    # Configuration Class Validator
    # =============================================================================

    @classmethod
    def validate(cls) -> None:
        """Raise ValueError if any configuration value is invalid or inconsistent.

        Raises:
            ValueError: If any field has the wrong type or an out-of-range value.
        """
        if not isinstance(cls.EMBEDDING_MODEL_NAME, str):
            raise ValueError("MODEL_NAME must be a string")
        if len(cls.EMBEDDING_MODEL_NAME) == 0:
            raise ValueError("MODEL_NAME cannot be empty")

        if not isinstance(cls.EMBEDDING_DIMENSION, int):
            raise ValueError("EMBEDDING_DIMENSION must be an int")
        if cls.EMBEDDING_DIMENSION <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be > 0")

        if not isinstance(cls.RERANKING_MODEL_NAME, str):
            raise ValueError("RERANKING_MODEL_NAME must be a string")
        if len(cls.RERANKING_MODEL_NAME) == 0:
            raise ValueError("RERANKING_MODEL_NAME cannot be empty")

        if not isinstance(cls.SHOULD_RERANK, bool):
            raise ValueError("SHOULD_RERANK must be a bool")

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

        if not isinstance(cls.MIN_TOP_N, int):
            raise ValueError("MIN_TOP_N must be an int")
        if cls.MIN_TOP_N <= 0:
            raise ValueError("MIN_TOP_N must be > 0")

        if not isinstance(cls.MAX_TOP_N, int):
            raise ValueError("MAX_TOP_N must be an int")
        if cls.MAX_TOP_N <= 0:
            raise ValueError("MAX_TOP_N must be > 0")
        if cls.MAX_TOP_N < cls.MIN_TOP_N:
            raise ValueError("MAX_TOP_N must be >= MIN_TOP_N")

        if not isinstance(cls.DEFAULT_TOP_N, int):
            raise ValueError("DEFAULT_TOP_N must be an int")
        if cls.DEFAULT_TOP_N <= 0:
            raise ValueError("DEFAULT_TOP_N must be > 0")
        if not (cls.MIN_TOP_N <= cls.DEFAULT_TOP_N <= cls.MAX_TOP_N):
            raise ValueError("DEFAULT_TOP_N must be between MIN_TOP_N and MAX_TOP_N")

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

        if not isinstance(cls.VALID_SCALAR_TYPES, frozenset):
            raise ValueError("VALID_SCALAR_TYPES must be a frozenset.")
        if len(cls.VALID_SCALAR_TYPES) == 0:
            raise ValueError("Config must allow at least one valid metadata type.")
        if not all(isinstance(typ, type) for typ in cls.VALID_SCALAR_TYPES):
            raise ValueError(
                "Each entry in VALID_METADATA_TYPES must be a valid type (str, int, float, bool)."
            )

        if not isinstance(cls.VALID_FILTER_OPERATORS, frozenset):
            raise ValueError("VALID_FILTER_OPERATORS must be a frozenset.")
        if len(cls.VALID_FILTER_OPERATORS) == 0:
            raise ValueError(
                "Config must allow at least one valid search filter operator."
            )
        if not all(
            isinstance(op, str) and op.startswith("$")
            for op in cls.VALID_FILTER_OPERATORS
        ):
            raise ValueError(
                "Each entry in VALID_FILTER_OPERATORS must be a string starting with '$'."
            )

        if not isinstance(cls.VALID_DOCUMENT_FILTER_OPERATORS, frozenset):
            raise ValueError("VALID_DOCUMENT_FILTER_OPERATORS must be a frozenset.")
        if len(cls.VALID_DOCUMENT_FILTER_OPERATORS) == 0:
            raise ValueError(
                "Config must allow at least one valid document filter operator."
            )
        if not all(
            isinstance(op, str) and op.startswith("$")
            for op in cls.VALID_DOCUMENT_FILTER_OPERATORS
        ):
            raise ValueError(
                "Each entry in VALID_DOCUMENT_FILTER_OPERATORS must be a string starting with '$'."
            )
