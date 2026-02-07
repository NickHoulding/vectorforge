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

    DEFAULT_DATA_DIR: str = "./data"
    """Default directory for saving/loading index state."""

    METADATA_FILENAME: str = "metadata.json"
    """Filename for metadata storage."""

    EMBEDDINGS_FILENAME: str = "embeddings.npz"
    """Filename for embeddings storage."""

    MAX_PATH_LEN: int = 4096
    """Maximum valid path length for save/load functionality"""

    MAX_FILENAME_LENGTH: int = 255
    """Maximum filename length (including extension)."""

    # =============================================================================
    # Index Management
    # =============================================================================

    COMPACTION_THRESHOLD: float = 0.25
    """Ratio of deleted docs to total embeddings that triggers auto-compaction."""

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
    # Configuration Class Validator
    # =============================================================================

    @classmethod
    def validate(cls) -> None:
        """Validate configuration values."""
        assert isinstance(cls.MODEL_NAME, str)
        assert len(cls.MODEL_NAME) > 0

        assert isinstance(cls.EMBEDDING_DIMENSION, int)
        assert cls.EMBEDDING_DIMENSION > 0

        assert isinstance(cls.DEFAULT_DATA_DIR, str)
        assert len(cls.DEFAULT_DATA_DIR) > 0
        assert len(cls.DEFAULT_DATA_DIR) < cls.MAX_PATH_LEN

        assert isinstance(cls.METADATA_FILENAME, str)
        assert len(cls.METADATA_FILENAME) > 0
        assert len(cls.METADATA_FILENAME) <= cls.MAX_FILENAME_LENGTH
        assert os.path.basename(cls.EMBEDDINGS_FILENAME) == cls.EMBEDDINGS_FILENAME

        assert isinstance(cls.EMBEDDINGS_FILENAME, str)
        assert len(cls.EMBEDDINGS_FILENAME) > 0
        assert len(cls.EMBEDDINGS_FILENAME) <= cls.MAX_FILENAME_LENGTH
        assert os.path.basename(cls.METADATA_FILENAME) == cls.METADATA_FILENAME

        assert isinstance(cls.MAX_PATH_LEN, int)
        assert cls.MAX_PATH_LEN > 0

        assert isinstance(cls.MAX_FILENAME_LENGTH, int)
        assert cls.MAX_FILENAME_LENGTH > 0
        assert cls.MAX_FILENAME_LENGTH <= cls.MAX_PATH_LEN

        assert isinstance(cls.COMPACTION_THRESHOLD, float)
        assert 0 < cls.COMPACTION_THRESHOLD < 1.0

        assert isinstance(cls.MIN_CONTENT_LENGTH, int)
        assert cls.MIN_CONTENT_LENGTH > 0

        assert isinstance(cls.MAX_CONTENT_LENGTH, int)
        assert cls.MAX_CONTENT_LENGTH > 0
        assert cls.MAX_CONTENT_LENGTH >= cls.MIN_CONTENT_LENGTH

        assert isinstance(cls.MIN_QUERY_LENGTH, int)
        assert cls.MIN_QUERY_LENGTH > 0

        assert isinstance(cls.MAX_QUERY_LENGTH, int)
        assert cls.MAX_QUERY_LENGTH > 0
        assert cls.MAX_QUERY_LENGTH >= cls.MIN_QUERY_LENGTH

        assert isinstance(cls.DEFAULT_CHUNK_SIZE, int)
        assert cls.DEFAULT_CHUNK_SIZE > 0

        assert isinstance(cls.DEFAULT_CHUNK_OVERLAP, int)
        assert cls.DEFAULT_CHUNK_OVERLAP >= 0
        assert cls.DEFAULT_CHUNK_OVERLAP <= cls.DEFAULT_CHUNK_SIZE

        assert isinstance(cls.DEFAULT_TOP_K, int)
        assert cls.DEFAULT_TOP_K > 0
        assert cls.MIN_TOP_K <= cls.DEFAULT_TOP_K <= cls.MAX_TOP_K

        assert isinstance(cls.MIN_TOP_K, int)
        assert cls.MIN_TOP_K > 0

        assert isinstance(cls.MAX_TOP_K, int)
        assert cls.MAX_TOP_K > 0
        assert cls.MAX_TOP_K >= cls.MIN_TOP_K

        assert isinstance(cls.MAX_QUERY_HISTORY, int)
        assert cls.MAX_QUERY_HISTORY >= 0

        assert isinstance(cls.SUPPORTED_FILE_EXTENSIONS, tuple)
        assert len(cls.SUPPORTED_FILE_EXTENSIONS) > 0
        assert all(
            isinstance(ext, str) and ext.startswith(".") and len(ext) > 1
            for ext in cls.SUPPORTED_FILE_EXTENSIONS
        )
