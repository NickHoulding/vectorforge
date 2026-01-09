"""VectorForge API + Engine configuration class"""


class Config:
    """Central configuration for VectorForge engine and API.
    
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
    # API Configuration
    # =============================================================================
    
    API_PORT: int = 3001
    """Default port for the FastAPI server."""
    
    API_HOST: str = "0.0.0.0"
    """Default host binding for the FastAPI server."""
    
    # =============================================================================
    # Supported File Types
    # =============================================================================
    
    SUPPORTED_FILE_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")
    """Tuple of supported file extensions for upload."""
