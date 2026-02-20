from pydantic import BaseModel, Field


class IndexStatsResponse(BaseModel):
    """Lightweight index health and statistics.

    Provides essential metrics about the current state of the vector index.

    Attributes:
        total_documents: Number of active documents in the index.
        total_embeddings: Total number of embedding vectors in the index.
        embedding_dimension: Dimensionality of the embedding vectors.
    """

    status: str = Field(..., description="Operation status")
    total_documents: int = Field(..., description="Active documents")
    total_embeddings: int = Field(..., description="Total embeddings")
    embedding_dimension: int = Field(..., description="Embedding size")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "success",
                "total_documents": 1250,
                "total_embeddings": 1250,
                "embedding_dimension": 384,
            }
        }


class IndexSaveResponse(BaseModel):
    """Response from persisting the index to disk.

    Provides confirmation and statistics after saving the vector database
    to persistent storage.

    Attributes:
        status: Operation result (typically 'saved').
        directory: File system path where data was written.
        total_size_mb: Combined size of all saved files in megabytes.
        documents_saved: Number of document records persisted.
        embeddings_saved: Number of embedding vectors persisted.
        version: VectorForge version used to save the index.
    """

    status: str = Field(..., description="Operation status")
    directory: str = Field(..., description="Save directory path")
    total_size_mb: float = Field(..., ge=0, description="Total disk space used")
    documents_saved: int = Field(..., ge=0, description="Number of documents saved")
    embeddings_saved: int = Field(..., ge=0, description="Number of embeddings saved")
    version: str = Field(..., description="Index format version")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "saved",
                "directory": "./data",
                "total_size_mb": 2.816,
                "documents_saved": 1250,
                "embeddings_saved": 1250,
            }
        }


class IndexLoadResponse(BaseModel):
    """Response from loading a persisted index from disk.

    Provides confirmation and statistics after restoring the vector database
    from a backup. Includes counts of loaded documents and version information
    for compatibility checking.

    Attributes:
        status: Operation result (typically 'loaded').
        directory: File system path where backup was read from.
        documents_loaded: Number of document records restored.
        embeddings_loaded: Number of embedding vectors restored.
        version: VectorForge version that created the backup.
    """

    status: str = Field(..., description="Operation status")
    directory: str = Field(..., description="Load directory path")
    documents_loaded: int = Field(..., ge=0, description="Number of documents loaded")
    embeddings_loaded: int = Field(..., ge=0, description="Number of embeddings loaded")
    version: str = Field(..., description="Index format version")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "loaded",
                "directory": "./data/chroma_data",
                "documents_loaded": 1250,
                "embeddings_loaded": 1250,
                "version": "1.0",
            }
        }
