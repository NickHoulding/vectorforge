from pydantic import BaseModel, Field


class IndexStatsResponse(BaseModel):
    """Lightweight index health and statistics.

    Provides essential metrics about the current state of the vector index.

    Attributes:
        total_documents: Number of active documents in the index.
        embedding_dimension: Dimensionality of the embedding vectors.
    """

    status: str = Field(..., description="Operation status")
    total_documents: int = Field(..., description="Active documents")
    embedding_dimension: int = Field(..., description="Embedding size")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "success",
                "total_documents": 1250,
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
        documents_saved: Number of document records persisted.
        version: VectorForge version used to save the index.
    """

    status: str = Field(..., description="Operation status")
    directory: str = Field(..., description="Save directory path")
    documents_saved: int = Field(..., ge=0, description="Number of documents saved")
    version: str = Field(..., description="Index format version")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "saved",
                "directory": "./data",
                "documents_saved": 1250,
                "version": "0.9.0",
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
        version: VectorForge version that created the backup.
    """

    status: str = Field(..., description="Operation status")
    directory: str = Field(..., description="Load directory path")
    documents_loaded: int = Field(..., ge=0, description="Number of documents loaded")
    version: str = Field(..., description="Index format version")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "loaded",
                "directory": "./data/chroma_data",
                "documents_loaded": 1250,
                "version": "1.0",
            }
        }
