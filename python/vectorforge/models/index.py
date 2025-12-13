from pydantic import BaseModel, Field


class IndexStatsResponse(BaseModel):
    """Lightweight index health and statistics.
    
    Provides essential metrics about the current state of the vector index,
    including document counts, deletion ratios, and compaction status. This is
    a lightweight endpoint compared to the full metrics endpoint.
    
    Attributes:
        total_documents: Number of documents in storage (active + deleted).
        total_embeddings: Total number of embedding vectors in the index.
        deleted_documents: Count of documents marked for deletion.
        deleted_ratio: Ratio of deleted to total embeddings (0-1).
        needs_compaction: Whether index cleanup is recommended.
        embedding_dimension: Dimensionality of the embedding vectors.
    """
    total_documents: int = Field(..., description="Active documents")
    total_embeddings: int = Field(..., description="Total embeddings")
    deleted_documents: int = Field(..., description="Deleted count")
    deleted_ratio: float = Field(..., ge=0, le=1, description="Deletion ratio")
    needs_compaction: bool = Field(..., description="Compaction needed")
    embedding_dimension: int = Field(..., description="Embedding size")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "total_documents": 1250,
                "total_embeddings": 1500,
                "deleted_documents": 250,
                "deleted_ratio": 0.167,
                "needs_compaction": False,
                "embedding_dimension": 384
            }
        }

class IndexSaveResponse(BaseModel):
    """Response from persisting the index to disk.
    
    Provides confirmation and detailed statistics after saving the vector
    database to persistent storage. Includes file sizes and document counts
    for monitoring storage usage.
    
    Attributes:
        status: Operation result (typically 'saved').
        directory: File system path where data was written.
        metadata_size_mb: Size of the metadata JSON file in megabytes.
        embeddings_size_mb: Size of the embeddings NPZ file in megabytes.
        total_size_mb: Combined size of all saved files in megabytes.
        documents_saved: Number of document records persisted.
        embeddings_saved: Number of embedding vectors persisted.
        version: VectorForge version used to save the index.
    """
    status: str = Field(..., description="Operation status")
    directory: str = Field(..., description="Save directory path")
    metadata_size_mb: float = Field(..., ge=0, description="Metadata file size")
    embeddings_size_mb: float = Field(..., ge=0, description="Embeddings file size")
    total_size_mb: float = Field(..., ge=0, description="Total disk space used")
    documents_saved: int = Field(..., ge=0, description="Number of documents saved")
    embeddings_saved: int = Field(..., ge=0, description="Number of embeddings saved")
    version: str = Field(..., description="Index format version")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "saved",
                "directory": "./data",
                "metadata_size_mb": 0.512,
                "embeddings_size_mb": 2.304,
                "total_size_mb": 2.816,
                "documents_saved": 1250,
                "embeddings_saved": 1250
            }
        }

class IndexLoadResponse(BaseModel):
    """Response from loading a persisted index from disk.
    
    Provides confirmation and statistics after restoring the vector database
    from persistent storage. Includes counts of loaded documents and information
    about the index format version for compatibility checking.
    
    Attributes:
        status: Operation result (typically 'loaded').
        directory: File system path where data was read from.
        documents_loaded: Number of document records restored.
        embeddings_loaded: Number of embedding vectors restored.
        deleted_docs: Count of documents marked as deleted in the loaded index.
        version: VectorForge version that created the saved index.
    """
    status: str = Field(..., description="Operation status")
    directory: str = Field(..., description="Load directory path")
    documents_loaded: int = Field(..., ge=0, description="Number of documents loaded")
    embeddings_loaded: int = Field(..., ge=0, description="Number of embeddings loaded")
    deleted_docs: int = Field(..., ge=0, description="Number of deleted documents")
    version: str = Field(..., description="Index format version")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "loaded",
                "directory": "./data",
                "documents_loaded": 1250,
                "embeddings_loaded": 1500,
                "deleted_docs": 250,
                "version": "1.0"
            }
        }
