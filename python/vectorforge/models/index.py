from pydantic import BaseModel, Field


class IndexStatsResponse(BaseModel):
    """Lightweight index statistics"""
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
    """Response from saving index to disk"""
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
    """Response from loading index from disk"""
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
