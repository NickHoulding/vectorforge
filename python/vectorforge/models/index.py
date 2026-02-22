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
