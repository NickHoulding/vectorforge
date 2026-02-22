from pydantic import BaseModel, Field


class HNSWConfig(BaseModel):
    """HNSW index configuration parameters.

    Hierarchical Navigable Small World (HNSW) is the indexing algorithm used
    by ChromaDB for efficient approximate nearest neighbor search.

    Attributes:
        space: Distance metric used for similarity (e.g., 'cosine', 'l2', 'ip').
        ef_construction: Size of the dynamic candidate list for index construction.
                        Higher values improve index quality but slow construction.
        ef_search: Size of the dynamic candidate list for search operations.
                  Higher values improve search accuracy but slow queries.
        max_neighbors: Maximum number of connections per node in the graph.
                      Higher values improve recall but increase memory usage.
        resize_factor: Factor by which to grow the index when capacity is reached.
        sync_threshold: Number of operations before forcing index persistence.
    """

    space: str = Field(..., description="Distance metric (cosine, l2, ip)")
    ef_construction: int = Field(..., description="Index construction parameter")
    ef_search: int = Field(..., description="Search quality parameter")
    max_neighbors: int = Field(..., description="Max connections per node")
    resize_factor: float = Field(..., description="Index growth factor")
    sync_threshold: int = Field(..., description="Persistence threshold")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "space": "cosine",
                "ef_construction": 100,
                "ef_search": 100,
                "max_neighbors": 16,
                "resize_factor": 1.2,
                "sync_threshold": 1000,
            }
        }


class IndexStatsResponse(BaseModel):
    """Lightweight index health and statistics.

    Provides essential metrics about the current state of the vector index.

    Attributes:
        total_documents: Number of active documents in the index.
        embedding_dimension: Dimensionality of the embedding vectors.
        hnsw_config: HNSW index configuration parameters.
    """

    status: str = Field(..., description="Operation status")
    total_documents: int = Field(..., description="Active documents")
    embedding_dimension: int = Field(..., description="Embedding size")
    hnsw_config: HNSWConfig = Field(..., description="HNSW index configuration")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "success",
                "total_documents": 1250,
                "embedding_dimension": 384,
                "hnsw_config": {
                    "space": "cosine",
                    "ef_construction": 100,
                    "ef_search": 100,
                    "max_neighbors": 16,
                    "resize_factor": 1.2,
                    "sync_threshold": 1000,
                },
            }
        }
