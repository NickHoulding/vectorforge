from typing import Any

from pydantic import BaseModel, Field

from vectorforge.config import VFGConfig


class SearchQuery(BaseModel):
    """Input model for semantic search queries.
    
    Defines the parameters for performing a semantic similarity search across
    the vector database. Supports filtering by metadata and controlling the
    number of results returned.
    
    Attributes:
        query: The search text to find semantically similar documents.
        top_k: Maximum number of results to return (1-100, default: 10).
        filters: Optional metadata filters as key-value pairs for narrowing results.
    """
    query: str = Field(
        ..., 
        min_length=VFGConfig.MIN_QUERY_LENGTH, 
        max_length=VFGConfig.MAX_QUERY_LENGTH, 
        description="Search query text"
    )
    top_k: int = Field(
        VFGConfig.DEFAULT_TOP_K, 
        ge=VFGConfig.MIN_TOP_K, 
        le=VFGConfig.MAX_TOP_K, 
        description="Number of results to return"
    )
    filters: dict[str, Any] | None = Field(default=None, description="Optional metadata filters")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "top_k": 5,
                "filters": {"source_file": "textbook.pdf"}
            }
        }

class SearchResult(BaseModel):
    """Individual search result with similarity scoring.
    
    Represents a single document match from a semantic search query, including
    the document's content, metadata, and cosine similarity score indicating
    how closely it matches the search query.
    
    Attributes:
        id: Unique identifier of the matched document.
        content: Full text content of the document chunk.
        metadata: Document metadata including source file and chunk information.
        score: Cosine similarity score (0-1), where 1 is perfect match.
    """
    id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document text content")
    metadata: dict[str, Any] | None = Field(default=None, description="Document metadata")
    score: float = Field(..., description="Similarity score")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "id": "abc-123-def",
                "content": "Machine learning is a subset of AI that focuses on...",
                "metadata": {"source_file": "textbook.pdf", "chunk_index": 0},
                "score": 0.89
            }
        }

class SearchResponse(BaseModel):
    """Complete search results response.
    
    Contains all matching documents from a semantic search query, ranked by
    similarity score in descending order. Includes the original query for
    reference and the total count of results returned.
    
    Attributes:
        query: The original search query text that was submitted.
        results: List of matching documents ranked by similarity score.
        count: Total number of results returned (may be less than top_k if fewer matches exist).
    """
    query: str = Field(..., description="Original search query")
    results: list[SearchResult] = Field(..., description="List of search results")
    count: int = Field(..., description="Number of results returned")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "results": [
                    {
                        "id": "abc-123",
                        "content": "Machine learning is a subset of AI...",
                        "metadata": {"source_file": "textbook.pdf", "chunk_index": 0},
                        "score": 0.89
                    },
                    {
                        "id": "def-456",
                        "content": "ML algorithms learn from data patterns...",
                        "metadata": {"source_file": "textbook.pdf", "chunk_index": 1},
                        "score": 0.82
                    }
                ],
                "count": 2
            }
        }
