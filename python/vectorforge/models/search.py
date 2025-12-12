from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    filters: dict | None = Field(default=None, description="Optional metadata filters")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "top_k": 5,
                "filters": {"source_file": "textbook.pdf"}
            }
        }

class SearchResult(BaseModel):
    id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document text content")
    metadata: dict | None = Field(default=None, description="Document metadata")
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
