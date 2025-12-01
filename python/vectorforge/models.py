from pydantic import BaseModel, Field


# =============================================================================
# API Request Models (Input)
# =============================================================================

class DocumentInput(BaseModel):
    content: str = Field(..., min_length=1, description="Document text content")
    metadata: dict | None = Field(default=None, description="Optional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Machine learning is a subset of AI...",
                "metadata": {"title": "ML Intro", "author": "Jane"}
            }
        }

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    filters: dict | None = None

# =============================================================================
# API Response Models (Output)
# =============================================================================
class DocumentResponse(BaseModel):
    id: str
    status: str

class DocumentDetail(BaseModel):
    id: str
    content: str
    metadata: dict | None = None

class FileUploadResponse(BaseModel):
    filename: str
    chunks_created: int
    doc_ids: list[str]
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "textbook.pdf",
                "chunks_created": 5,
                "document_ids": ["abc-123", "def-456", "ghi-789", ...],
                "status": "indexed"
            }
        }

class FileDeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    doc_ids: list[str]
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "textbook.pdf",
                "chunks_deleted": 5,
                "document_ids": ["abc-123", "def-456", "ghi-789", ...],
                "status": "deleted"
                ""
            }
        }

class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict | None = None
    score: float

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    count: int
