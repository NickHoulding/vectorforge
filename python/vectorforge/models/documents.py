from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    content: str = Field(..., min_length=1, description="Document text content")
    metadata: dict | None = Field(default=None, description="Optional metadata")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "content": "Machine learning is a subset of AI...",
                "metadata": {"title": "ML Intro", "author": "Jane"}
            }
        }

class DocumentResponse(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    status: str = Field(..., description="Operation status")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "id": "abc-123-def",
                "status": "indexed"
            }
        }

class DocumentDetail(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document text content")
    metadata: dict | None = Field(default=None, description="Document metadata")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "id": "abc-123-def",
                "content": "Machine learning is a subset of artificial intelligence...",
                "metadata": {"source_file": "textbook.pdf", "chunk_index": 0}
            }
        }
