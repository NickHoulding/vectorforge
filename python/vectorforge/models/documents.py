from typing import Any

from pydantic import BaseModel, Field

from vectorforge.config import VFConfig


class DocumentInput(BaseModel):
    """Input model for adding a new document to the vector index.
    
    Represents a document to be indexed with its text content and optional
    metadata. This is used when adding individual documents directly via the
    API rather than uploading files.
    
    Attributes:
        content: The text content to be indexed and searched.
        metadata: Optional key-value pairs for filtering and identification.
    """
    content: str = Field(
        ..., 
        min_length=VFConfig.MIN_CONTENT_LENGTH, 
        max_length=VFConfig.MAX_CONTENT_LENGTH, 
        description="Document text content"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "content": "Machine learning is a subset of AI...",
                "metadata": {"title": "ML Intro", "author": "Jane"}
            }
        }

class DocumentResponse(BaseModel):
    """Response model for document operations.
    
    Returned after successfully adding or deleting a document, providing
    confirmation with the document ID and operation status.
    
    Attributes:
        id: Unique identifier (UUID) of the document.
        status: Operation result (e.g., 'indexed', 'deleted').
    """
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
    """Detailed document information including content and metadata.
    
    Complete representation of a document in the vector database, including
    its unique identifier, full text content, and associated metadata.
    Returned when retrieving a specific document by ID.
    
    Attributes:
        id: Unique identifier (UUID) of the document.
        content: The full text content of the document.
        metadata: Associated metadata including source file and chunk information.
    """
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document text content")
    metadata: dict[str, Any] | None = Field(default=None, description="Document metadata")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "id": "abc-123-def",
                "content": "Machine learning is a subset of artificial intelligence...",
                "metadata": {"source_file": "textbook.pdf", "chunk_index": 0}
            }
        }
