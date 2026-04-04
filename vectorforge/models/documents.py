"""Pydantic request and response models for document management."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vectorforge.config import VFGConfig


class DocumentInput(BaseModel):
    """Input model for adding a new document to the vector index.

    Represents a document to be indexed with its text content and optional
    metadata. This is used when adding individual documents directly via the
    API rather than uploading files.

    Attributes:
        content: The text content to be indexed and searched.
        metadata: Optional key-value pairs for filtering and identification.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Machine learning is a subset of AI...",
                "metadata": {"title": "ML Intro", "author": "Jane"},
            }
        }
    )

    content: str = Field(
        ...,
        min_length=VFGConfig.MIN_CONTENT_LENGTH,
        max_length=VFGConfig.MAX_CONTENT_LENGTH,
        description="Document text content",
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata"
    )


class DocumentResponse(BaseModel):
    """Response model for document operations.

    Returned after successfully adding or deleting a document, providing
    confirmation with the document ID and operation status.

    Attributes:
        id: Unique identifier (UUID) of the document.
        status: Operation result (e.g., 'indexed', 'deleted').
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"id": "abc-123-def", "status": "indexed"}}
    )

    id: str = Field(..., description="Unique document identifier")
    status: str = Field(..., description="Operation status")


class BatchDocumentResponse(BaseModel):
    """Response model for batch document operations.

    Returned after successfully adding or deleting multiple documents, providing
    the list of affected document IDs and the operation status.

    Attributes:
        ids: Unique identifiers (UUIDs) of the affected documents.
        status: Operation result (e.g., 'indexed', 'deleted').
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"ids": ["abc-123-def", "ghi-456-jkl"], "status": "indexed"}
        }
    )

    ids: list[str] = Field(..., description="Unique document identifiers")
    status: str = Field(..., description="Operation status")


class BatchDocumentInput(BaseModel):
    """Input model for adding multiple documents in a single batch request.

    Accepts a list of document inputs (content + optional metadata) and indexes
    them all in one operation.

    Attributes:
        documents: List of documents to add; must contain between 1 and
            MAX_BATCH_SIZE entries.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [
                    {"content": "First document text", "metadata": {"author": "Alice"}},
                    {"content": "Second document text", "metadata": {"author": "Bob"}},
                ]
            }
        }
    )

    documents: list[DocumentInput] = Field(
        ...,
        min_length=1,
        max_length=VFGConfig.MAX_BATCH_SIZE,
        description="Documents to add",
    )


class BatchDeleteInput(BaseModel):
    """Input model for deleting multiple documents by ID in a single batch request.

    Accepts a list of document IDs and removes all matching documents in one
    operation. IDs that do not exist are silently ignored.

    Attributes:
        ids: List of document IDs to delete; must contain between 1 and
            MAX_BATCH_SIZE entries.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"ids": ["abc-123-def", "ghi-456-jkl"]}}
    )

    ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=VFGConfig.MAX_BATCH_SIZE,
        description="Document IDs to delete",
    )


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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "abc-123-def",
                "content": "Machine learning is a subset of artificial intelligence...",
                "metadata": {"source": "textbook.pdf", "chunk_index": 0},
            }
        }
    )

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document text content")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Document metadata"
    )
