"""FastAPI router for document CRUD endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, status

from vectorforge.api import manager
from vectorforge.api.decorators import handle_api_errors, require_collection
from vectorforge.models import (
    DocumentDetail,
    DocumentIdsInput,
    DocumentsInput,
    DocumentsResponse,
)

router: APIRouter = APIRouter()


@router.get(
    "/collections/{collection_name}/documents/{doc_id}",
    status_code=status.HTTP_200_OK,
    response_model=DocumentDetail,
)
@require_collection
@handle_api_errors
def get_document(collection_name: str, doc_id: str) -> DocumentDetail:
    """
    Retrieve a single document by ID from a collection

    Fetches the content and metadata for a specific document chunk using its
    unique identifier from the specified collection.

    Args:
        collection_name: Name of the collection
        doc_id: Unique document identifier

    Returns:
        DocumentDetail: Document content, metadata, and ID

    Raises:
        HTTPException: 404 if collection or document not found
        HTTPException: 500 if retrieval fails

    Example:
        GET /collections/customer_docs/documents/doc_123
    """
    engine = manager.get_engine(collection_name)
    doc: dict[str, Any] | None = engine.get_doc(doc_id=doc_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetail(id=doc_id, content=doc["content"], metadata=doc["metadata"])


@router.post(
    "/collections/{collection_name}/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentsResponse,
)
@require_collection
@handle_api_errors
def add_documents(collection_name: str, body: DocumentsInput) -> DocumentsResponse:
    """
    Add one or more pre-extracted documents to a collection

    Indexes documents that have already been extracted and chunked externally.
    Useful for adding custom content without file upload. Batch-encodes all
    document embeddings in a single model.encode() call, then persists all
    documents atomically. Documents are validated before any are persisted; if
    any document fails validation the entire request returns an error.

    Args:
        collection_name: Name of the collection
        body: Documents to add (1-MAX_BATCH_SIZE entries)

    Returns:
        DocumentsResponse: List of created document IDs and status

    Raises:
        HTTPException: 404 if collection not found
        HTTPException: 422 if any document fails validation

    Example:
        POST /collections/customer_docs/documents
        {"documents": [{"content": "Vector databases enable semantic search",
                        "metadata": {"source": "manual", "category": "tech"}}]}
    """
    engine = manager.get_engine(collection_name)
    docs: list[dict[str, Any]] = [doc.model_dump() for doc in body.documents]
    doc_ids: list[str] = engine.add_docs(docs)

    return DocumentsResponse(ids=doc_ids, status="indexed")


@router.delete(
    "/collections/{collection_name}/documents",
    response_model=DocumentsResponse,
)
@require_collection
@handle_api_errors
def delete_documents(collection_name: str, body: DocumentIdsInput) -> DocumentsResponse:
    """
    Delete one or more documents by ID from a collection

    Removes all matching documents and their embeddings in one ChromaDB call.
    IDs that do not exist are silently ignored; only the IDs that were actually
    deleted are returned.

    Args:
        collection_name: Name of the collection
        body: Document IDs to delete (1-MAX_BATCH_SIZE entries)

    Returns:
        DocumentsResponse: List of deleted document IDs and status

    Raises:
        HTTPException: 404 if collection not found or none of the provided IDs exist
        HTTPException: 500 if deletion fails

    Example:
        DELETE /collections/customer_docs/documents
        {"ids": ["doc_123", "doc_456"]}
    """
    engine = manager.get_engine(collection_name)
    deleted: list[str] = engine.delete_docs(body.ids)

    if not deleted:
        raise HTTPException(status_code=404, detail="No matching documents found")

    return DocumentsResponse(ids=deleted, status="deleted")
