"""FastAPI router for document CRUD endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, status

from vectorforge.api import manager
from vectorforge.api.decorators import handle_api_errors, require_collection
from vectorforge.models import (
    BatchDeleteInput,
    BatchDocumentInput,
    BatchDocumentResponse,
    DocumentDetail,
    DocumentInput,
    DocumentResponse,
)

router: APIRouter = APIRouter()


@router.get(
    "/collections/{collection_name}/documents/{doc_id}",
    status_code=status.HTTP_200_OK,
    response_model=DocumentDetail,
)
@require_collection
@handle_api_errors
def get_doc(collection_name: str, doc_id: str) -> DocumentDetail:
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
    response_model=DocumentResponse,
)
@require_collection
@handle_api_errors
def add_doc(collection_name: str, doc: DocumentInput) -> DocumentResponse:
    """
    Add a single pre-extracted document to a collection

    Indexes a document that has already been extracted and chunked externally.
    Useful for adding custom content without file upload.

    Args:
        collection_name: Name of the collection
        doc: Document with content and optional metadata

    Returns:
        DocumentResponse: Created document ID and status

    Raises:
        HTTPException: 404 if collection not found
        HTTPException: 500 if indexing fails

    Example:
        POST /collections/customer_docs/documents
        {"content": "Vector databases enable semantic search",
         "metadata": {"source": "manual", "category": "tech"}}
    """
    engine = manager.get_engine(collection_name)
    doc_id: str = engine.add_docs([{"content": doc.content, "metadata": doc.metadata}])[
        0
    ]

    return DocumentResponse(id=doc_id, status="indexed")


@router.post(
    "/collections/{collection_name}/documents/batch",
    status_code=status.HTTP_201_CREATED,
    response_model=BatchDocumentResponse,
)
@require_collection
@handle_api_errors
def add_docs_batch(
    collection_name: str, body: BatchDocumentInput
) -> BatchDocumentResponse:
    """
    Add multiple documents to a collection in a single batch request

    Indexes each document individually, returning all assigned IDs. Documents
    are validated before any are persisted; if any document fails validation
    the entire request returns an error.

    Args:
        collection_name: Name of the collection
        body: Batch of documents (1–MAX_BATCH_SIZE entries)

    Returns:
        BatchDocumentResponse: List of created document IDs and status

    Raises:
        HTTPException: 404 if collection not found
        HTTPException: 422 if any document fails validation

    Example:
        POST /collections/customer_docs/documents/batch
        {"documents": [{"content": "First doc", "metadata": {"author": "Alice"}},
                       {"content": "Second doc", "metadata": {"author": "Bob"}}]}
    """
    engine = manager.get_engine(collection_name)
    docs: list[dict[str, Any]] = [doc.model_dump() for doc in body.documents]
    doc_ids: list[str] = engine.add_docs(docs)

    return BatchDocumentResponse(ids=doc_ids, status="indexed")


@router.delete(
    "/collections/{collection_name}/documents/{doc_id}",
    response_model=DocumentResponse,
)
@require_collection
@handle_api_errors
def delete_doc(collection_name: str, doc_id: str) -> DocumentResponse:
    """
    Delete a single document by ID from a collection

    Removes a specific document chunk and its embedding from the collection.

    Args:
        collection_name: Name of the collection
        doc_id: Unique document identifier to delete

    Returns:
        DocumentResponse: Deletion confirmation with document ID

    Raises:
        HTTPException: 404 if collection or document not found
        HTTPException: 500 if deletion fails

    Example:
        DELETE /collections/customer_docs/documents/doc_123
    """
    engine = manager.get_engine(collection_name)
    deleted: list[str] = engine.delete_docs([doc_id])

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(id=doc_id, status="deleted")


@router.delete(
    "/collections/{collection_name}/documents",
    response_model=BatchDocumentResponse,
)
@require_collection
@handle_api_errors
def delete_docs_batch(
    collection_name: str, body: BatchDeleteInput
) -> BatchDocumentResponse:
    """
    Delete multiple documents by ID from a collection in a single batch request

    Removes all matching documents and their embeddings in one ChromaDB call.
    IDs that do not exist are silently ignored; only the IDs that were actually
    deleted are returned.

    Args:
        collection_name: Name of the collection
        body: List of document IDs to delete (1-MAX_BATCH_SIZE entries)

    Returns:
        BatchDocumentResponse: List of deleted document IDs and status

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

    return BatchDocumentResponse(ids=deleted, status="deleted")
