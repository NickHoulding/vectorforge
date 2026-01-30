"""Document Management Endpoints"""

from typing import Any

from fastapi import APIRouter, HTTPException, status

from vectorforge.api import engine
from vectorforge.api.decorators import handle_api_errors
from vectorforge.models import DocumentDetail, DocumentInput, DocumentResponse


router: APIRouter = APIRouter()

@router.get('/doc/{doc_id}', status_code=status.HTTP_200_OK, response_model=DocumentDetail)
@handle_api_errors
def get_doc(doc_id: str) -> DocumentDetail:
    """
    Retrieve a single document by ID
    
    Fetches the content and metadata for a specific document chunk using its
    unique identifier.
    
    Args:
        doc_id (str): Unique document identifier
        
    Returns:
        DocumentDetail: Document content, metadata, and ID
        
    Raises:
        HTTPException: 404 if document not found
        HTTPException: 500 if retrieval fails
    """
    doc: dict[str, Any] | None = engine.get_doc(doc_id=doc_id)

    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Doc not found"
        )

    return DocumentDetail(
        id=doc_id,
        content=doc["content"],
        metadata=doc["metadata"]
    )


@router.post('/doc/add', status_code=status.HTTP_201_CREATED, response_model=DocumentResponse)
@handle_api_errors
def add_doc(doc: DocumentInput) -> DocumentResponse:
    """
    Add a single pre-extracted document
    
    Indexes a document that has already been extracted and chunked externally.
    Useful for adding custom content without file upload.
    
    Args:
        doc (DocumentInput): Document with content and optional metadata
        
    Returns:
        DocumentResponse: Created document ID and status
        
    Raises:
        HTTPException: 500 if indexing fails
    """
    doc_id: str = engine.add_doc(
        content=doc.content,
        metadata=doc.metadata
    )

    return DocumentResponse(
        id=doc_id,
        status="indexed"
    )


@router.delete('/doc/{doc_id}', response_model=DocumentResponse)
@handle_api_errors
def delete_doc(doc_id: str) -> DocumentResponse:
    """
    Delete a single document by ID
    
    Removes a specific document chunk and its embedding from the index.
    
    Args:
        doc_id (str): Unique document identifier to delete
        
    Returns:
        DocumentResponse: Deletion confirmation with document ID
        
    Raises:
        HTTPException: 404 if document not found
        HTTPException: 500 if deletion fails
    """
    delete_success: bool = engine.delete_doc(doc_id)

    if not delete_success:
        raise HTTPException(
            status_code=404,
            detail="Doc not found"
        )
    
    return DocumentResponse(
        id=doc_id,
        status="deleted"
    )
