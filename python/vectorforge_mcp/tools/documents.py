from vectorforge import api
from vectorforge.models.documents import DocumentInput

from ..decorators import handle_api_errors
from ..instance import mcp
from ..utils import build_success_response


@mcp.tool(
    description="Fetch document content and metadata by ID. Use to verify stored content, inspect search results, or retrieve metadata."
)
@handle_api_errors
def get_document(doc_id: str) -> dict:
    """Retrieve a single document by ID.
    
    Args:
        doc_id: Unique document identifier (UUID).
        
    Returns:
        Dictionary with document ID, content, and metadata.
    """
    response = api.get_doc(doc_id)
    return build_success_response(response)


@mcp.tool(
    description="Index text content for semantic search. Generates embeddings automatically. Optionally add metadata for organization and filtering."
)
@handle_api_errors
def add_document(content: str, metadata: dict | None = None) -> dict:
    """Add a single document to the index.
    
    Args:
        content: The document text content to index (required, non-empty).
        metadata: Optional metadata dictionary (e.g., {"source": "email", "date": "2026-01-20"}).
        
    Returns:
        Dictionary with created document ID and status.
    """
    doc_input = DocumentInput(content=content, metadata=metadata)
    response = api.add_doc(doc_input)
    return build_success_response(response)


@mcp.tool(
    description="Permanently remove a document and its embeddings from the index. Cannot be undone."
)
@handle_api_errors
def delete_document(doc_id: str) -> dict:
    """Delete a single document by ID.
    
    Args:
        doc_id: Unique document identifier (UUID) to permanently delete.
        
    Returns:
        Dictionary with document ID and deletion status.
    """
    response = api.delete_doc(doc_id=doc_id)
    return build_success_response(response)
