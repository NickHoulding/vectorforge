from ..server import mcp


@mcp.tool
def get_document(doc_id: str) -> dict:
    """Retrieve a single document by ID.
    
    Args:
        doc_id: Unique document identifier.
        
    Returns:
        Dictionary with document ID, content, and metadata.
    """
    raise NotImplementedError


@mcp.tool
def add_document(content: str, metadata: dict | None = None) -> dict:
    """Add a single document to the index.
    
    Args:
        content: The document text content to index.
        metadata: Optional metadata dictionary for the document.
        
    Returns:
        Dictionary with created document ID and status.
    """
    raise NotImplementedError


@mcp.tool
def delete_document(doc_id: str) -> dict:
    """Delete a single document by ID.
    
    Args:
        doc_id: Unique document identifier to delete.
        
    Returns:
        Dictionary with document ID and deletion status.
    """
    raise NotImplementedError
