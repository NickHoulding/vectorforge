from fastapi import HTTPException
from pydantic import ValidationError

from vectorforge import api
from vectorforge.models.documents import DocumentInput

from ..server import mcp


@mcp.tool
def get_document(doc_id: str) -> dict:
    """Retrieve a single document by ID.
    
    Args:
        doc_id: Unique document identifier.
        
    Returns:
        Dictionary with document ID, content, and metadata.
    """
    try:
        response = api.get_doc(doc_id)

        return {
            "success": True,
            "data": {
                "doc_id": response.id,
                "content": response.content,
                "metadata": response.metadata
            }
        }

    except HTTPException as e:
        return {
            "success": False,
            "error": e.detail,
            "details": e.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Operation failed",
            "details": str(e)
        }


@mcp.tool
def add_document(content: str, metadata: dict | None = None) -> dict:
    """Add a single document to the index.
    
    Args:
        content: The document text content to index.
        metadata: Optional metadata dictionary for the document.
        
    Returns:
        Dictionary with created document ID and status.
    """
    try:
        doc_input = DocumentInput(
            content=content,
            metadata=metadata
        )
        response = api.add_doc(doc_input)

        return {
            "success": True,
            "status": response.status,
            "doc_id": response.id
        }

    except HTTPException as e:
        return {
            "success": False,
            "error": e.detail,
            "details": e.status_code
        }
    except ValidationError as e:
        return {
            "success": False,
            "error": "Invalid input",
            "details": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Operation failed",
            "details": str(e)
        }


@mcp.tool
def delete_document(doc_id: str) -> dict:
    """Delete a single document by ID.
    
    Args:
        doc_id: Unique document identifier to delete.
        
    Returns:
        Dictionary with document ID and deletion status.
    """
    try:
        response = api.delete_doc(doc_id=doc_id)
        
        return {
            "success": True,
            "status": response.status,
            "id": response.id
        }

    except HTTPException as e:
        return {
            "success": False,
            "error": e.detail,
            "details": e.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Operation failed",
            "details": str(e)
        }
