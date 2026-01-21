import httpx

from fastapi import HTTPException
from pydantic import ValidationError

from vectorforge import api
from vectorforge.models.documents import DocumentInput

from ..instance import mcp
from ..utils import build_error_response, build_success_response


@mcp.tool(
    description="Fetch document content and metadata by ID. Use to verify stored content, inspect search results, or retrieve metadata."
)
def get_document(doc_id: str) -> dict:
    """Retrieve a single document by ID.
    
    Args:
        doc_id: Unique document identifier (UUID).
        
    Returns:
        Dictionary with document ID, content, and metadata.
    """
    try:
        response = api.get_doc(doc_id)
        return build_success_response(response)

    except ConnectionRefusedError:
        return build_error_response(
            Exception("VectorForge API is not available"),
            details="Connection refused - check if API is running"
        )
    except (ConnectionError, httpx.ConnectError):
        return build_error_response(
            Exception("Network error"),
            details="Unable to connect to VectorForge API"
        )
    except (TimeoutError, httpx.TimeoutException):
        return build_error_response(
            Exception("Request timeout"),
            details="VectorForge API request timed out"
        )
    except HTTPException as e:
        if e.status_code == 422:
            return build_error_response(
                Exception("API version mismatch"),
                details="Request format incompatible with API version"
            )
        return build_error_response(e, details=e.status_code)
    except Exception as e:
        return build_error_response(
            Exception("Operation failed"), 
            details=str(e)
        )


@mcp.tool(
    description="Index text content for semantic search. Generates embeddings automatically. Optionally add metadata for organization and filtering."
)
def add_document(content: str, metadata: dict | None = None) -> dict:
    """Add a single document to the index.
    
    Args:
        content: The document text content to index (required, non-empty).
        metadata: Optional metadata dictionary (e.g., {"source": "email", "date": "2026-01-20"}).
        
    Returns:
        Dictionary with created document ID and status.
    """
    try:
        doc_input = DocumentInput(
            content=content,
            metadata=metadata
        )
        response = api.add_doc(doc_input)
        return build_success_response(response)

    except ConnectionRefusedError:
        return build_error_response(
            Exception("VectorForge API is not available"),
            details="Connection refused - check if API is running"
        )
    except (ConnectionError, httpx.ConnectError):
        return build_error_response(
            Exception("Network error"),
            details="Unable to connect to VectorForge API"
        )
    except (TimeoutError, httpx.TimeoutException):
        return build_error_response(
            Exception("Request timeout"),
            details="VectorForge API request timed out"
        )
    except HTTPException as e:
        if e.status_code == 422:
            return build_error_response(
                Exception("API version mismatch"),
                details="Request format incompatible with API version"
            )
        return build_error_response(e, details=e.status_code)
    except ValidationError as e:
        return build_error_response(
            Exception("Invalid input"), 
            details=str(e)
        )
    except Exception as e:
        return build_error_response(
            Exception("Operation failed"), 
            details=str(e)
        )


@mcp.tool(
    description="Permanently remove a document and its embeddings from the index. Cannot be undone."
)
def delete_document(doc_id: str) -> dict:
    """Delete a single document by ID.
    
    Args:
        doc_id: Unique document identifier (UUID) to permanently delete.
        
    Returns:
        Dictionary with document ID and deletion status.
    """
    try:
        response = api.delete_doc(doc_id=doc_id)
        return build_success_response(response)

    except ConnectionRefusedError:
        return build_error_response(
            Exception("VectorForge API is not available"),
            details="Connection refused - check if API is running"
        )
    except (ConnectionError, httpx.ConnectError):
        return build_error_response(
            Exception("Network error"),
            details="Unable to connect to VectorForge API"
        )
    except (TimeoutError, httpx.TimeoutException):
        return build_error_response(
            Exception("Request timeout"),
            details="VectorForge API request timed out"
        )
    except HTTPException as e:
        if e.status_code == 422:
            return build_error_response(
                Exception("API version mismatch"),
                details="Request format incompatible with API version"
            )
        return build_error_response(e, details=e.status_code)
    except Exception as e:
        return build_error_response(
            Exception("Operation failed"), 
            details=str(e)
        )
