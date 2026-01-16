import httpx

from fastapi import HTTPException
from pydantic import ValidationError

from vectorforge import api
from vectorforge.models.documents import DocumentInput

from ..instance import mcp
from ..utils import build_error_response, build_success_response


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
