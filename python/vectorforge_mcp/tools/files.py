import os

import httpx

from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from vectorforge import api

from ..instance import mcp
from ..utils import build_error_response, build_success_response


@mcp.tool
def list_files() -> dict:
    """List all indexed files in the vector store.
    
    Returns:
        List of filenames that have been uploaded and indexed.
    """
    try:
        response = api.list_files()
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
async def upload_file(file_path: str) -> dict:
    """Upload and index a file.
    
    Args:
        file_path: Path to the file to upload and index.
        
    Returns:
        Dictionary with upload status, filename, chunks created, and document IDs.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            file = UploadFile(
                filename=os.path.basename(file_path),
                file=f
            )
            response = await api.upload_file(file)

        return build_success_response(response)

    except FileNotFoundError as e:
        return build_error_response(e)
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
            Exception("Upload timeout"),
            details="File upload timed out - try a smaller file"
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
def delete_file(filename: str) -> dict:
    """Delete all chunks associated with an indexed file.
    
    Args:
        filename: Name of the source file to delete.
        
    Returns:
        Dictionary with deletion status, filename, chunks deleted, and document IDs.
    """
    try:
        response = api.delete_file(filename=filename)
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
