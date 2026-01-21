import httpx

from fastapi import HTTPException

from vectorforge import api
from vectorforge.config import Config

from ..instance import mcp
from ..utils import build_error_response, build_success_response


@mcp.tool(
    description="Get lightweight index health check: document counts, embedding dimension, deletion ratio, compaction status."
)
def get_index_stats() -> dict:
    """Get quick index statistics.
    
    Returns:
        Dictionary with index statistics including document counts, embedding dimension, and compaction status.
    """
    try:
        response = api.get_index_stats()
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
    description="Rebuild entire vector index from scratch. Regenerates all embeddings. Optimizes search performance but time-intensive."
)
def build_index() -> dict:
    """Build or rebuild the vector index.
    
    Returns:
        Dictionary with updated index statistics after rebuild.
    """
    try:
        response = api.build_index()
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
            details="Index rebuild timed out - large index may take longer"
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
    description="Persist index to disk (embeddings + metadata). Enables fast recovery and reduces startup time. Returns file sizes and counts."
)
def save_index(directory: str = Config.DEFAULT_DATA_DIR) -> dict:
    """Persist index to disk.
    
    Args:
        directory: Directory path for saving (default: './data').
        
    Returns:
        Dictionary with save confirmation, file sizes, and document counts.
    """
    try:
        response = api.save_index(directory=directory)
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
            details="Save operation timed out - large index may take longer"
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
    description="Restore index from disk. Loads previously saved embeddings and metadata. Faster than rebuilding from documents."
)
def load_index(directory: str = Config.DEFAULT_DATA_DIR) -> dict:
    """Load index from disk.
    
    Args:
        directory: Directory path to load from (default: './data').
        
    Returns:
        Dictionary with load confirmation, counts, and version information.
    """
    try:
        response = api.load_index(directory=directory)
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
            details="Load operation timed out - large index may take longer"
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
