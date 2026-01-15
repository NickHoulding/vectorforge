from fastapi import HTTPException

from vectorforge import api
from vectorforge.config import Config

from ..server import mcp
from ..utils import build_error_response, build_success_response


@mcp.tool
def get_index_stats() -> dict:
    """Get quick index statistics.
    
    Returns:
        Dictionary with index statistics including document counts, embedding dimension, and compaction status.
    """
    try:
        response = api.get_index_stats()
        return build_success_response(response)

    except HTTPException as e:
        return build_error_response(e, details=e.status_code)
    except Exception as e:
        return build_error_response(Exception("Operation failed"), details=str(e))


@mcp.tool
def build_index() -> dict:
    """Build or rebuild the vector index.
    
    Returns:
        Dictionary with updated index statistics after rebuild.
    """
    try:
        response = api.build_index()
        return build_success_response(response)
    
    except HTTPException as e:
        return build_error_response(e, details=e.status_code)
    except Exception as e:
        return build_error_response(Exception("Operation failed"), details=str(e))


@mcp.tool
def save_index(directory: str = Config.DEFAULT_DATA_DIR) -> dict:
    """Persist index to disk.
    
    Args:
        directory: Optional directory path for saving (default: './data').
        
    Returns:
        Dictionary with save confirmation, file sizes, and document counts.
    """
    try:
        response = api.save_index(directory=directory)
        return build_success_response(response)
    
    except HTTPException as e:
        return build_error_response(e, details=e.status_code)
    except Exception as e:
        return build_error_response(Exception("Operation failed"), details=str(e))


@mcp.tool
def load_index(directory: str = Config.DEFAULT_DATA_DIR) -> dict:
    """Load index from disk.
    
    Args:
        directory: Optional directory path to load from (default: './data').
        
    Returns:
        Dictionary with load confirmation, counts, and version information.
    """
    try:
        response = api.load_index(directory=directory)
        return build_success_response(response)
    
    except HTTPException as e:
        return build_error_response(e, details=e.status_code)
    except Exception as e:
        return build_error_response(Exception("Operation failed"), details=str(e))


@mcp.tool
def get_metrics() -> dict:
    """Get comprehensive system metrics.
    
    Returns:
        Dictionary with detailed performance, usage, memory, timestamp, and system metrics.
    """
    try:
        response = api.get_metrics()
        return build_success_response(response)
    
    except HTTPException as e:
        return build_error_response(e, details=e.status_code)
    except Exception as e:
        return build_error_response(Exception("Operation failed"), details=str(e))
