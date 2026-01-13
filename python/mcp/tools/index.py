from ..server import mcp


@mcp.tool
def get_index_stats() -> dict:
    """Get quick index statistics.
    
    Returns:
        Dictionary with index statistics including document counts, embedding dimension,
        and compaction status.
    """
    raise NotImplementedError


@mcp.tool
def build_index() -> dict:
    """Build or rebuild the vector index.
    
    Returns:
        Dictionary with updated index statistics after rebuild.
    """
    raise NotImplementedError


@mcp.tool
def save_index(directory: str | None = None) -> dict:
    """Persist index to disk.
    
    Args:
        directory: Optional directory path for saving (default: './data').
        
    Returns:
        Dictionary with save confirmation, file sizes, and document counts.
    """
    raise NotImplementedError


@mcp.tool
def load_index(directory: str | None = None) -> dict:
    """Load index from disk.
    
    Args:
        directory: Optional directory path to load from (default: './data').
        
    Returns:
        Dictionary with load confirmation, counts, and version information.
    """
    raise NotImplementedError


@mcp.tool
def get_metrics() -> dict:
    """Get comprehensive system metrics.
    
    Returns:
        Dictionary with detailed performance, usage, memory, timestamp, and system metrics.
    """
    raise NotImplementedError
