from ..server import mcp


@mcp.tool
def search_documents(query: str, top_k: int = 10) -> list[dict]:
    """Perform semantic search on indexed documents.
    
    Args:
        query: Search query string.
        top_k: Number of top results to return (default: 10).
        
    Returns:
        List of search results with document IDs, content, scores, and metadata.
    """
    raise NotImplementedError
