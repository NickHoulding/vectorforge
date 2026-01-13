from ..server import mcp


@mcp.tool
def list_files() -> list[str]:
    """List all indexed files in the vector store.
    
    Returns:
        List of filenames that have been uploaded and indexed.
    """
    raise NotImplementedError


@mcp.tool
def upload_file(file_path: str) -> dict:
    """Upload and index a file.
    
    Args:
        file_path: Path to the file to upload and index.
        
    Returns:
        Dictionary with upload status, filename, chunks created, and document IDs.
    """
    raise NotImplementedError


@mcp.tool
def delete_file(filename: str) -> dict:
    """Delete all chunks associated with a file.
    
    Args:
        filename: Name of the source file to delete.
        
    Returns:
        Dictionary with deletion status, filename, chunks deleted, and document IDs.
    """
    raise NotImplementedError
