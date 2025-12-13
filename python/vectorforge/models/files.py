from pydantic import BaseModel, Field


class FileUploadResponse(BaseModel):
    """Response model for file upload operations.
    
    Provides confirmation and details after successfully uploading and processing
    a file. Includes information about how the file was chunked and the IDs of
    all created document chunks.
    
    Attributes:
        filename: Name of the uploaded file.
        chunks_created: Number of text chunks the file was split into.
        doc_ids: List of unique IDs for each created document chunk.
        status: Upload operation result (typically 'indexed').
    """
    filename: str = Field(..., description="Name of uploaded file")
    chunks_created: int = Field(..., description="Number of text chunks created")
    doc_ids: list[str] = Field(..., description="List of document IDs created")
    status: str = Field(..., description="Upload operation status")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "filename": "textbook.pdf",
                "chunks_created": 5,
                "document_ids": ["abc-123", "def-456", "ghi-789", ...],
                "status": "indexed"
            }
        }

class FileDeleteResponse(BaseModel):
    """Response model for file deletion operations.
    
    Provides confirmation and details after deleting all chunks associated with
    a source file. Includes the count of deleted chunks and their IDs.
    
    Attributes:
        filename: Name of the deleted file.
        chunks_deleted: Number of document chunks that were deleted.
        doc_ids: List of unique IDs for each deleted document chunk.
        status: Deletion operation result (e.g., 'deleted', 'not_found').
    """
    filename: str = Field(..., description="Name of deleted file")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")
    doc_ids: list[str] = Field(..., description="List of deleted document IDs")
    status: str = Field(..., description="Delete operation status")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "filename": "textbook.pdf",
                "chunks_deleted": 5,
                "document_ids": ["abc-123", "def-456", "ghi-789", ...],
                "status": "deleted"
                ""
            }
        }

class FileListResponse(BaseModel):
    """Response model listing all indexed files.
    
    Returns a sorted list of all unique source filenames that have been uploaded
    and indexed in the vector database. Useful for discovering what files are
    currently in the system.
    
    Attributes:
        filenames: Sorted list of unique source file names.
    """
    filenames: list[str] = Field(..., description="List of indexed filenames")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "filenames": ["textbook.pdf", "research_paper.pdf", "notes.txt"]
            }
        }
