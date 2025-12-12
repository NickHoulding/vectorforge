from pydantic import BaseModel, Field


class FileUploadResponse(BaseModel):
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
    filenames: list[str] = Field(..., description="List of indexed filenames")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "filenames": ["textbook.pdf", "research_paper.pdf", "notes.txt"]
            }
        }
