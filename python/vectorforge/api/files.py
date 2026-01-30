"""File Management Endpoints"""

from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, status

from vectorforge.api import engine
from vectorforge.api.decorators import handle_api_errors
from vectorforge.config import VFGConfig
from vectorforge.doc_processor import chunk_text, extract_file_content
from vectorforge.models import FileDeleteResponse, FileListResponse, FileUploadResponse


router: APIRouter = APIRouter()

@router.get('/file/list', response_model=FileListResponse)
@handle_api_errors
def list_files() -> FileListResponse:
    """
    List all indexed files
    
    Returns a list of all source filenames that have been uploaded and indexed
    in the vector store.
    
    Returns:
        FileListResponse: Object containing array of filenames
        
    Raises:
        HTTPException: 500 if internal server error occurs
    """
    filenames: list[str] = engine.list_files()

    return FileListResponse(
        filenames=filenames
    )


@router.post('/file/upload', status_code=status.HTTP_201_CREATED, response_model=FileUploadResponse)
@handle_api_errors
async def upload_file(file: UploadFile) -> FileUploadResponse:
    """
    Upload and index a file
    
    Uploads a file, extracts its text content, chunks it into smaller pieces,
    and indexes each chunk as a separate document with embeddings for semantic search.
    
    Args:
        file (UploadFile): File to upload (supports PDF, TXT, DOCX, MD, etc.)
        
    Returns:
        FileUploadResponse: Upload confirmation with chunk count and document IDs
        
    Raises:
        HTTPException: 400 if file format is missing, too long, or unsupported
        HTTPException: 400 if file has no content
        HTTPException: 500 if processing or indexing fails
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    if len(file.filename) > VFGConfig.MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Filename too long: {file.filename[:25]}..."
        )

    doc_ids: list[str] = []
    text: str = await extract_file_content(file)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Uploaded file(s) contains no content"
        )

    chunks: list[str] = chunk_text(text)

    for i, chunk in enumerate(chunks):
        doc_id: str = engine.add_doc(
            content=chunk,
            metadata={
                "source_file": file.filename,
                "chunk_index": i
            }
        )
        doc_ids.append(doc_id)

    return FileUploadResponse(
        filename=file.filename or "",
        chunks_created=len(doc_ids),
        doc_ids=doc_ids,
        status="indexed"
    )


@router.delete('/file/delete/{filename}', response_model=FileDeleteResponse)
@handle_api_errors
def delete_file(filename: str) -> FileDeleteResponse:
    """
    Delete all chunks associated with a file
    
    Removes all document chunks that were created from the specified source file,
    including their embeddings and metadata.
    
    Args:
        filename (str): Name of the source file to delete
        
    Returns:
        FileDeleteResponse: Deletion confirmation with count of chunks removed
        
    Raises:
        HTTPException: 404 if no documents found for the specified filename
        HTTPException: 500 if deletion fails
    """
    deletion_metrics: dict[str, Any] = engine.delete_file(filename=filename)

    if deletion_metrics["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"No documents found for file: {filename}"
        )

    return FileDeleteResponse(
        status=deletion_metrics["status"],
        filename=deletion_metrics["filename"],
        chunks_deleted=deletion_metrics["chunks_deleted"],
        doc_ids=deletion_metrics["doc_ids"],
    )
