from fastapi import FastAPI, UploadFile, HTTPException, File, status
import uvicorn

from doc_processor import extract_file_content, chunk_text 
from models import *
from vector_engine import VectorEngine


API_PORT = 3001
app = FastAPI(title="VectorForge API")
engine = VectorEngine()


# =============================================================================
# File Management Endpoints 
# =============================================================================

@app.get('/file/list', response_model=FileListResponse)
def list_files():
    """List file names of all files"""
    try:
        return FileListResponse(
            filenames=engine.list_files()
        )

    except Exception as e:
        print(f"Unexpected error: {e}")

@app.post('/file/upload', status_code=status.HTTP_201_CREATED, response_model=FileUploadResponse)
async def upload_file(file: UploadFile):
    """Upload a file"""
    try:
        doc_ids: list[str] = []
        text: str = await extract_file_content(file)
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
    
    except HTTPException as e:
        print(f"HTTPException: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

@app.delete('/file/delete/{filename}', response_model=FileDeleteResponse)
def delete_file(filename: str):
    """Delete all chunks associated with the given file"""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )


# =============================================================================
# Document Management Endpoints 
# =============================================================================

@app.get('/doc/{doc_id}', status_code=status.HTTP_200_OK, response_model=DocumentDetail)
def get_doc(doc_id: str):
    """Retrieve a single doc"""
    try:
        doc = engine.get_doc(doc_id=doc_id)

        if not doc:
            raise HTTPException(
                status_code=404,
                detail="Doc not found"
            )

        return DocumentDetail(
            id=doc_id,
            content=doc["content"],
            metadata=doc["metadata"]
        )
    
    except HTTPException as e:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.post('/doc/add', status_code=status.HTTP_201_CREATED, response_model=DocumentResponse)
def add_doc(doc: DocumentInput):
    """Add a single, pre-extracted document"""
    try:
        doc_id = engine.add_doc(
            content=doc.content,
            metadata=doc.metadata
        )

        return DocumentResponse(
            id=doc_id,
            status="indexed"
        )

    except Exception as e:
        print(f"Unexpected error: {e}")

@app.delete('/doc/{doc_id}', response_model=DocumentResponse)
def delete_doc(doc_id: str):
    """Remove a single doc"""
    try:
        result: bool = engine.remove_doc(doc_id)

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Doc not found"
            )
        
        return DocumentResponse(
            id=doc_id,
            status="deleted"
        )
    
    except HTTPException as e:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# =============================================================================
# Search Endpoints
# =============================================================================

@app.post('/search', response_model=SearchResponse)
def search(search_params: SearchQuery):
    """Perform a search on the index"""
    try:
        results = engine.search(
            query=search_params.query, 
            top_k=search_params.top_k
        )

        return SearchResponse(
            query=search_params.query,
            results=results,
            count=len(results)
        )
    
    except Exception as e:
        print(f"Unexpected error: {e}")


# =============================================================================
# Index Management Endpoints
# =============================================================================

@app.get('/index/stats')
def get_index_stats():
    """Index statistics (size, doc count, etc)"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )

@app.post('/index/build')
def build_index():
    """Build/rebuild index"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )

@app.post('/index/save')
def save_index():
    """Persist to disk"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )

@app.post('/index/load')
def load_index():
    """Load from disk"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )


# =============================================================================
# Health and Metrics Endpoints
# =============================================================================

@app.get('/health')
def check_health():
    """API health check"""
    return {
        "status": "healthy"
    }

@app.get('/metrics')
def get_performance_metrics():
    """Performance metrics"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )


if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=API_PORT)
