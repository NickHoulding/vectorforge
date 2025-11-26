from fastapi import FastAPI, File, UploadFile, HTTPException, status
from typing import List
import uvicorn

from doc_processor import extract_file_content, chunk_text 
from models import Document, SearchQuery, SearchResult
from vector_engine import VectorEngine

API_VERSION = "0.1.0"
API_PORT = 3001
app = FastAPI(
    title="VectorForge API",
    version=API_VERSION
)
engine = VectorEngine()


# =============================================================================
# Document Management Endpoints 
# =============================================================================

@app.post('/docs/upload', status_code=status.HTTP_201_CREATED)
async def doc(file: UploadFile = File(...)):
    """Upload a file"""
    try:
        doc_ids: List[str] = []
        text: str = await extract_file_content(file)
        chunks: List[str] = chunk_text(text)

        for i, chunk in enumerate(chunks):
            doc_id: str = engine.add_doc(
                content=chunk,
                metadata={
                    "source_file": file.filename,
                    "chunk_index": i
                }
            )
            doc_ids.append(doc_id)

        return {
            "doc_ids": doc_ids
        }
    
    except HTTPException as e:
        print(f"HTTPException: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

@app.post('/docs/batch')
def doc_batch():
    """Query for similar docs"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )

@app.delete('/docs/{doc_id}')
def doc_delete(doc_id: str):
    """Remove a doc"""
    try:
        if engine.remove_doc(doc_id):
            message = f"Doc: {doc_id} deleted successfully"
        else:
            message = f"Issue encountered deleting doc: {doc_id}"
        
        return {
            "message": message
        }
    
    except Exception as e:
        print(f"Unexpected error: {e}")

@app.get('/docs/{doc_id}')
def doc_get(doc_id: str):
    """Retrieve a doc"""
    try: 
        doc = engine.get_doc(doc_id=doc_id)

        if doc:
            message = f"Doc successfully retreived"
        else:
            message = f"Doc: {doc_id} not found"

        return {
            "message": message,
            "doc": doc
        }

    except Exception as e:
        print(f"Unexpected error: {e}")

# =============================================================================
# Search Endpoints
# =============================================================================

@app.post('/search')
def search():
    """Single search query"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )

@app.post('/search/batch')
def search_batch():
    """Search multiple queries at once"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )


# =============================================================================
# Index Management Endpoints
# =============================================================================

@app.post('/index/build')
def build_index():
    """Build/rebuild index"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )
    
@app.get('/index/stats')
def get_index_stats():
    """Index statistics (size, doc count, etc)"""
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
        "status": "healthy",
        "version": API_VERSION,
    }

@app.get('/metrics')
def get_performance_metrics():
    """Performance metrics"""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint not yet implemented"
    )


if __name__ == "__main__":
    uvicorn.run(
        app=app, 
        host='0.0.0.0', 
        port=API_PORT
    )
