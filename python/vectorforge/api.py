from fastapi import FastAPI, UploadFile, HTTPException, status
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
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

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
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.delete('/file/delete/{filename}', response_model=FileDeleteResponse)
def delete_file(filename: str):
    """Delete all chunks associated with the given file"""
    try:
        deletion_metrics = engine.delete_file(filename=filename)

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
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
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
        if not doc.metadata:
            doc.metadata = {
                "chunk_index": 0
            }

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
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.delete('/doc/{doc_id}', response_model=DocumentResponse)
def delete_doc(doc_id: str):
    """Remove a single doc"""
    try:
        result: bool = engine.delete_doc(doc_id)

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
        results: list[SearchResult] = engine.search(
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
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# =============================================================================
# Index Management Endpoints
# =============================================================================

@app.get('/index/stats')
def get_index_stats():
    """
    get quick index statistics.

    Lightweight endpoint for checking index health and size.
    For detailed metrics, use GET /metrics.
    """
    try:
        stats = engine.get_index_stats()

        return IndexStatsResponse(
            total_documents=stats["total_documents"],
            total_embeddings=stats["total_embeddings"],
            deleted_documents=stats["deleted_documents"],
            deleted_ratio=stats["deleted_ratio"],
            needs_compaction=stats["needs_compaction"],
            embedding_dimension=stats["embedding_dimension"]
        )
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.post('/index/build')
def build_index():
    """Build/rebuild index"""
    try:
        engine.build()
        stats = engine.get_index_stats()

        return IndexStatsResponse(
            total_documents=stats["total_documents"],
            total_embeddings=stats["total_embeddings"],
            deleted_documents=stats["deleted_documents"],
            deleted_ratio=stats["deleted_ratio"],
            needs_compaction=stats["needs_compaction"],
            embedding_dimension=stats["embedding_dimension"]
        )

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.post('/index/save', response_model=IndexSaveResponse)
def save_index(directory: str = "./data"):
    """Persist to disk"""
    try:
        save_metrics = engine.save(directory=directory)

        return IndexSaveResponse(
            status=save_metrics["status"],
            directory=save_metrics["directory"],
            metadata_size_mb=save_metrics["metadata_size_mb"],
            embeddings_size_mb=save_metrics["embeddings_size_mb"],
            total_size_mb=save_metrics["total_size_mb"],
            documents_saved=save_metrics["documents_saved"],
            embeddings_saved=save_metrics["embeddings_saved"]
        )

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.post('/index/load', response_model=IndexLoadResponse)
def load_index():
    """Load from disk"""
    try:
        load_metrics = engine.load()

        return IndexLoadResponse(
            status=load_metrics["status"],
            directory=load_metrics["directory"],
            documents_loaded=load_metrics["documents_loaded"],
            embeddings_loaded=load_metrics["embeddings_loaded"],
            deleted_docs=load_metrics["deleted_docs"],
            version=load_metrics["version"]
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Index files not found: {str(e)}"
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
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

@app.get('/metrics', response_model=MetricsResponse)
def get_performance_metrics():
    """Get comprehensive system metrics"""
    metrics = engine.get_metrics()
    
    index_metrics = IndexMetrics(
        total_documents=metrics["active_documents"],
        total_embeddings=metrics["total_embeddings"],
        deleted_documents=metrics["docs_deleted"],
        deleted_ratio=metrics["deleted_ratio"],
        needs_compaction=metrics["needs_compaction"],
        compact_threshold=metrics["compact_threshold"]
    )
    performance_metrics = PerformanceMetrics(
        total_queries=metrics["total_queries"],
        avg_query_time_ms=metrics["avg_query_time_ms"],
        total_query_time_ms=metrics["total_query_time_ms"],
        min_query_time_ms=metrics["min_query_time_ms"],
        max_query_time_ms=metrics["max_query_time_ms"],
        p50_query_time_ms=metrics["p50_query_time_ms"],
        p95_query_time_ms=metrics["p95_query_time_ms"],
        p99_query_time_ms=metrics["p99_query_time_ms"]
    )
    usage_metrics = UsageMetrics(
        documents_added=metrics["docs_added"],
        documents_deleted=metrics["docs_deleted"],
        compactions_performed=metrics["compactions_performed"],
        chunks_created=metrics["chunks_created"],
        files_uploaded=metrics["files_uploaded"]
    )
    memory_metrics = MemoryMetrics(
        embeddings_mb=metrics["embeddings_mb"],
        documents_mb=metrics["documents_mb"],
        total_mb=metrics["total_mb"]
    )
    timestamp_metrics = TimestampMetrics(
        engine_created_at=metrics["created_at"],
        last_query_at=metrics["last_query_at"],
        last_document_added_at=metrics["last_doc_added_at"],
        last_compaction_at=metrics["last_compaction_at"],
        last_file_uploaded_at=metrics["last_file_uploaded_at"]
    )
    system_info = SystemInfo(
        model_name=metrics["model_name"],
        model_dimension=metrics["model_dimension"],
        uptime_seconds=metrics["uptime_seconds"]
    )

    return MetricsResponse(
        index=index_metrics,
        performance=performance_metrics,
        usage=usage_metrics,
        memory=memory_metrics,
        timestamps=timestamp_metrics,
        system=system_info
    )


if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=API_PORT)
