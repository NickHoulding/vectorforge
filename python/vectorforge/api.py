import uvicorn

from fastapi import FastAPI, HTTPException, UploadFile, status

from vectorforge import __version__
from vectorforge.doc_processor import chunk_text, extract_file_content
from vectorforge.models import (
    DocumentDetail,
    DocumentInput,
    DocumentResponse,
    FileDeleteResponse,
    FileListResponse,
    FileUploadResponse,
    IndexLoadResponse,
    IndexMetrics,
    IndexSaveResponse,
    IndexStatsResponse,
    MemoryMetrics,
    MetricsResponse,
    PerformanceMetrics,
    SearchQuery,
    SearchResponse,
    SearchResult,
    SystemInfo,
    TimestampMetrics,
    UsageMetrics,
)
from vectorforge.vector_engine import VectorEngine


API_PORT = 3001
app = FastAPI(
    title="VectorForge API",
    version=__version__,
    description="High-performance in-memory vector database with semantic search"
)
engine = VectorEngine()


# =============================================================================
# File Management Endpoints 
# =============================================================================

# --- List Files ---

@app.get('/file/list', response_model=FileListResponse)
def list_files():
    """
    List all indexed files
    
    Returns a list of all source filenames that have been uploaded and indexed
    in the vector store.
    
    Returns:
        FileListResponse: Object containing array of filenames
        
    Raises:
        HTTPException: 500 if internal server error occurs
    """
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

# --- Upload File ---

@app.post('/file/upload', status_code=status.HTTP_201_CREATED, response_model=FileUploadResponse)
async def upload_file(file: UploadFile):
    """
    Upload and index a file
    
    Uploads a file, extracts its text content, chunks it into smaller pieces,
    and indexes each chunk as a separate document with embeddings for semantic search.
    
    Args:
        file (UploadFile): File to upload (supports PDF, TXT, DOCX, MD, etc.)
        
    Returns:
        FileUploadResponse: Upload confirmation with chunk count and document IDs
        
    Raises:
        HTTPException: 400 if file format is unsupported
        HTTPException: 500 if processing or indexing fails
    """
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
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

# --- Delete File ---

@app.delete('/file/delete/{filename}', response_model=FileDeleteResponse)
def delete_file(filename: str):
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
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# =============================================================================
# Document Management Endpoints 
# =============================================================================

# --- Get Document ---

@app.get('/doc/{doc_id}', status_code=status.HTTP_200_OK, response_model=DocumentDetail)
def get_doc(doc_id: str):
    """
    Retrieve a single document by ID
    
    Fetches the content and metadata for a specific document chunk using its
    unique identifier.
    
    Args:
        doc_id (str): Unique document identifier
        
    Returns:
        DocumentDetail: Document content, metadata, and ID
        
    Raises:
        HTTPException: 404 if document not found
        HTTPException: 500 if retrieval fails
    """
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
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

# --- Add Document ---

@app.post('/doc/add', status_code=status.HTTP_201_CREATED, response_model=DocumentResponse)
def add_doc(doc: DocumentInput):
    """
    Add a single pre-extracted document
    
    Indexes a document that has already been extracted and chunked externally.
    Useful for adding custom content without file upload.
    
    Args:
        doc (DocumentInput): Document with content and optional metadata
        
    Returns:
        DocumentResponse: Created document ID and status
        
    Raises:
        HTTPException: 500 if indexing fails
    """
    try:
        doc_id = engine.add_doc(
            content=doc.content,
            metadata=doc.metadata
        )

        return DocumentResponse(
            id=doc_id,
            status="indexed"
        )

    except ValueError as e:
        print(f"ValueError: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Malformed data: {e}"
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

# --- Delete Document ---

@app.delete('/doc/{doc_id}', response_model=DocumentResponse)
def delete_doc(doc_id: str):
    """
    Delete a single document by ID
    
    Removes a specific document chunk and its embedding from the index.
    
    Args:
        doc_id (str): Unique document identifier to delete
        
    Returns:
        DocumentResponse: Deletion confirmation with document ID
        
    Raises:
        HTTPException: 404 if document not found
        HTTPException: 500 if deletion fails
    """
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
    
    except HTTPException:
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
    """
    Perform semantic search on indexed documents
    
    Searches the vector index using semantic similarity to find the most relevant
    document chunks for the given query. Returns results ranked by similarity score.
    
    Args:
        search_params (SearchQuery): Query string and number of results (top_k)
        
    Returns:
        SearchResponse: Ranked search results with similarity scores and metadata
        
    Raises:
        HTTPException: 500 if search fails
        
    Example:
        ```json
        {
            "query": "What is machine learning?",
            "top_k": 5
        }
        ```
    """
    try:
        query = search_params.query.strip()
        results: list[SearchResult] = engine.search(
            query=query, 
            top_k=search_params.top_k
        )

        return SearchResponse(
            query=query,
            results=results,
            count=len(results)
        )
    
    except ValueError as e:
        print(f"ValueError: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Malformed data: {e}"
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

# --- Get Index Stats ---

@app.get('/index/stats', response_model=IndexStatsResponse)
def get_index_stats():
    """
    Get quick index statistics
    
    Lightweight endpoint for checking index health and size. Returns essential
    metrics including document counts, embedding dimension, and compaction status.
    For comprehensive metrics, use GET /metrics instead.
    
    Returns:
        IndexStatsResponse: Core index statistics and health indicators
        
    Raises:
        HTTPException: 500 if stats retrieval fails
    """
    try:
        stats = engine.get_index_stats()

        return IndexStatsResponse(
            status="success",
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

# --- Build Index ---

@app.post('/index/build', response_model=IndexStatsResponse)
def build_index():
    """
    Build or rebuild the vector index
    
    Reconstructs the entire vector index from scratch. Useful for optimizing
    search performance after many document additions/deletions. This operation
    can be time-consuming for large indexes.
    
    Returns:
        IndexStatsResponse: Updated index statistics after rebuild
        
    Raises:
        HTTPException: 500 if index build fails
    """
    try:
        engine.build()
        stats = engine.get_index_stats()

        return IndexStatsResponse(
            status="success",
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

# --- Save Index ---

@app.post('/index/save', response_model=IndexSaveResponse)
def save_index(directory: str = "./data"):
    """
    Persist index to disk
    
    Saves the current vector index and all document metadata to the specified
    directory. Creates persistent storage for index recovery and reduces startup time.
    
    Args:
        directory (str, optional): Directory path for saving. Defaults to "./data"
        
    Returns:
        IndexSaveResponse: Save confirmation with file sizes and document counts
        
    Raises:
        HTTPException: 500 if save operation fails
        
    Example:
        POST /index/save?directory=/path/to/storage
    """
    try:
        save_metrics = engine.save(directory=directory)

        return IndexSaveResponse(
            status=save_metrics["status"],
            directory=save_metrics["directory"],
            metadata_size_mb=save_metrics["metadata_size_mb"],
            embeddings_size_mb=save_metrics["embeddings_size_mb"],
            total_size_mb=save_metrics["total_size_mb"],
            documents_saved=save_metrics["documents_saved"],
            embeddings_saved=save_metrics["embeddings_saved"],
            version=save_metrics["version"]
        )

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

# --- Load Index ---

@app.post('/index/load', response_model=IndexLoadResponse)
def load_index():
    """
    Load index from disk
    
    Restores a previously saved vector index from disk, including all embeddings,
    documents, and metadata. Faster than rebuilding the index from scratch.
    
    Returns:
        IndexLoadResponse: Load confirmation with counts and version information
        
    Raises:
        HTTPException: 404 if index files not found
        HTTPException: 500 if load operation fails
    """
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

# --- Health Check ---

@app.get('/health')
def check_health():
    """
    API health check
    
    Simple endpoint to verify the API is running and responsive. Returns a
    success status for monitoring and load balancer health checks.
    
    Returns:
        dict: Health status object
        
    Example Response:
        ```json
        {
            "status": "healthy",
            "version": "0.9.0"
        }
        ```
    """
    return {
        "status": "healthy",
        "version": __version__
    }

# --- Metrics ---

@app.get('/metrics', response_model=MetricsResponse)
def get_metrics():
    """
    Get comprehensive system metrics
    
    Returns detailed performance, usage, and system statistics including:
    - Index statistics (documents, embeddings, compaction status)
    - Performance metrics (query times, percentiles)
    - Usage statistics (operations performed)
    - Memory consumption
    - System information and uptime
    
    This endpoint provides complete observability into the vector engine's state
    and performance characteristics.
    
    Returns:
        MetricsResponse: Comprehensive metrics across all categories
        
    Raises:
        HTTPException: 500 if metrics collection fails
    """
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
        uptime_seconds=metrics["uptime_seconds"],
        version=metrics["version"]
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
