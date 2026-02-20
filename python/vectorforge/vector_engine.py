import os
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from vectorforge import __version__
from vectorforge.config import VFGConfig
from vectorforge.models import SearchResult


@dataclass
class EngineMetrics:
    """Internal metrics tracking for VectorEngine.

    Tracks query performance, usage statistics, and system events for monitoring
    and analysis. This class maintains counters, performance metrics, storage
    statistics, and timestamps for various engine operations.

    Attributes:
        total_queries: Total number of search queries executed.
        docs_added: Total number of documents added to the index.
        docs_deleted: Total number of documents marked for deletion.
        chunks_created: Total number of document chunks created from files.
        files_uploaded: Total number of files uploaded and processed.
        total_query_time_ms: Cumulative time spent on all queries in milliseconds.
        total_doc_size_bytes: Total size of all document content in bytes.
        created_at: ISO timestamp when the engine was initialized.
        last_query_at: ISO timestamp of the most recent query, or None.
        last_doc_added_at: ISO timestamp of the most recent document addition, or None.
        last_file_uploaded_at: ISO timestamp of the most recent file upload, or None.
        query_times: Rolling window of recent query execution times in milliseconds.
        max_query_history: Maximum number of query times to retain in the rolling window.
    """

    # Counters
    total_queries: int = 0
    docs_added: int = 0
    docs_deleted: int = 0
    chunks_created: int = 0
    files_uploaded: int = 0

    # Performance tracking
    total_query_time_ms: float = 0.0

    # Storage tracking
    total_doc_size_bytes: int = 0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_query_at: Optional[str] = None
    last_doc_added_at: Optional[str] = None
    last_file_uploaded_at: Optional[str] = None

    # Query performance history
    query_times: deque[float] = field(default_factory=deque)
    max_query_history: int = VFGConfig.MAX_QUERY_HISTORY

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary for serialization.

        Transforms the dataclass into a JSON-serializable dictionary, converting
        the deque of query_times into a list for proper serialization.

        Returns:
            Dictionary representation of all metrics with query_times converted
            to a list.
        """
        data: dict[str, Any] = asdict(self)
        data["query_times"] = list(data["query_times"])

        return data


class VectorEngine:
    """High-performance vector database for semantic search using ChromaDB.

    VectorEngine provides document storage, embedding generation, and similarity
    search capabilities using sentence transformers and ChromaDB for persistence.
    ChromaDB handles index management, compaction, and persistence automatically.

    The engine uses cosine similarity on normalized embeddings for efficient
    semantic search across documents.

    Attributes:
        chroma_client: ChromaDB PersistentClient for database operations.
        collection: ChromaDB collection storing documents and embeddings.
        model_name: Name of the sentence transformer model being used.
        model: Loaded SentenceTransformer model instance.
        metrics: EngineMetrics instance tracking usage and performance.

    Example:
        >>> engine = VectorEngine()
        >>> doc_id = engine.add_doc("Hello world", {"source_file": "test.txt"})
        >>> results = engine.search("greeting", top_k=5)
        >>> engine.save()
    """

    def __init__(self) -> None:
        """Initialize the VectorEngine with ChromaDB backend.

        Creates a vector database using ChromaDB for storage and retrieval,
        with the 'all-MiniLM-L6-v2' sentence transformer model for embedding
        generation. Initializes metrics tracking.
        """
        engine_dir = os.path.dirname(os.path.abspath(__file__))
        chroma_path = os.path.join(engine_dir, VFGConfig.CHROMA_PERSIST_DIR)

        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=VFGConfig.CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.model_name: str = VFGConfig.MODEL_NAME
        self.model: SentenceTransformer = SentenceTransformer(self.model_name)
        self.metrics: EngineMetrics = EngineMetrics()

    def save(self, directory: str = VFGConfig.DEFAULT_DATA_DIR) -> dict[str, Any]:
        """Get information about ChromaDB's persistent storage.

        ChromaDB PersistentClient automatically persists all data. This endpoint
        provides information about the persisted data location and current stats.

        For backup: Copy the ChromaDB data directory when the application is stopped.

        Args:
            directory: Parameter maintained for API compatibility (not used).

        Returns:
            A dictionary containing storage information and statistics:
                - status: 'saved' (data is auto-persisted)
                - directory: Path where ChromaDB persists data
                - documents_saved: Current number of documents
                - total_size_mb: Estimated size of persisted data
                - version: VectorForge version
                - note: Instructions for manual backup
        """
        doc_count = self.collection.count()
        persist_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), VFGConfig.CHROMA_PERSIST_DIR
        )

        total_size_mb = 0.0
        if os.path.exists(persist_dir):
            total_size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(persist_dir)
                for filename in filenames
            )
            total_size_mb = total_size_bytes / (1024 * 1024)
        else:
            total_size_mb = (doc_count * 384 * 4 + doc_count * 1024) / (1024 * 1024)

        return {
            "status": "saved",
            "directory": persist_dir,
            "documents_saved": doc_count,
            "embeddings_saved": doc_count,
            "metadata_size_mb": total_size_mb * 0.3,  # Rough estimate
            "embeddings_size_mb": total_size_mb * 0.7,  # Rough estimate
            "total_size_mb": total_size_mb,
            "version": __version__,
            "note": "ChromaDB auto-persists data. For backups, copy the data directory when the application is stopped.",
        }

    def load(self, directory: str = VFGConfig.DEFAULT_DATA_DIR) -> dict[str, Any]:
        """Get information about ChromaDB's current loaded data.

        ChromaDB PersistentClient automatically loads data on initialization.
        This endpoint returns information about the currently loaded data.

        For restore: Replace the ChromaDB data directory when the application is stopped,
        then restart the application.

        Args:
            directory: Parameter maintained for API compatibility (not used).

        Returns:
            A dictionary containing current data information:
                - status: 'loaded'
                - directory: Path where ChromaDB loads/persists data
                - documents_loaded: Current number of documents
                - version: VectorForge version
                - note: Instructions for manual restore
        """
        persist_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), VFGConfig.CHROMA_PERSIST_DIR
        )

        doc_count = self.collection.count()

        return {
            "status": "loaded",
            "directory": persist_dir,
            "documents_loaded": doc_count,
            "embeddings_loaded": doc_count,
            "version": __version__,
            "note": "ChromaDB auto-loads on initialization. For restore, replace the data directory when stopped and restart.",
        }

    def search(
        self,
        query: str,
        top_k: int = VFGConfig.DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the vector index for documents similar to the query.

        Encodes the query text, uses ChromaDB to find similar documents,
        and returns the top-k most similar results. Tracks query performance
        metrics.

        Args:
            query: Text query to search for.
            top_k: Maximum number of results to return. Defaults to 10.
            filters: Optional metadata filters as key-value pairs. All filters
                    must match (AND logic). Matching is case-sensitive and
                    uses exact equality. Example: {"source_file": "doc.pdf"}

        Returns:
            List of SearchResult objects sorted by similarity score in
            descending order. Returns empty list if index is empty or no
            documents match the filters.
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        start_time: float = time.perf_counter()
        self.metrics.total_queries += 1

        if self.collection.count() == 0:
            elapsed_ms: float = (time.perf_counter() - start_time) * 1000
            self._update_query_metrics(elapsed_ms)
            return []

        query_embedding: np.ndarray = self.model.encode(
            sentences=query, convert_to_numpy=True
        )
        normalized_query_embedding: np.ndarray = query_embedding / np.linalg.norm(
            query_embedding
        )

        where_clause: Optional[Dict[str, Any]] = None
        if filters:
            if len(filters) == 1:
                where_clause = dict(filters)
            else:
                where_clause = {"$and": [dict({k: v}) for k, v in filters.items()]}

        results = self.collection.query(
            query_embeddings=[normalized_query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        if (
            results["ids"]
            and results["documents"]
            and results["metadatas"]
            and results["distances"]
            and len(results["ids"][0]) > 0
        ):
            ids_list = results["ids"][0]
            docs_list = results["documents"][0]
            meta_list = results["metadatas"][0]
            dist_list = results["distances"][0]

            for i in range(len(ids_list)):
                doc_id = ids_list[i]
                content = docs_list[i]
                metadata = meta_list[i] or {}
                distance = dist_list[i]
                score = 1.0 - distance

                search_results.append(
                    SearchResult(
                        id=doc_id,
                        content=content,
                        metadata=dict(metadata) if metadata else {},
                        score=score,
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_query_metrics(elapsed_ms)

        return search_results

    def list_files(self) -> list[str]:
        """List all unique source files referenced in documents.

        Extracts the 'source_file' field from document metadata and returns
        a sorted list of unique filenames.

        Returns:
            Sorted list of unique source filenames. Documents lacking source_file
            metadata are skipped. Returns an empty list if an exception is thrown.
        """
        try:
            results = self.collection.get(include=["metadatas"])

            filenames: set[str] = set()
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata and "source_file" in metadata:
                        source_file = metadata["source_file"]
                        if isinstance(source_file, str):
                            filenames.add(source_file)

            unique_filenames = list(filenames)
            unique_filenames.sort()

            return unique_filenames

        except Exception:
            return []

    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a document by its ID.

        Args:
            doc_id: Unique identifier of the document to retrieve.

        Returns:
            Dictionary containing 'content' and 'metadata' keys if found,
            None upon error, or if document doesn't exist.
        """
        try:
            result = self.collection.get(
                ids=[doc_id], include=["documents", "metadatas"]
            )

            if (
                result["ids"]
                and result["documents"]
                and result["metadatas"]
                and len(result["ids"]) > 0
            ):
                docs = result["documents"]
                metas = result["metadatas"]
                return {
                    "content": docs[0],
                    "metadata": dict(metas[0]) if metas[0] else {},
                }
            return None

        except Exception:
            return None

    def add_doc(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a new document to the vector index.

        Generates a unique ID, encodes the content, and adds the document to
        the ChromaDB collection. Updates metrics for document additions, file
        uploads, and chunk creation based on metadata.

        Args:
            content: Text content of the document to index.
            metadata: Optional metadata dictionary. Special handling for:
                - source_file: Tracks file uploads when present
                - chunk_index: When 0, increments files_uploaded counter

        Returns:
            Unique document ID (UUID v4) for the newly added document.
        """
        if not content.strip():
            raise ValueError("Document content cannot be empty")
        if metadata is None:
            metadata = {}

        has_source: bool = "source_file" in metadata
        has_chunk_index: bool = "chunk_index" in metadata

        if has_source != has_chunk_index:
            raise ValueError(
                "Metadata must contain both 'source_file' and 'chunk_index' or neither"
            )

        doc_id: str = str(uuid.uuid4())
        embedding: np.ndarray = self.model.encode(content, convert_to_numpy=True)
        normalized_embedding: np.ndarray = embedding / np.linalg.norm(embedding)
        metadata_to_store = metadata if metadata else None

        self.collection.add(
            ids=[doc_id],
            embeddings=[normalized_embedding.tolist()],
            documents=[content],
            metadatas=[metadata_to_store] if metadata_to_store else None,
        )

        self.metrics.docs_added += 1
        self.metrics.total_doc_size_bytes += len(content)
        self.metrics.last_doc_added_at = datetime.now().isoformat()

        if metadata and metadata.get("source_file"):
            if metadata.get("chunk_index") == 0:
                self.metrics.files_uploaded += 1
                self.metrics.last_file_uploaded_at = datetime.now().isoformat()

            self.metrics.chunks_created += 1

        return doc_id

    def delete_doc(self, doc_id: str) -> bool:
        """Remove a document from the vector index.

        ChromaDB performs immediate deletion (not lazy). Updates deletion metrics.

        Args:
            doc_id: Unique identifier of the document to remove.

        Returns:
            True if document was found and deleted, False upon error or if
            document ID doesn't exist.
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["documents"])
            if not result["ids"] or not result["documents"] or len(result["ids"]) == 0:
                return False

            docs = result["documents"]
            doc_content = docs[0]
            self.metrics.total_doc_size_bytes -= len(doc_content)
            self.collection.delete(ids=[doc_id])
            self.metrics.docs_deleted += 1

            return True

        except Exception:
            return False

    def delete_file(self, filename: str) -> dict[str, Any]:
        """Delete all document chunks associated with a specific source file.

        Finds all documents where metadata['source_file'] matches the given
        filename and deletes them immediately.

        Args:
            filename: Name of the source file whose chunks should be deleted.

        Returns:
            Dictionary containing:
                - status: 'deleted' if chunks found, 'not_found' if no matches
                - filename: The filename that was searched for
                - chunks_deleted: Number of chunks deleted
                - doc_ids: List of deleted document IDs
        """
        try:
            results = self.collection.get(
                where={"source_file": filename}, include=["documents"]
            )

            doc_ids = results["ids"] if results["ids"] else []

            if not doc_ids:
                return {
                    "status": "not_found",
                    "filename": filename,
                    "chunks_deleted": 0,
                    "doc_ids": [],
                }

            deleted_ids = []
            for doc_id in doc_ids:
                if self.delete_doc(doc_id):
                    deleted_ids.append(doc_id)

            return {
                "status": "deleted" if deleted_ids else "not_found",
                "filename": filename,
                "chunks_deleted": len(deleted_ids),
                "doc_ids": deleted_ids,
            }
        except Exception:
            return {
                "status": "not_found",
                "filename": filename,
                "chunks_deleted": 0,
                "doc_ids": [],
            }

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics about the vector engine's state and performance.

        Returns:
            Dictionary containing:
                - Counters: total_queries, docs_added, docs_deleted, etc.
                - Performance: avg/min/max/p50/p95/p99 query times
                - Index stats: active_documents, total_embeddings
                - Memory usage: embeddings_mb, documents_mb, total_mb
                - System info: model_name, model_dimension, uptime_seconds
                - Timestamps: created_at, last_query_at, last_doc_added_at, etc.
        """
        metrics_dict: dict[str, Any] = self.metrics.to_dict()

        total_docs: int = self.collection.count()
        active_docs: int = total_docs
        total_embeddings: int = total_docs

        avg_query_time: float = (
            self.metrics.total_query_time_ms / self.metrics.total_queries
            if self.metrics.total_queries > 0
            else 0.0
        )

        sorted_times: list[float] = (
            sorted(self.metrics.query_times) if self.metrics.query_times else []
        )
        p50: float | None = (
            float(np.percentile(sorted_times, 50)) if sorted_times else None
        )
        p95: float | None = (
            float(np.percentile(sorted_times, 95)) if sorted_times else None
        )
        p99: float | None = (
            float(np.percentile(sorted_times, 99)) if sorted_times else None
        )
        min_time: float | None = min(sorted_times) if sorted_times else None
        max_time: float | None = max(sorted_times) if sorted_times else None

        embedding_dim: int = self.model.get_sentence_embedding_dimension() or 0
        embeddings_mb: float = (total_embeddings * embedding_dim * 4) / (1024 * 1024)
        documents_mb: float = self.metrics.total_doc_size_bytes / (1024 * 1024)

        created: datetime = datetime.fromisoformat(self.metrics.created_at)
        uptime: float = (datetime.now() - created).total_seconds()

        metrics_dict.update(
            {
                # Index metrics
                "active_documents": active_docs,
                "total_embeddings": total_embeddings,
                # Performance metrics
                "avg_query_time_ms": avg_query_time,
                "min_query_time_ms": min_time,
                "max_query_time_ms": max_time,
                "p50_query_time_ms": p50,
                "p95_query_time_ms": p95,
                "p99_query_time_ms": p99,
                # Memory metrics
                "embeddings_mb": embeddings_mb,
                "documents_mb": documents_mb,
                "total_mb": embeddings_mb + documents_mb,
                # System info
                "model_name": self.model_name,
                "model_dimension": embedding_dim,
                "uptime_seconds": uptime,
                "version": __version__,
            }
        )

        return metrics_dict

    def get_index_stats(self) -> dict[str, Any]:
        """Get key statistics about the current index state.

        Returns:
            Dictionary containing:
                - total_documents: Total number of documents in the collection
                - total_embeddings: Number of embedding vectors in the index
                - embedding_dimension: Dimensionality of embedding vectors
        """
        total_docs: int = self.collection.count()
        total_embeddings: int = total_docs

        return {
            "total_documents": total_docs,
            "total_embeddings": total_embeddings,
            "embedding_dimension": self.model.get_sentence_embedding_dimension() or 0,
        }

    def _update_query_metrics(self, elapsed_ms: float) -> None:
        """Helper to update query metrics."""
        self.metrics.total_query_time_ms += elapsed_ms
        self.metrics.last_query_at = datetime.now().isoformat()
        self.metrics.query_times.append(elapsed_ms)

        if len(self.metrics.query_times) > self.metrics.max_query_history:
            self.metrics.query_times.popleft()
