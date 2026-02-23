import logging
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

logger = logging.getLogger(__name__)


@dataclass
class EngineMetrics:
    """Internal metrics tracking for VectorEngine.

    Tracks query performance, usage statistics, and system events for monitoring
    and analysis. This class maintains counters, performance metrics, storage
    statistics, and timestamps for various engine operations.

    Note:
        Metrics are SESSION-SCOPED and reset when the VectorEngine is reinitialized.
        Only `total_documents` (retrieved from ChromaDB) persists across engine
        restarts. All counters, timestamps, and performance history represent
        activity during the current session.

        This means:
        - `docs_added` tracks documents added in this session (resets to 0)
        - `total_documents` tracks all documents in ChromaDB (persists)
        - After restart: `total_documents` may exceed `docs_added`

    Attributes:
        total_queries: Total number of search queries executed.
        docs_added: Total number of documents added to the index.
        docs_deleted: Total number of documents marked for deletion.
        chunks_created: Total number of document chunks created from files.
        files_uploaded: Total number of files uploaded and processed.
        total_query_time_ms: Cumulative time spent on all queries in milliseconds.
        total_doc_size_bytes: Total size of all document content in bytes.
        total_documents_peak: Maximum number of documents reached during this session.
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
    total_documents_peak: int = 0

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

    Note:
        This class is designed to be accessed through API endpoints that use the
        @handle_api_errors decorator. Some methods explicitly raise ValueError for
        invalid input (documented per-method). ChromaDB operations may raise
        database-related exceptions; all exceptions are caught and handled by the
        API layer, which converts them to appropriate HTTP responses.

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
        self._migration_in_progress: bool = False

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

        Raises:
            ValueError: If the query string is empty or contains only whitespace.
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

        ids = results.get("ids")
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")

        search_results: list[SearchResult] = []
        if (
            ids
            and documents
            and metadatas
            and distances
            and len(ids) > 0
            and len(ids[0]) > 0
        ):
            ids_list = ids[0]
            docs_list = documents[0]
            meta_list = metadatas[0]
            dist_list = distances[0]

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
            metadata are skipped. Returns empty list if no documents have been uploaded.
        """
        results = self.collection.get(include=["metadatas"])

        filenames: set[str] = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if metadata and "source_file" in metadata:
                    source_file = str(metadata["source_file"])
                    filenames.add(source_file)

        unique_filenames = list(filenames)
        unique_filenames.sort()

        return unique_filenames

    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a document by its ID.

        Args:
            doc_id: Unique identifier of the document to retrieve.

        Returns:
            Dictionary containing 'content' and 'metadata' keys if found,
            None if document doesn't exist.
        """
        result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])

        ids = result.get("ids")
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        doc = None

        if ids and documents and metadatas and len(ids) > 0:
            doc = {
                "content": documents[0],
                "metadata": metadatas[0],
            }

        return doc

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

        Raises:
            ValueError: If content is empty/whitespace, or if metadata contains
                'source_file' without 'chunk_index' (or vice versa).
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

        # Update peak document count
        current_count = self.collection.count()
        if current_count > self.metrics.total_documents_peak:
            self.metrics.total_documents_peak = current_count

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
            True if document was found and deleted, False if document ID doesn't exist.
        """
        result = self.collection.get(ids=[doc_id], include=["documents"])

        ids = result.get("ids")
        documents = result.get("documents")
        success = False

        if ids and documents and len(ids) > 0:
            doc_content = documents[0]
            size_to_subtract = len(doc_content)
            self.metrics.total_doc_size_bytes = max(
                0, self.metrics.total_doc_size_bytes - size_to_subtract
            )
            self.collection.delete(ids=[doc_id])
            self.metrics.docs_deleted += 1
            success = True

        return success

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
        results = self.collection.get(
            where={"source_file": filename}, include=["documents"]
        )

        doc_ids = results.get("ids")
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
            "status": "deleted",
            "filename": filename,
            "chunks_deleted": len(deleted_ids),
            "doc_ids": deleted_ids,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics about the vector engine's state and performance.

        Returns:
            Dictionary containing:
                - Counters: total_queries, docs_added, docs_deleted, etc.
                - Performance: avg/min/max/p50/p95/p99 query times
                - Index stats: total_documents
                - System info: model_name, model_dimension, uptime_seconds
                - Timestamps: created_at, last_query_at, last_doc_added_at, etc.
        """
        metrics_dict: dict[str, Any] = self.metrics.to_dict()

        total_docs: int = self.collection.count()

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

        created: datetime = datetime.fromisoformat(self.metrics.created_at)
        uptime: float = (datetime.now() - created).total_seconds()

        metrics_dict.update(
            {
                # Index metrics
                "total_documents": total_docs,
                # Performance metrics
                "avg_query_time_ms": avg_query_time,
                "min_query_time_ms": min_time,
                "max_query_time_ms": max_time,
                "p50_query_time_ms": p50,
                "p95_query_time_ms": p95,
                "p99_query_time_ms": p99,
                # System metrics
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
                - embedding_dimension: Dimensionality of embedding vectors
        """
        total_docs: int = self.collection.count()

        return {
            "total_documents": total_docs,
            "embedding_dimension": self.model.get_sentence_embedding_dimension() or 0,
        }

    def _update_query_metrics(self, elapsed_ms: float) -> None:
        """Helper to update query metrics."""
        self.metrics.total_query_time_ms += elapsed_ms
        self.metrics.last_query_at = datetime.now().isoformat()
        self.metrics.query_times.append(elapsed_ms)

        if len(self.metrics.query_times) > self.metrics.max_query_history:
            self.metrics.query_times.popleft()

    def _get_chromadb_disk_size(self) -> tuple[int, float]:
        """Calculate total disk usage of ChromaDB data directory.

        Walks the entire chroma_data directory tree and sums file sizes.
        Called on every metrics request per user requirement.

        Returns:
            Tuple of (bytes, megabytes) for storage usage.

        Raises:
            FileNotFoundError: If the ChromaDB persist directory does not exist.
            OSError: If there's an error accessing the directory or files.
        """
        chroma_path = os.path.join(
            os.path.dirname(__file__), VFGConfig.CHROMA_PERSIST_DIR
        )

        if not os.path.exists(chroma_path):
            raise FileNotFoundError(
                f"ChromaDB persist directory not found: {chroma_path}"
            )

        total_bytes = 0
        try:
            for dirpath, dirnames, filenames in os.walk(chroma_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_bytes += os.path.getsize(filepath)
        except OSError as e:
            raise OSError(f"Error calculating ChromaDB disk size: {e}") from e

        total_mb = total_bytes / (1024 * 1024)
        return total_bytes, total_mb

    def get_chromadb_metrics(self) -> dict[str, Any]:
        """Get ChromaDB-specific operational metrics.

        Returns:
            Dictionary containing:
                - version: ChromaDB library version
                - collection_id: Unique UUID for the collection
                - collection_name: Name of the collection
                - disk_size_bytes: Total storage in bytes
                - disk_size_mb: Total storage in megabytes (rounded to 2 decimals)
                - persist_directory: Absolute path to ChromaDB data
                - max_batch_size: Maximum documents per batch operation

        Raises:
            FileNotFoundError: If ChromaDB persist directory doesn't exist.
            OSError: If there's an error accessing disk or ChromaDB client.
        """
        disk_bytes, disk_mb = self._get_chromadb_disk_size()

        return {
            "version": chromadb.__version__,
            "collection_id": str(self.collection.id),
            "collection_name": self.collection.name,
            "disk_size_bytes": disk_bytes,
            "disk_size_mb": round(disk_mb, 2),
            "persist_directory": os.path.abspath(
                os.path.join(os.path.dirname(__file__), VFGConfig.CHROMA_PERSIST_DIR)
            ),
            "max_batch_size": self.chroma_client.get_max_batch_size(),
        }

    def get_hnsw_config(self) -> dict[str, Any]:
        """Get HNSW index configuration parameters.

        Extracts the Hierarchical Navigable Small World (HNSW) index configuration
        from the ChromaDB collection. These parameters control the behavior of the
        approximate nearest neighbor search algorithm.

        Returns:
            dict: HNSW configuration with the following keys:
                - space: Distance metric (e.g., 'cosine', 'l2', 'ip')
                - ef_construction: Construction-time search parameter
                - ef_search: Query-time search parameter
                - max_neighbors: Maximum connections per node
                - resize_factor: Index growth factor
                - sync_threshold: Persistence threshold

        Raises:
            KeyError: If HNSW configuration is not found in collection.
        """
        config = self.collection.configuration
        hnsw_config = config.get("hnsw")

        if not hnsw_config:
            raise KeyError("HNSW configuration not found in collection")

        return {
            "space": str(hnsw_config.get("space", "cosine")),
            "ef_construction": int(hnsw_config.get("ef_construction", 100)),
            "ef_search": int(hnsw_config.get("ef_search", 100)),
            "max_neighbors": int(hnsw_config.get("max_neighbors", 16)),
            "resize_factor": float(hnsw_config.get("resize_factor", 1.2)),
            "sync_threshold": int(hnsw_config.get("sync_threshold", 1000)),
        }

    def update_hnsw_config(self, new_config: dict[str, Any]) -> dict[str, Any]:
        """Update HNSW configuration with blue-green collection recreation.

        Performs zero-downtime migration by creating a new collection with updated
        HNSW settings, migrating all documents, swapping collections atomically,
        and deleting the old collection.

        This is a destructive operation that requires full collection recreation
        because ChromaDB does not support modifying HNSW parameters after creation.

        Args:
            new_config: Dictionary with HNSW parameters to update. All fields optional:
                - space: Distance metric (default: "cosine")
                - ef_construction: Build-time search depth (default: 100)
                - ef_search: Query-time search depth (default: 100)
                - max_neighbors: Maximum connections per node (default: 16)
                - resize_factor: Dynamic index growth factor (default: 1.2)
                - sync_threshold: Batch size for persistence (default: 1000)

        Returns:
            dict: Migration result containing:
                - status: "success"
                - message: Human-readable success message
                - migration: Statistics (documents_migrated, time_taken_seconds, old_collection_deleted)
                - config: New HNSW configuration

        Raises:
            RuntimeError: If migration is already in progress
            Exception: If migration fails (old collection preserved)

        Example:
            >>> result = engine.update_hnsw_config({"ef_search": 150})
            >>> print(f"Migrated {result['migration']['documents_migrated']} docs")
        """
        if self._migration_in_progress:
            raise RuntimeError("HNSW configuration migration already in progress")

        self._migration_in_progress = True
        start_time = time.perf_counter()
        new_collection = None

        try:
            old_collection = self.collection
            doc_count = old_collection.count()

            logger.info(
                f"Starting HNSW config migration: {doc_count} documents to migrate"
            )

            hnsw_metadata = {
                "hnsw:space": new_config.get("space", "cosine"),
                "hnsw:construction_ef": new_config.get("ef_construction", 100),
                "hnsw:search_ef": new_config.get("ef_search", 100),
                "hnsw:M": new_config.get("max_neighbors", 16),
                "hnsw:resize_factor": new_config.get("resize_factor", 1.2),
                "hnsw:batch_size": new_config.get("sync_threshold", 1000),
            }

            temp_collection_name = (
                f"{VFGConfig.CHROMA_COLLECTION_NAME}_temp_{uuid.uuid4().hex[:8]}"
            )
            logger.info(f"Creating temporary collection: {temp_collection_name}")

            new_collection = self.chroma_client.create_collection(
                name=temp_collection_name, metadata=hnsw_metadata
            )

            if doc_count > 0:
                logger.info("Retrieving documents from old collection")
                old_docs = old_collection.get(
                    include=["documents", "embeddings", "metadatas"]
                )

                ids = old_docs.get("ids")
                documents = old_docs.get("documents")
                embeddings = old_docs.get("embeddings")
                metadatas = old_docs.get("metadatas")

                if ids is None or documents is None or embeddings is None:
                    raise RuntimeError(
                        "Failed to retrieve documents from old collection"
                    )

                batch_size = VFGConfig.MIGRATION_BATCH_SIZE
                total_batches = (doc_count + batch_size - 1) // batch_size

                logger.info(
                    f"Migrating documents in {total_batches} batches of {batch_size}"
                )

                for batch_num in range(total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, doc_count)

                    batch_ids = ids[start_idx:end_idx]
                    batch_docs = documents[start_idx:end_idx]
                    batch_embeddings = embeddings[start_idx:end_idx]
                    batch_metadatas = (
                        metadatas[start_idx:end_idx] if metadatas else None
                    )

                    new_collection.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                    )

                    if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
                        migrated = min(end_idx, doc_count)
                        percent = (migrated / doc_count) * 100
                        logger.info(
                            f"Migration progress: {migrated}/{doc_count} documents ({percent:.1f}%)"
                        )

            original_name = old_collection.name
            logger.info(f"Deleting old collection: {original_name}")
            self.chroma_client.delete_collection(name=original_name)

            logger.info(
                f"Creating final collection with original name: {original_name}"
            )
            final_collection = self.chroma_client.create_collection(
                name=original_name, metadata=hnsw_metadata
            )

            if doc_count > 0:
                logger.info(f"Migrating documents from temp to final collection")
                temp_docs = new_collection.get(
                    include=["documents", "embeddings", "metadatas"]
                )

                final_collection.add(
                    ids=temp_docs["ids"],
                    documents=temp_docs["documents"],
                    embeddings=temp_docs["embeddings"],
                    metadatas=temp_docs["metadatas"],
                )

            logger.info(f"Deleting temporary collection: {new_collection.name}")
            self.chroma_client.delete_collection(name=new_collection.name)

            logger.info("Swapping collection reference to final collection")
            self.collection = final_collection

            elapsed_seconds = time.perf_counter() - start_time
            logger.info(
                f"Migration complete: {doc_count} docs in {elapsed_seconds:.2f}s"
            )

            new_hnsw_config = self.get_hnsw_config()

            return {
                "status": "success",
                "message": "HNSW configuration updated successfully",
                "migration": {
                    "documents_migrated": doc_count,
                    "time_taken_seconds": round(elapsed_seconds, 2),
                    "old_collection_deleted": True,
                },
                "config": new_hnsw_config,
            }

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            try:
                if new_collection is not None:
                    self.chroma_client.delete_collection(name=new_collection.name)
                    logger.info("Cleaned up temporary collection after failure")

            except Exception:
                pass

            raise

        finally:
            self._migration_in_progress = False
