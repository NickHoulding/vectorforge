import json
import os
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from vectorforge import __version__
from vectorforge.config import VFGConfig
from vectorforge.models import SearchResult

try:
    from vectorforge.vectorforge_cpp import cosine_similarity_batch

    _CPP_AVAILABLE: bool = True
except ImportError:
    _CPP_AVAILABLE = False


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
        compactions_performed: Number of times the index has been compacted.
        chunks_created: Total number of document chunks created from files.
        files_uploaded: Total number of files uploaded and processed.
        total_query_time_ms: Cumulative time spent on all queries in milliseconds.
        total_doc_size_bytes: Total size of all document content in bytes.
        created_at: ISO timestamp when the engine was initialized.
        last_query_at: ISO timestamp of the most recent query, or None.
        last_doc_added_at: ISO timestamp of the most recent document addition, or None.
        last_compaction_at: ISO timestamp of the most recent compaction, or None.
        last_file_uploaded_at: ISO timestamp of the most recent file upload, or None.
        query_times: Rolling window of recent query execution times in milliseconds.
        max_query_history: Maximum number of query times to retain in the rolling window.
    """

    # Counters
    total_queries: int = 0
    docs_added: int = 0
    docs_deleted: int = 0
    compactions_performed: int = 0
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
    last_compaction_at: Optional[str] = None
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
    """High-performance in-memory vector database for semantic search.

    VectorEngine provides document storage, embedding generation, and similarity
    search capabilities using sentence transformers. It supports lazy deletion,
    automatic compaction, persistence to disk, and comprehensive metrics tracking.

    The engine uses cosine similarity on normalized embeddings for efficient
    semantic search across documents. It maintains separate storage for document
    content and their vector embeddings, with automatic index management.

    Attributes:
        DEFAULT_DATA_DIR: Default directory path for saving/loading engine state.
        documents: Dictionary mapping document IDs to document content and metadata.
        embeddings: List of normalized embedding vectors for all documents.
        index_to_doc_id: List mapping embedding indices to document IDs.
        doc_id_to_index: Dictionary mapping document IDs to embedding indices.
        deleted_docs: Set of document IDs marked for lazy deletion.
        model_name: Name of the sentence transformer model being used.
        model: Loaded SentenceTransformer model instance.
        metrics: EngineMetrics instance tracking usage and performance.
        compaction_threshold: Ratio threshold (0-1) that triggers automatic compaction.

    Example:
        >>> engine = VectorEngine()
        >>> doc_id = engine.add_doc("Hello world", {"source_file": "test.txt"})
        >>> results = engine.search("greeting", top_k=5)
        >>> engine.save()
    """

    def __init__(self) -> None:
        """Initialize the VectorEngine with an empty index and default settings.

        Creates an empty vector database with the 'all-MiniLM-L6-v2' sentence
        transformer model. Initializes metrics tracking and performs compaction
        if needed.
        """
        self.documents: dict[str, dict[str, Any]] = {}
        self.embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.index_to_doc_id: list[str] = []
        self.doc_id_to_index: dict[str, int] = {}
        self.deleted_docs: set[str] = set()

        self.model_name: str = VFGConfig.MODEL_NAME
        self.model: SentenceTransformer = SentenceTransformer(self.model_name)

        self.metrics: EngineMetrics = EngineMetrics()
        self.compaction_threshold: float = VFGConfig.COMPACTION_THRESHOLD

        if self._should_compact():
            self._compact()

    def save(self, directory: str = VFGConfig.DEFAULT_DATA_DIR) -> dict[str, Any]:
        """Save the vector engine state to disk.

        Persists the complete engine state including active documents,
        embeddings, metadata, and metrics to the specified directory.

        Args:
            directory: Path to the directory where data will be saved.
                Defaults to './data'.

        Returns:
            A dictionary containing save operation status and statistics:
                - status: 'saved'
                - directory: Path where data was saved
                - metadata_size_mb: Size of metadata file in MB
                - embeddings_size_mb: Size of embeddings file in MB
                - total_size_mb: Combined size in MB
                - documents_saved: Number of documents persisted
                - embeddings_saved: Number of embeddings persisted
        """
        if len(directory) > VFGConfig.MAX_PATH_LEN:
            raise ValueError(f"Save path length is too long: {directory}")

        os.makedirs(directory, exist_ok=True)

        active_documents: dict[str, dict[str, Any]] = {
            doc_id: doc
            for doc_id, doc in self.documents.items()
            if doc_id not in self.deleted_docs
        }

        active_indices = [
            i
            for i, doc_id in enumerate(self.index_to_doc_id)
            if doc_id not in self.deleted_docs
        ]
        active_embeddings: np.ndarray = self.embeddings[active_indices]
        active_index_to_doc_id: list[str] = []
        active_doc_id_to_index: dict[str, int] = {}

        for new_pos, old_pos in enumerate(active_indices):
            doc_id = self.index_to_doc_id[old_pos]
            active_index_to_doc_id.append(doc_id)
            active_doc_id_to_index[doc_id] = new_pos

        metadata: dict[str, Any] = {
            "documents": active_documents,
            "index_to_doc_id": active_index_to_doc_id,
            "doc_id_to_index": active_doc_id_to_index,
            "deleted_docs": [],
            "model_name": self.model_name,
            "compaction_threshold": self.compaction_threshold,
            "metrics": self.metrics.to_dict(),
            "version": __version__,
        }

        metadata_path: str = os.path.join(directory, VFGConfig.METADATA_FILENAME)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        embeddings_path: str = os.path.join(directory, VFGConfig.EMBEDDINGS_FILENAME)
        np.savez_compressed(embeddings_path, embeddings=active_embeddings)

        metadata_size: int = os.path.getsize(metadata_path)
        metadata_size_mb: float = metadata_size / (1024**2)
        embeddings_size: int = os.path.getsize(embeddings_path)
        embeddings_size_mb: float = embeddings_size / (1024**2)

        return {
            "status": "saved",
            "directory": directory,
            "metadata_size_mb": metadata_size_mb,
            "embeddings_size_mb": embeddings_size_mb,
            "total_size_mb": metadata_size_mb + embeddings_size_mb,
            "documents_saved": len(active_documents),
            "embeddings_saved": len(active_embeddings),
            "version": __version__,
        }

    def load(self, directory: str = VFGConfig.DEFAULT_DATA_DIR) -> dict[str, Any]:
        """Load the vector engine state from disk.

        Restores the complete engine state including documents, embeddings,
        metadata, and metrics from the specified directory.

        Args:
            directory: Path to the directory containing saved data.
                Defaults to './data'.

        Returns:
            A dictionary containing load operation status and statistics:
                - status: 'loaded'
                - directory: Path where data was loaded from
                - documents_loaded: Number of documents restored
                - embeddings_loaded: Number of embeddings restored
                - deleted_docs: Number of deleted documents tracked
                - version: Version of the saved data format

        Raises:
            FileNotFoundError: If metadata.json or embeddings.npz files
                are not found in the specified directory.
        """
        if len(directory) > VFGConfig.MAX_PATH_LEN:
            raise ValueError(f"Load path length is too long: {directory}")
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        metadata_path: str = os.path.join(directory, VFGConfig.METADATA_FILENAME)
        embeddings_path: str = os.path.join(directory, VFGConfig.EMBEDDINGS_FILENAME)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        with open(metadata_path, "r") as f:
            metadata: dict[str, Any] = json.load(f)

        self.documents = metadata["documents"]
        self.index_to_doc_id = metadata["index_to_doc_id"]
        self.doc_id_to_index = metadata["doc_id_to_index"]
        self.deleted_docs = set(metadata["deleted_docs"])
        self.model_name = metadata["model_name"]
        self.compaction_threshold = metadata["compaction_threshold"]

        embeddings_data: dict[str, Any] = np.load(embeddings_path)
        embeddings_array: np.ndarray = embeddings_data["embeddings"]
        self.embeddings = embeddings_array.astype(np.float32)

        saved_metrics: dict[str, Any] = metadata["metrics"]
        self.metrics = EngineMetrics(
            total_queries=saved_metrics["total_queries"],
            docs_added=saved_metrics["docs_added"],
            docs_deleted=saved_metrics["docs_deleted"],
            compactions_performed=saved_metrics["compactions_performed"],
            chunks_created=saved_metrics["chunks_created"],
            files_uploaded=saved_metrics["files_uploaded"],
            total_query_time_ms=saved_metrics["total_query_time_ms"],
            total_doc_size_bytes=saved_metrics["total_doc_size_bytes"],
            created_at=saved_metrics["created_at"],
            last_query_at=saved_metrics["last_query_at"],
            last_doc_added_at=saved_metrics["last_doc_added_at"],
            last_compaction_at=saved_metrics["last_compaction_at"],
            last_file_uploaded_at=saved_metrics["last_file_uploaded_at"],
            query_times=deque(
                saved_metrics["query_times"], maxlen=saved_metrics["max_query_history"]
            ),
            max_query_history=saved_metrics["max_query_history"],
        )

        self.model = SentenceTransformer(self.model_name)

        return {
            "status": "loaded",
            "directory": directory,
            "documents_loaded": len(self.documents),
            "embeddings_loaded": len(self.embeddings),
            "deleted_docs": len(self.deleted_docs),
            "version": metadata.get("version", "unknown"),
        }

    def build(self) -> None:
        """Rebuild the entire index from scratch.

        Re-encodes all active documents (excluding deleted ones) and reconstructs
        the embedding array and index mappings. This is a more aggressive operation
        than compact() as it regenerates all embeddings.
        """
        if len(self.embeddings) == 0:
            return

        self.documents = {
            doc_id: doc
            for doc_id, doc in self.documents.items()
            if doc_id not in self.deleted_docs
        }
        self.embeddings = np.empty((0, 384), dtype=np.float32)
        self.index_to_doc_id = []
        self.doc_id_to_index = {}

        for doc_id, doc in self.documents.items():
            embedding: np.ndarray = self.model.encode(
                sentences=doc["content"], convert_to_numpy=True
            )
            normalized_embedding: np.ndarray = embedding / np.linalg.norm(embedding)

            vector_index: int = len(self.embeddings)
            self.embeddings = np.vstack(
                [
                    self.embeddings,
                    normalized_embedding.astype(np.float32).reshape(1, -1),
                ]
            )
            self.index_to_doc_id.append(doc_id)
            self.doc_id_to_index[doc_id] = vector_index

        self.deleted_docs.clear()

        self.metrics.compactions_performed += 1
        self.metrics.last_compaction_at = datetime.now().isoformat()

    def search(
        self,
        query: str,
        top_k: int = VFGConfig.DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the vector index for documents similar to the query.

        Encodes the query text, computes similarity scores against all active
        documents, and returns the top-k most similar results. Tracks query
        performance metrics.

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

        if len(self.embeddings) == 0:
            elapsed_ms: float = (time.perf_counter() - start_time) * 1000
            self.metrics.total_query_time_ms += elapsed_ms
            self.metrics.last_query_at = datetime.now().isoformat()
            self.metrics.query_times.append(elapsed_ms)

            if len(self.metrics.query_times) > self.metrics.max_query_history:
                self.metrics.query_times.popleft()

            return []

        query_embedding: np.ndarray = self.model.encode(
            sentences=query, convert_to_numpy=True
        )
        normalized_query_embedding: np.ndarray = query_embedding / np.linalg.norm(
            query_embedding
        )

        results: list[tuple[int, float]] = []

        if _CPP_AVAILABLE and len(self.embeddings) > 0:
            query_f32: np.ndarray = normalized_query_embedding.astype(np.float32)
            scores: np.ndarray = cosine_similarity_batch(query_f32, self.embeddings)

            results = [
                (pos, float(scores[pos]))
                for pos in range(len(self.embeddings))
                if self.index_to_doc_id[pos] not in self.deleted_docs
            ]
        else:
            for pos, embedding in enumerate(self.embeddings):
                doc_id: str = self.index_to_doc_id[pos]

                if doc_id in self.deleted_docs:
                    continue

                score: float = self._cosine_similarity(
                    embedding_a=normalized_query_embedding, embedding_b=embedding
                )
                results.append((pos, score))

        results.sort(key=lambda result: result[1], reverse=True)

        search_results: list[SearchResult] = []
        for pos, score in results:
            doc_id = self.index_to_doc_id[pos]
            doc: dict[str, Any] = self.documents[doc_id]

            if filters and not self._matches_filters(doc["metadata"], filters):
                continue

            search_results.append(
                SearchResult(
                    id=doc_id,
                    content=doc["content"],
                    metadata=doc["metadata"],
                    score=score,
                )
            )

            if len(search_results) >= top_k:
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.metrics.total_query_time_ms += elapsed_ms
        self.metrics.last_query_at = datetime.now().isoformat()

        self.metrics.query_times.append(elapsed_ms)
        if len(self.metrics.query_times) > self.metrics.max_query_history:
            self.metrics.query_times.popleft()

        return search_results

    def list_files(self) -> list[str]:
        """List all unique source files referenced in active documents.

        Extracts the 'source_file' field from document metadata and returns
        a sorted list of unique filenames.

        Returns:
            Sorted list of unique source filenames. Documents lacking source_file
            metadata are skipped.
        """
        filenames: set[str] = set()
        active_docs: list[dict[str, Any]] = [
            doc
            for doc_id, doc in self.documents.items()
            if doc_id not in self.deleted_docs
        ]

        for doc in active_docs:
            if "source_file" in doc["metadata"]:
                filenames.add(doc["metadata"]["source_file"])

        unique_filenames: list[str] = list(filenames)
        unique_filenames.sort()

        return unique_filenames

    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a document by its ID.

        Args:
            doc_id: Unique identifier of the document to retrieve.

        Returns:
            Dictionary containing 'content' and 'metadata' keys if found,
            None if document doesn't exist or has been deleted.
        """
        if doc_id in self.deleted_docs:
            return None

        return self.documents.get(doc_id, None)

    def add_doc(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a new document to the vector index.

        Generates a unique ID, encodes the content, and adds the document to
        the index. Updates metrics for document additions, file uploads, and
        chunk creation based on metadata.

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

        self.documents[doc_id] = {"content": content, "metadata": metadata}

        embedding: np.ndarray = self.model.encode(content, convert_to_numpy=True)
        normalized_embedding: np.ndarray = embedding / np.linalg.norm(embedding)
        vector_index: int = len(self.embeddings)

        self.embeddings = np.vstack(
            [self.embeddings, normalized_embedding.astype(np.float32).reshape(1, -1)]
        )
        self.index_to_doc_id.append(doc_id)
        self.doc_id_to_index[doc_id] = vector_index

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
        """Remove a document from the vector index (lazy deletion).

        Marks the document as deleted without immediately freeing memory.
        Triggers compaction if the deletion threshold is exceeded. Updates
        deletion metrics.

        Args:
            doc_id: Unique identifier of the document to remove.

        Returns:
            True if document was found and marked for deletion, False if
            document ID doesn't exist.
        """
        if doc_id not in self.documents:
            return False

        self.metrics.total_doc_size_bytes -= len(self.documents[doc_id]["content"])
        self.deleted_docs.add(doc_id)

        self.metrics.docs_deleted += 1

        if self._should_compact():
            self._compact()

        return True

    def delete_file(self, filename: str) -> dict[str, Any]:
        """Delete all document chunks associated with a specific source file.

        Finds all documents where metadata['source_file'] matches the given
        filename and marks them for deletion. Triggers compaction if needed.

        Args:
            filename: Name of the source file whose chunks should be deleted.

        Returns:
            Dictionary containing:
                - status: 'deleted' if chunks found, 'not_found' if no matches
                - filename: The filename that was searched for
                - chunks_deleted: Number of chunks marked for deletion
                - doc_ids: List of deleted document IDs
        """
        doc_ids: list[str] = []

        matching_doc_ids: list[str] = []
        for doc_id, doc in self.documents.items():
            if doc_id in self.deleted_docs:
                continue

            source: str | None = doc["metadata"].get("source_file", None)

            if source == filename:
                matching_doc_ids.append(doc_id)

        for doc_id in matching_doc_ids:
            if self.delete_doc(doc_id=doc_id):
                doc_ids.append(doc_id)

        return {
            "status": "deleted" if doc_ids else "not_found",
            "filename": filename,
            "chunks_deleted": len(doc_ids),
            "doc_ids": doc_ids,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics about the vector engine's state and performance.

        Returns:
            Dictionary containing:
                - Counters: total_queries, docs_added, docs_deleted, etc.
                - Performance: avg/min/max/p50/p95/p99 query times
                - Index stats: active_documents, total_embeddings, deleted_ratio
                - Memory usage: embeddings_mb, documents_mb, total_mb
                - System info: model_name, model_dimension, uptime_seconds
                - Timestamps: created_at, last_query_at, last_doc_added_at, etc.
        """
        metrics_dict: dict[str, Any] = self.metrics.to_dict()

        total_docs: int = len(self.documents)
        deleted_docs: int = len(self.deleted_docs)
        active_docs: int = total_docs - deleted_docs
        total_embeddings: int = len(self.embeddings)
        deleted_ratio: float = (
            deleted_docs / total_embeddings if total_embeddings > 0 else 0.0
        )

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
                "deleted_ratio": deleted_ratio,
                "needs_compaction": self._should_compact(),
                "compact_threshold": self.compaction_threshold,
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
                - total_documents: Total number of documents in storage
                - total_embeddings: Number of embeddings in the index
                - deleted_documents: Number of documents marked as deleted
                - deleted_ratio: Ratio of deleted docs to total embeddings
                - needs_compaction: Whether compaction is recommended
                - embedding_dimension: Dimensionality of embedding vectors
        """
        total_docs: int = len(self.documents)
        total_embeddings: int = len(self.embeddings)
        deleted_docs: int = len(self.deleted_docs)
        deleted_ratio: float = (
            deleted_docs / total_embeddings if total_embeddings > 0 else 0.0
        )

        return {
            "total_documents": total_docs,
            "total_embeddings": total_embeddings,
            "deleted_documents": deleted_docs,
            "deleted_ratio": deleted_ratio,
            "needs_compaction": self._should_compact(),
            "embedding_dimension": self.model.get_sentence_embedding_dimension() or 0,
        }

    def _should_compact(self) -> bool:
        """Determine whether compaction is needed based on deleted document ratio.

        Compaction is recommended when the ratio of deleted documents to total
        embeddings exceeds the configured threshold (default 0.25).

        Returns:
            True if compaction should be performed, False otherwise.
        """
        if len(self.embeddings) == 0:
            return False

        deleted_ratio: float = len(self.deleted_docs) / len(self.embeddings)

        return deleted_ratio > self.compaction_threshold

    def _compact(self) -> None:
        """Clean up the index by removing deleted documents and rebuilding indices.

        Removes all deleted documents from memory, rebuilds the embedding array
        and index mappings to eliminate fragmentation. Updates compaction metrics
        and timestamps.
        """
        if len(self.deleted_docs) == 0:
            return

        active_indices = [
            i
            for i, doc_id in enumerate(self.index_to_doc_id)
            if doc_id not in self.deleted_docs
        ]

        self.embeddings = self.embeddings[active_indices]
        new_index_to_doc_id: list[str] = []
        new_doc_id_to_index: dict[str, int] = {}

        for new_pos, old_pos in enumerate(active_indices):
            doc_id = self.index_to_doc_id[old_pos]
            new_index_to_doc_id.append(doc_id)
            new_doc_id_to_index[doc_id] = new_pos

        self.index_to_doc_id = new_index_to_doc_id
        self.doc_id_to_index = new_doc_id_to_index

        for doc_id in self.deleted_docs:
            if doc_id in self.documents:
                del self.documents[doc_id]

        self.deleted_docs.clear()

        self.metrics.compactions_performed += 1
        self.metrics.last_compaction_at = datetime.now().isoformat()

    def _cosine_similarity(
        self, embedding_a: np.ndarray, embedding_b: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two pre-normalized embeddings.

        Since embeddings are pre-normalized, this is equivalent to the dot product.

        Args:
            emb_a: First normalized embedding vector.
            emb_b: Second normalized embedding vector.

        Returns:
            Cosine similarity score between -1 and 1, where 1 indicates
            identical vectors and -1 indicates opposite vectors.
        """
        return float(np.dot(embedding_a, embedding_b))

    def _matches_filters(
        self, metadata: dict[str, Any] | None, filters: dict[str, Any]
    ) -> bool:
        """Check if document metadata matches all filter criteria.

        Applies AND logic: all filter key-value pairs must match exactly
        for the document to pass. Matching is case-sensitive and uses
        equality comparison.

        Args:
            metadata: Document metadata dictionary, or None.
            filters: Filter criteria as key-value pairs.

        Returns:
            True if all filters match the metadata, False otherwise.
            Returns False if metadata is None and filters are provided.
        """
        if metadata is None:
            return False

        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False

        return True
