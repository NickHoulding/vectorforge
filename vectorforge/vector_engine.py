"""Core vector engine backed by ChromaDB and sentence-transformers.

Contains VectorEngine and EngineMetrics for document storage, embedding, and semantic search.
"""

import logging
import os
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.types import Metadata, WhereDocument
from sentence_transformers import CrossEncoder, SentenceTransformer

from vectorforge import __version__
from vectorforge.config import VFGConfig
from vectorforge.logging import _sanitize_text_for_logging
from vectorforge.metrics_store import MetricsStore
from vectorforge.models import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class EngineMetrics:
    """Internal metrics tracking for VectorEngine.

    Tracks query performance, usage statistics, and system events for monitoring
    and analysis. This class maintains counters, performance metrics, storage
    statistics, and timestamps for various engine operations.

    Note:
        Counters and timestamps use a **hybrid persistence model**:

        - ``total_queries``, ``docs_added``, ``docs_deleted``, ``chunks_created``,
          ``files_uploaded``, ``total_query_time_ms``, ``total_doc_size_bytes``,
          ``total_documents_peak``, ``lifetime_created_at``, and the ``last_*``
          timestamps are **lifetime values** persisted in SQLite and survive engine
          restarts.
        - ``query_times`` (the rolling deque used for p50/p95/p99 computation) is
          **session-scoped** and resets on restart.

    Attributes:
        total_queries: Lifetime count of search queries executed.
        docs_added: Lifetime count of documents added to the index.
        docs_deleted: Lifetime count of documents deleted from the index.
        chunks_created: Lifetime count of document chunks created from files.
        files_uploaded: Lifetime count of files uploaded and processed.
        total_query_time_ms: Lifetime cumulative time spent on queries in milliseconds.
        total_doc_size_bytes: Net size of all currently stored document content in bytes.
        total_documents_peak: Lifetime maximum document count ever reached.
        lifetime_created_at: ISO timestamp of first engine initialisation for this collection.
        last_query_at: ISO timestamp of the most recent query, or None.
        last_doc_added_at: ISO timestamp of the most recent document addition, or None.
        last_file_uploaded_at: ISO timestamp of the most recent file upload, or None.
        query_times: Session-scoped rolling window of recent query times in milliseconds.
        max_query_history: Maximum number of query times to retain in the rolling window.
    """

    total_queries: int = 0
    docs_added: int = 0
    docs_deleted: int = 0
    chunks_created: int = 0
    files_uploaded: int = 0
    total_query_time_ms: float = 0.0
    total_doc_size_bytes: int = 0
    total_documents_peak: int = 0
    lifetime_created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_query_at: str | None = None
    last_doc_added_at: str | None = None
    last_file_uploaded_at: str | None = None

    # Session-scoped only: resets on restart, not persisted to SQLite.
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
        collection: ChromaDB collection storing documents and embeddings.
        embedding_model: Shared SentenceTransformer model instance.
        reranking_model: Shared CrossEncoder model instance for result reranking.
        chroma_client: ChromaDB client for database operations.
        embedding_model_name: Name of the sentence transformer model being used.
        metrics: EngineMetrics instance tracking usage and performance.

    Example:
        >>> from vectorforge.collection_manager import CollectionManager
        >>> manager = CollectionManager()
        >>> engine = manager.get_engine("my_collection")
        >>> doc_ids = engine.add_docs([{"content": "Hello world", "metadata": {"source_file": "test.txt"}}])
        >>> results = engine.search("greeting", top_k=5)
    """

    __slots__ = (
        "collection",
        "embedding_model_name",
        "embedding_model",
        "reranking_model",
        "chroma_client",
        "metrics",
        "migration_in_progress",
        "chroma_path",
        "_metrics_store",
        "_cached_disk_size",
    )

    def __init__(
        self,
        collection: chromadb.Collection,
        embedding_model: SentenceTransformer,
        reranking_model: CrossEncoder,
        chroma_client: ClientAPI,
    ) -> None:
        """Initialize the VectorEngine for a specific collection.

        Args:
            collection: ChromaDB collection to operate on.
            embedding_model: Shared SentenceTransformer model for generating embeddings.
            reranking_model: CrossEncoder model for re-scoring search results.
            chroma_client: ChromaDB client used for HNSW migration operations.

        Note:
            This constructor is called by CollectionManager, not directly. For
            multi-collection support, use CollectionManager.get_engine() instead.
        """
        self.collection = collection
        self.embedding_model_name: str = VFGConfig.EMBEDDING_MODEL_NAME
        self.embedding_model: SentenceTransformer = embedding_model
        self.reranking_model: CrossEncoder = reranking_model
        self.chroma_client = chroma_client
        self.migration_in_progress: bool = False

        settings = getattr(chroma_client, "_settings", None)
        if settings and hasattr(settings, "persist_directory"):
            self.chroma_path: str = settings.persist_directory
        else:
            self.chroma_path = VFGConfig.CHROMA_PERSIST_DIR

        db_path = os.path.join(self.chroma_path, "metrics.db")
        self._metrics_store: MetricsStore = MetricsStore(db_path)
        self.metrics: EngineMetrics = self._load_metrics(collection.name)

        self._cached_disk_size: dict[str, Any] = {}

    def _load_metrics(self, collection_name: str) -> EngineMetrics:
        """Load lifetime metrics from SQLite or seed a new zero row.

        On first run for a collection, inserts a zero-valued row with the
        current timestamp. On subsequent runs, restores all persisted counters
        and timestamps into a fresh ``EngineMetrics`` instance (with an empty
        session-scoped ``query_times`` deque).

        Args:
            collection_name: Name of the ChromaDB collection.

        Returns:
            ``EngineMetrics`` populated from the persisted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._metrics_store.insert(collection_name, now)
        row = self._metrics_store.load(collection_name)

        if row is None:
            return EngineMetrics(lifetime_created_at=now)

        return EngineMetrics(
            total_queries=row.get("total_queries", 0),
            docs_added=row.get("docs_added", 0),
            docs_deleted=row.get("docs_deleted", 0),
            chunks_created=row.get("chunks_created", 0),
            files_uploaded=row.get("files_uploaded", 0),
            total_query_time_ms=row.get("total_query_time_ms", 0.0),
            total_doc_size_bytes=row.get("total_doc_size_bytes", 0),
            total_documents_peak=row.get("total_documents_peak", 0),
            lifetime_created_at=row.get("created_at", now),
            last_query_at=row.get("last_query_at"),
            last_doc_added_at=row.get("last_doc_added_at"),
            last_file_uploaded_at=row.get("last_file_uploaded_at"),
        )

    def _sigmoid(self, score: float) -> float:
        """Map a raw cross-encoder logit to a [0, 1] similarity score.

        Args:
            score: Raw logit output from the CrossEncoder model.

        Returns:
            Float in (0, 1) representing the normalised relevance score.
        """
        return float(1 / (1 + np.exp(-score)))

    def _rerank(
        self,
        query: str,
        iteration_data: list[SearchResult],
        top_n: int = VFGConfig.DEFAULT_TOP_N,
    ) -> list[SearchResult]:
        """Re-score and re-order results using the cross-encoder reranking model.

        Runs each (query, document) pair through the CrossEncoder, converts the
        raw logit to a sigmoid score, sorts descending by score, and returns the
        top ``top_n`` results.

        Args:
            query: The original search query string.
            iteration_data: Candidate results from the initial vector search.
            top_n: Maximum number of results to return after reranking.

        Returns:
            Reranked list of up to ``top_n`` SearchResult objects, sorted by
            descending cross-encoder score.
        """
        pairs = [(query, doc.content) for doc in iteration_data]
        scores = self.reranking_model.predict(pairs)

        for i in range(len(scores)):
            iteration_data[i].score = self._sigmoid(scores[i])

        return sorted(iteration_data, key=lambda result: result.score, reverse=True)[
            :top_n
        ]

    def search(
        self,
        query: str,
        rerank: bool = VFGConfig.SHOULD_RERANK,
        top_k: int = VFGConfig.DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
        document_filter: WhereDocument | None = None,
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
                    uses exact equality or operator expressions
                    (``$gte``, ``$lte``, ``$ne``, ``$in``).
                    Example: ``{"year": {"$gte": 2020}, "category": "AI"}``
            document_filter: Optional document-text filter. Accepts
                    ``$contains`` or ``$not_contains`` with a string value.
                    Example: ``{"$contains": "machine learning"}``

        Returns:
            List of SearchResult objects sorted by similarity score in
            descending order. Returns empty list if index is empty or no
            documents match the filters.

        Raises:
            ValueError: If the query string is empty or contains only whitespace.
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        logger.debug(
            "search: collection=%s query=%s top_k=%d",
            self.collection.name,
            _sanitize_text_for_logging(query),
            top_k,
        )
        start_time: float = time.perf_counter()

        if self.collection.count() == 0:
            elapsed_ms: float = (time.perf_counter() - start_time) * 1000
            self._update_query_metrics(elapsed_ms)
            return []

        query_embedding: np.ndarray = self.embedding_model.encode(
            sentences=query, convert_to_numpy=True
        )
        normalized_query_embedding: np.ndarray = query_embedding / np.linalg.norm(
            query_embedding
        )

        where_clause: dict[str, Any] | None = None
        if filters:
            if len(filters) == 1:
                where_clause = dict(filters)
            else:
                where_clause = {"$and": [dict({k: v}) for k, v in filters.items()]}

        results = self.collection.query(
            query_embeddings=[normalized_query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
            where_document=document_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids_list = results["ids"]
        documents_list = results["documents"]
        metadatas_list = results["metadatas"]
        distances_list = results["distances"]

        if documents_list is None or metadatas_list is None or distances_list is None:
            return []

        search_results: list[SearchResult] = []
        for ids, documents, metadatas, distances in zip(
            ids_list, documents_list, metadatas_list, distances_list
        ):
            iteration_data = [
                SearchResult(
                    id=doc_id,
                    content=doc,
                    metadata=dict(meta) if meta is not None else None,
                    score=max(0.0, 1.0 - float(distance)),
                )
                for doc_id, doc, meta, distance in zip(
                    ids, documents, metadatas, distances
                )
            ]

            if rerank:
                iteration_data = self._rerank(
                    query, iteration_data, top_n=len(documents)
                )

            for result in iteration_data:
                search_results.append(result)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_query_metrics(elapsed_ms)

        logger.info(
            "search: collection=%s results=%d elapsed_ms=%.1f",
            self.collection.name,
            len(search_results),
            elapsed_ms,
        )
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

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        """Validate that all metadata values are ChromaDB-compatible types.

        Checks each value in the metadata dictionary against the set of types
        ChromaDB accepts as leaf values. ``None`` is explicitly rejected before
        the type check so the error message is unambiguous.

        Args:
            metadata: Metadata dictionary to validate.

        Raises:
            TypeError: If any value is ``None`` or not one of ``str``, ``int``,
                ``float``, or ``bool``.
        """
        for key, value in metadata.items():
            if value is None:
                raise TypeError(
                    f"Metadata value for key: {key} is None (None metadata values not allowed)."
                )
            elif type(value) not in VFGConfig.VALID_SCALAR_TYPES:
                raise TypeError(
                    f"Invalid metadata value: {value} of type: {type(value)}."
                )

    def add_docs(self, docs: list[dict[str, Any]]) -> list[str]:
        """Add multiple documents to the vector index in a single batch operation.

        Validates all documents before any are indexed, then batch-encodes
        embeddings and adds them in a single ChromaDB call. Updates metrics
        once for the entire batch.

        Args:
            docs: List of document dicts, each with a required ``content`` key
                and an optional ``metadata`` key. All documents are validated
                before any are written; a validation error on any single document
                aborts the whole batch.

        Returns:
            List of unique document IDs (UUID v4) in the same order as ``docs``.

        Raises:
            ValueError: If any document's content is empty/whitespace, or if any
                metadata contains 'source_file' without 'chunk_index' (or vice versa).
            TypeError: If any metadata value is ``None`` or not a ChromaDB-supported
                type (``str``, ``int``, ``float``, or ``bool``).
        """
        logger.debug(
            "add_docs: collection=%s count=%d", self.collection.name, len(docs)
        )
        contents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for doc in docs:
            content: str = doc.get("content", "")
            raw_metadata = doc.get("metadata")
            metadata: dict[str, Any] = (
                raw_metadata if isinstance(raw_metadata, dict) else {}
            )

            if not content.strip():
                raise ValueError("Document content cannot be empty")

            self._validate_metadata(metadata)

            has_source: bool = "source_file" in metadata
            has_chunk_index: bool = "chunk_index" in metadata
            if has_source != has_chunk_index:
                raise ValueError(
                    "Metadata must contain both 'source_file' and 'chunk_index' or neither"
                )

            contents.append(content)
            metadatas.append(metadata)

        doc_ids: list[str] = [str(uuid.uuid4()) for _ in contents]
        embeddings: np.ndarray = self.embedding_model.encode(
            contents, convert_to_numpy=True
        )
        norms: np.ndarray = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized: np.ndarray = embeddings / norms

        metadatas_to_store: list[Metadata] | None = (
            [cast(Metadata, m) for m in metadatas] if any(metadatas) else None
        )

        self.collection.add(
            ids=doc_ids,
            embeddings=normalized.tolist(),
            documents=contents,
            metadatas=metadatas_to_store,
        )

        now: str = datetime.now(timezone.utc).isoformat()
        files_uploaded_delta: int = 0
        last_file_uploaded_at: str | None = None

        for content, metadata in zip(contents, metadatas):
            self.metrics.docs_added += 1
            self.metrics.total_doc_size_bytes += len(content)

            if metadata.get("source_file"):
                if metadata.get("chunk_index") == 0:
                    files_uploaded_delta += 1
                    last_file_uploaded_at = now
                self.metrics.chunks_created += 1

        self.metrics.last_doc_added_at = now
        self.metrics.files_uploaded += files_uploaded_delta
        if last_file_uploaded_at:
            self.metrics.last_file_uploaded_at = last_file_uploaded_at

        current_count = self.collection.count()
        if current_count > self.metrics.total_documents_peak:
            self.metrics.total_documents_peak = current_count

        self._metrics_store.save(
            self.collection.name,
            {
                "docs_added": self.metrics.docs_added,
                "total_doc_size_bytes": self.metrics.total_doc_size_bytes,
                "last_doc_added_at": self.metrics.last_doc_added_at,
                "total_documents_peak": self.metrics.total_documents_peak,
                "files_uploaded": self.metrics.files_uploaded,
                "last_file_uploaded_at": self.metrics.last_file_uploaded_at,
                "chunks_created": self.metrics.chunks_created,
            },
        )

        logger.info(
            "add_docs: collection=%s added=%d", self.collection.name, len(doc_ids)
        )
        return doc_ids

    def delete_docs(self, ids: list[str]) -> list[str]:
        """Remove one or more documents from the vector index in a single operation.

        Fetches all matching documents in one ChromaDB call, deletes them together,
        then writes a single metrics update. ChromaDB performs immediate deletion.

        Args:
            ids: One or more document IDs to remove.

        Returns:
            List of IDs that were found and deleted. IDs not present in the
            collection are silently omitted from the result.
        """
        logger.debug(
            "delete_docs: collection=%s ids=%d", self.collection.name, len(ids)
        )
        result = self.collection.get(ids=ids, include=["documents"])

        found_ids: list[str] = result.get("ids") or []
        documents: list[str] = result.get("documents") or []

        if not found_ids:
            return []

        self.collection.delete(ids=found_ids)

        size_to_subtract = sum(len(doc) for doc in documents)
        self.metrics.total_doc_size_bytes = max(
            0, self.metrics.total_doc_size_bytes - size_to_subtract
        )
        self.metrics.docs_deleted += len(found_ids)
        self._metrics_store.save(
            self.collection.name,
            {
                "docs_deleted": self.metrics.docs_deleted,
                "total_doc_size_bytes": self.metrics.total_doc_size_bytes,
            },
        )

        logger.info(
            "delete_docs: collection=%s deleted=%d",
            self.collection.name,
            len(found_ids),
        )
        return found_ids

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
        logger.debug(
            "delete_file: collection=%s filename=%s", self.collection.name, filename
        )
        results = self.collection.get(
            where={"source_file": filename}, include=["documents"]
        )

        doc_ids: list[str] = results.get("ids", [])
        if not doc_ids:
            logger.info(
                "delete_file: collection=%s filename=%s not found",
                self.collection.name,
                filename,
            )
            return {
                "status": "not_found",
                "filename": filename,
                "chunks_deleted": 0,
                "doc_ids": [],
            }

        deleted_ids = self.delete_docs(doc_ids)

        logger.info(
            "delete_file: collection=%s filename=%s chunks_deleted=%d",
            self.collection.name,
            filename,
            len(deleted_ids),
        )
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
                - Timestamps: lifetime_created_at, last_query_at, last_doc_added_at, etc.
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

        embedding_dim: int = (
            self.embedding_model.get_sentence_embedding_dimension() or 0
        )

        created: datetime = datetime.fromisoformat(self.metrics.lifetime_created_at)
        uptime: float = (datetime.now(timezone.utc) - created).total_seconds()

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
                "model_name": self.embedding_model_name,
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
                - total_documents: Total number of documents in the collection.
                - embedding_dimension: Dimensionality of embedding vectors.
        """
        total_docs: int = self.collection.count()

        return {
            "total_documents": total_docs,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension()
            or 0,
        }

    def _update_query_metrics(self, elapsed_ms: float) -> None:
        """Record a completed query's execution time in the rolling metrics window.

        Also increments the lifetime ``total_queries`` counter and flushes
        cumulative counters to SQLite for persistence across restarts.

        Args:
            elapsed_ms: Query duration in milliseconds.
        """
        self.metrics.total_queries += 1
        self.metrics.total_query_time_ms += elapsed_ms
        self.metrics.last_query_at = datetime.now(timezone.utc).isoformat()
        self.metrics.query_times.append(elapsed_ms)

        if len(self.metrics.query_times) > self.metrics.max_query_history:
            self.metrics.query_times.popleft()

        self._metrics_store.save(
            self.collection.name,
            {
                "total_queries": self.metrics.total_queries,
                "total_query_time_ms": self.metrics.total_query_time_ms,
                "last_query_at": self.metrics.last_query_at,
            },
        )

    def _get_chromadb_disk_size(self) -> tuple[int, float]:
        """Calculate total disk usage of the ChromaDB data directory.

        Returns a cached result if it was computed within the last
        ``VFGConfig.DISK_SIZE_TTL_MINS`` minutes; otherwise walks the full
        persist directory tree, sums file sizes, and updates the cache.

        Returns:
            Tuple of (bytes, megabytes) for total storage usage.

        Raises:
            FileNotFoundError: If the ChromaDB persist directory does not exist.
            OSError: If there is an error accessing the directory or its files.
        """
        if self._cached_disk_size:
            current_timestamp = datetime.timestamp(datetime.now(timezone.utc))
            min_diff = (current_timestamp - self._cached_disk_size["timestamp"]) / 60

            if min_diff < VFGConfig.DISK_SIZE_TTL_MINS:
                total_bytes, total_mb = self._cached_disk_size["size"]
                return total_bytes, total_mb

        chroma_path = self.chroma_path

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

        self._cached_disk_size["timestamp"] = datetime.timestamp(
            datetime.now(timezone.utc)
        )
        self._cached_disk_size["size"] = (total_bytes, total_mb)

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
        max_batch_size = self.chroma_client.get_max_batch_size()

        return {
            "version": chromadb.__version__,
            "collection_id": str(self.collection.id),
            "collection_name": self.collection.name,
            "disk_size_bytes": disk_bytes,
            "disk_size_mb": round(disk_mb, 2),
            "persist_directory": self.chroma_path,
            "max_batch_size": max_batch_size,
        }

    def get_hnsw_config(self) -> dict[str, Any]:
        """Get HNSW index configuration parameters.

        Extracts the Hierarchical Navigable Small World (HNSW) index configuration
        from the ChromaDB collection. These parameters control the behavior of the
        approximate nearest neighbor search algorithm.

        Returns:
            Dictionary containing HNSW configuration keys:
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
        """Update HNSW configuration with zero-downtime collection migration.

        Performs a non-destructive blue-green style migration: the original collection
        is preserved until the temporary collection has been fully populated and its
        document count verified. Only after that verification succeeds is the original
        collection deleted, eliminating the data-loss risk on partial failure.

        This operation requires full collection recreation because ChromaDB does not
        support modifying HNSW parameters after creation. All collections share the
        same persistent storage (ChromaDB database), so this is not infrastructure-level
        blue-green deployment with separate volumes.

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
                - migration: Statistics (documents_migrated, time_taken_seconds,
                  old_collection_deleted, temp_verified)
                - config: New HNSW configuration

        Raises:
            RuntimeError: If migration is already in progress, or if the document
                count in the temp or final collection does not match the original.
                The original collection is preserved when the error occurs before
                the old collection is deleted (i.e. during temp population).

        Example:
            >>> result = engine.update_hnsw_config({"ef_search": 150})
            >>> print(f"Migrated {result['migration']['documents_migrated']} docs")
        """
        if self.migration_in_progress:
            raise RuntimeError("HNSW configuration migration already in progress")

        self.migration_in_progress = True
        start_time = time.perf_counter()
        new_collection = None

        try:
            old_collection = self.collection
            doc_count = old_collection.count()

            logger.info(
                "Starting HNSW config migration: %d documents to migrate", doc_count
            )

            hnsw_metadata = {
                "hnsw:space": new_config.get("space", "cosine"),
                "hnsw:construction_ef": new_config.get("ef_construction", 100),
                "hnsw:search_ef": new_config.get("ef_search", 100),
                "hnsw:M": new_config.get("max_neighbors", 16),
                "hnsw:resize_factor": new_config.get("resize_factor", 1.2),
                "hnsw:sync_threshold": new_config.get("sync_threshold", 1000),
            }

            old_metadata = old_collection.metadata or {}
            for key, value in old_metadata.items():
                if (
                    key == "vf:description"
                    or key == "vf:created_at"
                    or key.startswith("vf:meta:")
                ):
                    hnsw_metadata[key] = value

            max_base_len = (
                (VFGConfig.MAX_COLLECTION_NAME_LENGTH - 1) - len("_temp_") - 8
            )
            base_name = old_collection.name[:max_base_len]
            temp_collection_name = f"{base_name}_temp_{uuid.uuid4().hex[:8]}"
            logger.info("Creating temporary collection: %s", temp_collection_name)

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
                    "Migrating documents in %d batches of %d", total_batches, batch_size
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
                            "Migration progress: %d/%d documents (%.1f%%)",
                            migrated,
                            doc_count,
                            percent,
                        )

            temp_count = new_collection.count()
            if temp_count != doc_count:
                raise RuntimeError(
                    f"Temp collection count mismatch: expected {doc_count}, got {temp_count}"
                )
            logger.info(
                "Temp collection verified: %d documents match expected count",
                temp_count,
            )

            original_name = old_collection.name
            logger.info("Deleting old collection: %s", original_name)
            self.chroma_client.delete_collection(name=original_name)

            logger.info(
                "Creating final collection with original name: %s", original_name
            )
            final_collection = self.chroma_client.create_collection(
                name=original_name, metadata=hnsw_metadata
            )

            if doc_count > 0:
                logger.info("Migrating documents from temp to final collection")
                temp_docs = new_collection.get(
                    include=["documents", "embeddings", "metadatas"]
                )

                temp_metadatas = temp_docs.get("metadatas")

                final_collection.add(
                    ids=temp_docs["ids"],
                    documents=temp_docs["documents"],
                    embeddings=temp_docs["embeddings"],
                    metadatas=temp_metadatas if temp_metadatas else None,
                )

            final_count = final_collection.count()
            if final_count != doc_count:
                raise RuntimeError(
                    f"Final collection count mismatch: expected {doc_count}, got {final_count}"
                )
            logger.info(
                "Final collection verified: %d documents match expected count",
                final_count,
            )

            logger.info("Deleting temporary collection: %s", new_collection.name)
            self.chroma_client.delete_collection(name=new_collection.name)

            logger.info("Swapping collection reference to final collection")
            self.collection = final_collection

            elapsed_seconds = time.perf_counter() - start_time
            logger.info(
                "Migration complete: %d docs in %.2fs", doc_count, elapsed_seconds
            )

            new_hnsw_config = self.get_hnsw_config()

            return {
                "status": "success",
                "message": "HNSW configuration updated successfully",
                "migration": {
                    "documents_migrated": doc_count,
                    "time_taken_seconds": round(elapsed_seconds, 2),
                    "old_collection_deleted": True,
                    "temp_verified": True,
                },
                "config": new_hnsw_config,
            }

        except Exception as e:
            logger.error("Migration failed: %s", e, exc_info=True)
            try:
                if new_collection is not None:
                    self.chroma_client.delete_collection(name=new_collection.name)
                    logger.info("Cleaned up temporary collection after failure")

            except Exception:
                pass

            raise

        finally:
            self.migration_in_progress = False
