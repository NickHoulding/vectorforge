from collections import deque
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any
import numpy as np
import uuid
import time

from models import SearchResult


@dataclass
class EngineMetrics:
    """
    Internal metrics tracking for VectorEngine.

    Tracks query performance, usage statistics, and system events.
    Maps to MetricsResponse for API output.
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

    # query_performance history
    query_times: deque[float] = field(default_factory=deque)
    max_query_history: int = 1000

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["query_times"] = list(data["query_times"])
        
        return data

class VectorEngine:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.embeddings: list[np.ndarray] = []
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)
        self.index_to_doc_id = []
        self.doc_id_to_index = {}
        self.deleted_docs = set()
        self.metrics = EngineMetrics()
        self.compaction_threshold = 0.25

        if self.should_compact():
            self.compact()

    def should_compact(self) -> bool:
        """Decide whether compaction is needed based on ratio of deleted docs"""
        if not self.embeddings:
            return False

        deleted_ratio = len(self.deleted_docs) / len(self.embeddings)
        return deleted_ratio > self.compaction_threshold

    def compact(self) -> None:
        """Rebuild index and free deleted doc memory"""
        new_embeddings = []
        new_index_to_doc_id = []
        new_doc_id_to_index = {}

        for old_pos, doc_id in enumerate(self.index_to_doc_id):
            if doc_id not in self.deleted_docs:
                new_pos = len(new_embeddings)
                new_embeddings.append(self.embeddings[old_pos])
                new_index_to_doc_id.append(doc_id)
                new_doc_id_to_index[doc_id] = new_pos

        self.embeddings = new_embeddings
        self.index_to_doc_id = new_index_to_doc_id
        self.doc_id_to_index = new_doc_id_to_index

        for doc_id in self.deleted_docs:
            if doc_id in self.documents:
                del self.documents[doc_id]
        
        self.deleted_docs.clear()
        
        # Update metrics
        self.metrics.compactions_performed += 1
        self.metrics.last_compaction_at = datetime.now().isoformat()

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Calculates the cosine similarity between pre-normalized embeddings"""
        return np.dot(emb_a, emb_b)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search the vector index based on the query"""
        start_time = time.perf_counter()
        
        if not self.embeddings:
            return []

        query_embedding: np.ndarray = self.model.encode(
            sentences=query, 
            convert_to_numpy=True
        )
        normalized_query_embedding: np.ndarray = query_embedding / np.linalg.norm(query_embedding)
        results = []

        for pos, embedding in enumerate(self.embeddings):
            doc_id = self.index_to_doc_id[pos]

            if doc_id in self.deleted_docs:
                continue

            score: float = self.cosine_similarity(
                emb_a=normalized_query_embedding, 
                emb_b=embedding
            )
            results.append((pos, score))

        results.sort(
            key=lambda result: result[1], 
            reverse=True
        )

        search_results = []
        for pos, score in results[:top_k]:
            doc_id = self.index_to_doc_id[pos]
            doc = self.documents[doc_id]

            search_results.append(SearchResult(
                id=doc_id,
                content=doc["content"],
                metadata=doc["metadata"],
                score=score
            ))

        # Track query performance
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.metrics.total_queries += 1
        self.metrics.total_query_time_ms += elapsed_ms
        self.metrics.last_query_at = datetime.now().isoformat()
        
        # Keep rolling window of query times
        self.metrics.query_times.append(elapsed_ms)
        if len(self.metrics.query_times) > self.metrics.max_query_history:
            self.metrics.query_times.popleft()

        return search_results
    
    def list_files(self) -> list[str]:
        """List all files"""
        filenames = set()
        active_docs = [
            (doc_id, doc) for doc_id, doc in self.documents.items() 
            if doc_id not in self.deleted_docs
        ]

        for doc_id, doc in active_docs:
            filename = doc.get("metadata", {}).get("source_file", "")
            filenames.add(filename)

        filenames = list(filenames)
        filenames.sort()

        return filenames
    
    def get_doc(self, doc_id: str) -> dict | None:
        """Retreive a doc with the specified doc id"""
        if doc_id in self.deleted_docs:
            return None
        
        return self.documents.get(doc_id, None)

    def add_doc(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Adds a new doc with the specified content and metadata"""
        doc_id: str = str(uuid.uuid4())

        self.documents[doc_id] = {
            "content": content, 
            "metadata": metadata or {}
        }

        embedding: np.ndarray = self.model.encode(content, convert_to_numpy=True)
        normalized_embedding: np.ndarray = embedding / np.linalg.norm(embedding)
        vector_index: int = len(self.embeddings)

        self.embeddings.append(normalized_embedding)
        self.index_to_doc_id.append(doc_id)
        self.doc_id_to_index[doc_id] = vector_index

        # Update metrics
        self.metrics.docs_added += 1
        self.metrics.total_doc_size_bytes += len(content)
        self.metrics.last_doc_added_at = datetime.now().isoformat()
        
        # Track file upload if source_file in metadata
        if metadata and metadata.get("source_file"):
            if metadata.get("chunk_index") == 0:
                self.metrics.files_uploaded += 1
                self.metrics.last_file_uploaded_at = datetime.now().isoformat()
            self.metrics.chunks_created += 1

        return doc_id

    def remove_doc(self, doc_id: str) -> bool:
        """Removes the doc with the specified doc_id (lazy deletion)"""
        if doc_id not in self.documents:
            return False

        self.metrics.total_doc_size_bytes -= len(self.documents[doc_id]["content"])
        self.deleted_docs.add(doc_id)
        
        # Update metrics
        self.metrics.docs_deleted += 1

        if self.should_compact():
            self.compact()

        return True

    def get_metrics(self) -> dict:
        """Returns all vector engine metrics with calculated statistics"""
        # Start with base metrics
        metrics_dict = self.metrics.to_dict()
        
        # Calculate active documents
        total_docs = len(self.documents)
        deleted_docs = len(self.deleted_docs)
        active_docs = total_docs - deleted_docs
        total_embeddings = len(self.embeddings)
        deleted_ratio = (
            deleted_docs / total_embeddings 
            if total_embeddings > 0 else 0.0
        )
        
        # Calculate performance stats
        avg_query_time = (
            self.metrics.total_query_time_ms / self.metrics.total_queries 
            if self.metrics.total_queries > 0 else 0.0
        )
        
        # Calculate percentiles from query history
        sorted_times = sorted(self.metrics.query_times) if self.metrics.query_times else []
        p50 = float(np.percentile(sorted_times, 50)) if sorted_times else None
        p95 = float(np.percentile(sorted_times, 95)) if sorted_times else None
        p99 = float(np.percentile(sorted_times, 99)) if sorted_times else None
        min_time = min(sorted_times) if sorted_times else None
        max_time = max(sorted_times) if sorted_times else None
        
        # Calculate memory estimates (4 bytes per float32)
        embedding_dim = self.model.get_sentence_embedding_dimension() or 0
        embeddings_mb = (total_embeddings * embedding_dim * 4) / (1024 * 1024)
        documents_mb = self.metrics.total_doc_size_bytes / (1024 * 1024)
        
        # Calculate uptime
        created = datetime.fromisoformat(self.metrics.created_at)
        uptime = (datetime.now() - created).total_seconds()
        
        # Add calculated metrics to dictionary
        metrics_dict.update({
            # Index metrics
            "active_documents": active_docs,
            "total_embeddings": total_embeddings,
            "deleted_ratio": deleted_ratio,
            "needs_compaction": self.should_compact(),
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
            "uptime_seconds": uptime
        })
        
        return metrics_dict
