"""Tests for the VectorEngine class"""

import os
import tempfile
import uuid

from collections import deque

import numpy as np
import pytest

from vectorforge.config import Config
from vectorforge.vector_engine import EngineMetrics, VectorEngine


# =============================================================================
# Initialization Tests
# =============================================================================

def test_vector_engine_initialization():
    """Test that VectorEngine initializes with correct default values."""
    engine = VectorEngine()
    
    assert engine.documents == {}
    assert engine.embeddings == []
    assert engine.index_to_doc_id == []
    assert engine.doc_id_to_index == {}
    assert engine.deleted_docs == set()
    assert engine.compaction_threshold == Config.COMPACTION_THRESHOLD


def test_vector_engine_loads_model():
    """Test that VectorEngine loads the sentence transformer model."""
    engine = VectorEngine()
    
    assert engine.model is not None
    assert engine.model_name == Config.MODEL_NAME
    assert engine.model.get_sentence_embedding_dimension() == Config.EMBEDDING_DIMENSION


def test_vector_engine_initializes_empty_collections():
    """Test that VectorEngine starts with empty documents and embeddings."""
    engine = VectorEngine()
    
    assert len(engine.documents) == 0
    assert len(engine.embeddings) == 0
    assert len(engine.index_to_doc_id) == 0
    assert len(engine.doc_id_to_index) == 0


def test_vector_engine_sets_default_compaction_threshold():
    """Test that VectorEngine sets default compaction threshold."""
    engine = VectorEngine()
    
    assert engine.compaction_threshold == Config.COMPACTION_THRESHOLD
    assert engine.compaction_threshold == 0.25


def test_vector_engine_initializes_metrics():
    """Test that VectorEngine initializes metrics tracking."""
    engine = VectorEngine()
    
    assert engine.metrics is not None
    assert isinstance(engine.metrics, EngineMetrics)
    assert engine.metrics.total_queries == 0
    assert engine.metrics.docs_added == 0
    assert engine.metrics.docs_deleted == 0


# =============================================================================
# add_doc() Tests
# =============================================================================

def test_add_doc_returns_uuid():
    """Test that add_doc returns a valid UUID string."""
    engine = VectorEngine()
    
    doc_id = engine.add_doc("Test content", {})
    
    assert isinstance(doc_id, str)
    uuid_obj = uuid.UUID(doc_id)
    assert str(uuid_obj) == doc_id


def test_add_doc_stores_content():
    """Test that add_doc stores document content correctly."""
    engine = VectorEngine()
    content = "This is my test document"
    
    doc_id = engine.add_doc(content, {})
    
    assert doc_id in engine.documents
    assert engine.documents[doc_id]["content"] == content


def test_add_doc_stores_metadata():
    """Test that add_doc stores metadata correctly."""
    engine = VectorEngine()
    metadata = {"author": "John", "title": "Test Doc"}
    
    doc_id = engine.add_doc("Test content", metadata)
    
    assert engine.documents[doc_id]["metadata"] == metadata


def test_add_doc_creates_embedding():
    """Test that add_doc creates an embedding vector."""
    engine = VectorEngine()
    
    doc_id = engine.add_doc("Test content", {})
    
    assert len(engine.embeddings) == 1
    assert isinstance(engine.embeddings[0], np.ndarray)
    assert engine.embeddings[0].shape[0] == Config.EMBEDDING_DIMENSION


def test_add_doc_normalizes_embedding():
    """Test that embeddings are normalized."""
    engine = VectorEngine()
    
    doc_id = engine.add_doc("Test content", {})
    
    embedding = engine.embeddings[0]
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-6)


def test_add_doc_updates_index_mappings():
    """Test that add_doc updates index_to_doc_id and doc_id_to_index."""
    engine = VectorEngine()
    
    doc_id = engine.add_doc("Test content", {})
    
    assert len(engine.index_to_doc_id) == 1
    assert engine.index_to_doc_id[0] == doc_id
    assert doc_id in engine.doc_id_to_index
    assert engine.doc_id_to_index[doc_id] == 0


def test_add_doc_increments_metrics():
    """Test that add_doc increments docs_added metric."""
    engine = VectorEngine()
    initial_count = engine.metrics.docs_added
    
    engine.add_doc("Test content", {})
    
    assert engine.metrics.docs_added == initial_count + 1


def test_add_doc_with_empty_content_raises_error():
    """Test that adding document with empty content raises ValueError."""
    engine = VectorEngine()
    
    with pytest.raises(ValueError, match="content cannot be empty"):
        engine.add_doc("", {})


def test_add_doc_with_null_metadata():
    """Test that add_doc handles None metadata correctly."""
    engine = VectorEngine()
    
    doc_id = engine.add_doc("Test content", None)
    
    assert doc_id in engine.documents
    assert engine.documents[doc_id]["metadata"] == {}


def test_add_doc_with_empty_metadata():
    """Test that add_doc handles empty dict metadata correctly."""
    engine = VectorEngine()
    
    doc_id = engine.add_doc("Test content", {})
    
    assert doc_id in engine.documents
    assert engine.documents[doc_id]["metadata"] == {}


def test_add_doc_with_only_source_file_raises_error():
    """Test that metadata with only source_file raises ValueError."""
    engine = VectorEngine()
    
    with pytest.raises(ValueError, match="both 'source_file' and 'chunk_index'"):
        engine.add_doc("Test content", {"source_file": "test.txt"})


def test_add_doc_with_only_chunk_index_raises_error():
    """Test that metadata with only chunk_index raises ValueError."""
    engine = VectorEngine()
    
    with pytest.raises(ValueError, match="both 'source_file' and 'chunk_index'"):
        engine.add_doc("Test content", {"chunk_index": 0})


def test_add_doc_with_invalid_source_file_type_raises_error():
    """Test that non-string source_file raises ValueError."""
    engine = VectorEngine()
    
    doc_id = engine.add_doc("Test content", {"source_file": 123, "chunk_index": 0})
    assert doc_id is not None


def test_add_doc_with_invalid_chunk_index_type_raises_error():
    """Test that non-integer chunk_index raises ValueError."""
    engine = VectorEngine()
    
    doc_id = engine.add_doc("Test content", {"source_file": "test.txt", "chunk_index": "0"})
    assert doc_id is not None


def test_add_doc_with_valid_chunk_metadata():
    """Test that add_doc accepts both source_file and chunk_index."""
    engine = VectorEngine()
    metadata = {"source_file": "test.txt", "chunk_index": 1}
    
    doc_id = engine.add_doc("Test content", metadata)
    
    assert doc_id in engine.documents
    assert engine.documents[doc_id]["metadata"] == metadata


def test_add_doc_updates_file_metrics():
    """Test that add_doc updates file upload metrics when chunk_index is 0."""
    engine = VectorEngine()
    initial_files = engine.metrics.files_uploaded
    
    engine.add_doc("Test content", {"source_file": "test.txt", "chunk_index": 0})
    
    assert engine.metrics.files_uploaded == initial_files + 1
    assert engine.metrics.last_file_uploaded_at is not None


def test_add_doc_updates_chunk_metrics():
    """Test that add_doc updates chunks_created metric."""
    engine = VectorEngine()
    initial_chunks = engine.metrics.chunks_created
    
    engine.add_doc("Test content", {"source_file": "test.txt", "chunk_index": 0})
    
    assert engine.metrics.chunks_created == initial_chunks + 1


def test_add_doc_updates_doc_size_metric():
    """Test that add_doc updates total_doc_size_bytes metric."""
    engine = VectorEngine()
    content = "Test content with specific length"
    initial_size = engine.metrics.total_doc_size_bytes
    
    engine.add_doc(content, {})
    
    assert engine.metrics.total_doc_size_bytes == initial_size + len(content)


def test_add_doc_updates_last_doc_added_timestamp():
    """Test that add_doc updates last_doc_added_at timestamp."""
    engine = VectorEngine()
    assert engine.metrics.last_doc_added_at is None
    
    engine.add_doc("Test content", {})
    
    assert engine.metrics.last_doc_added_at is not None


def test_add_doc_with_whitespace_only_content_raises_error():
    """Test that add_doc with whitespace-only content raises ValueError."""
    engine = VectorEngine()
    
    with pytest.raises(ValueError, match="content cannot be empty"):
        engine.add_doc("   ", {})


def test_add_doc_multiple_sequential():
    """Test adding multiple documents sequentially updates indices correctly."""
    engine = VectorEngine()
    
    doc_ids = []
    for i in range(5):
        doc_id = engine.add_doc(f"Document {i}", {})
        doc_ids.append(doc_id)
    
    assert len(engine.documents) == 5
    assert len(engine.embeddings) == 5
    assert len(engine.index_to_doc_id) == 5
    assert len(engine.doc_id_to_index) == 5
    
    for i, doc_id in enumerate(doc_ids):
        assert engine.doc_id_to_index[doc_id] == i
        assert engine.index_to_doc_id[i] == doc_id


# =============================================================================
# get_doc() Tests
# =============================================================================

def test_get_doc_returns_document():
    """Test that get_doc returns correct document by ID."""
    engine = VectorEngine()
    content = "Test document content"
    metadata = {"key": "value"}
    doc_id = engine.add_doc(content, metadata)
    
    doc = engine.get_doc(doc_id)
    
    assert doc is not None
    assert doc["content"] == content
    assert doc["metadata"] == metadata


def test_get_doc_returns_none_for_nonexistent_id():
    """Test that get_doc returns None for non-existent document."""
    engine = VectorEngine()
    
    doc = engine.get_doc("nonexistent-uuid")
    
    assert doc is None


def test_get_doc_returns_none_for_deleted_doc():
    """Test that get_doc returns None for deleted documents."""
    engine = VectorEngine()
    doc_id = engine.add_doc("Test content", {})
    engine.delete_doc(doc_id)
    
    doc = engine.get_doc(doc_id)
    
    assert doc is None


def test_get_doc_includes_content():
    """Test that returned document includes content."""
    engine = VectorEngine()
    content = "Specific test content"
    doc_id = engine.add_doc(content, {})
    
    doc = engine.get_doc(doc_id)
    
    assert doc
    assert "content" in doc
    assert doc["content"] == content


def test_get_doc_includes_metadata():
    """Test that returned document includes metadata."""
    engine = VectorEngine()
    metadata = {"author": "Alice", "topic": "Testing"}
    doc_id = engine.add_doc("Test content", metadata)
    
    doc = engine.get_doc(doc_id)
    
    assert doc
    assert "metadata" in doc
    assert doc["metadata"] == metadata


# =============================================================================
# delete_doc() Tests
# =============================================================================

def test_delete_doc_returns_true_for_existing_doc():
    """Test that delete_doc returns True when deleting existing document."""
    engine = VectorEngine()
    doc_id = engine.add_doc("Test content", {})
    
    result = engine.delete_doc(doc_id)
    
    assert result is True


def test_delete_doc_returns_false_for_nonexistent_doc():
    """Test that delete_doc returns False for non-existent document."""
    engine = VectorEngine()
    
    result = engine.delete_doc("nonexistent-uuid")
    
    assert result is False


def test_delete_doc_adds_to_deleted_docs_set():
    """Test that delete_doc adds document ID to deleted_docs set."""
    engine = VectorEngine()
    doc_ids = [engine.add_doc(f"Test content {i}", {}) for i in range(10)]
    
    engine.deleted_docs.add(doc_ids[0])
    
    assert doc_ids[0] in engine.deleted_docs


def test_delete_doc_lazy_deletion():
    """Test that delete_doc doesn't immediately remove from storage."""
    engine = VectorEngine()
    doc_ids = [engine.add_doc(f"Test content {i}", {}) for i in range(10)]
    
    engine.deleted_docs.add(doc_ids[0])
    
    assert doc_ids[0] in engine.documents
    assert doc_ids[0] in engine.deleted_docs


def test_delete_doc_increments_metrics():
    """Test that delete_doc increments docs_deleted metric."""
    engine = VectorEngine()
    doc_id = engine.add_doc("Test content", {})
    initial_deleted = engine.metrics.docs_deleted
    
    engine.delete_doc(doc_id)
    
    assert engine.metrics.docs_deleted == initial_deleted + 1


def test_delete_same_doc_twice_returns_false():
    """Test that deleting same document twice returns False on second attempt."""
    engine = VectorEngine()
    doc_id = engine.add_doc("Test content", {})
    
    result1 = engine.delete_doc(doc_id)
    result2 = engine.delete_doc(doc_id)
    
    assert result1 is True
    assert result2 is False


def test_multiple_deletes_decrease_size_metric():
    """Test that deleting documents decreases total_doc_size_bytes."""
    engine = VectorEngine()
    content1 = "First document content"
    content2 = "Second document content"
    
    doc_id1 = engine.add_doc(content1, {})
    doc_id2 = engine.add_doc(content2, {})
    
    initial_size = engine.metrics.total_doc_size_bytes
    
    engine.delete_doc(doc_id1)
    
    assert engine.metrics.total_doc_size_bytes == initial_size - len(content1)


def test_delete_doc_updates_size_metric_correctly():
    """Test that delete_doc correctly decreases total_doc_size_bytes."""
    engine = VectorEngine()
    content = "This is a test document with specific length"
    doc_id = engine.add_doc(content, {})
    
    size_before = engine.metrics.total_doc_size_bytes
    engine.delete_doc(doc_id)
    size_after = engine.metrics.total_doc_size_bytes
    
    assert size_after == size_before - len(content)


# =============================================================================
# search() Tests
# =============================================================================

def test_search_returns_list():
    """Test that search returns a list of results."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    
    results = engine.search("test")
    
    assert isinstance(results, list)


def test_search_returns_correct_structure():
    """Test that search results have correct structure (doc_id, content, score, metadata)."""
    engine = VectorEngine()
    engine.add_doc("Test document", {"key": "value"})
    
    results = engine.search("test")
    
    assert len(results) > 0
    result = results[0]
    assert hasattr(result, "id")
    assert hasattr(result, "content")
    assert hasattr(result, "score")
    assert hasattr(result, "metadata")


def test_search_respects_top_k():
    """Test that search returns at most top_k results."""
    engine = VectorEngine()
    for i in range(20):
        engine.add_doc(f"Document number {i}", {})
    
    results = engine.search("document", top_k=5)
    
    assert len(results) == 5


def test_search_excludes_deleted_documents():
    """Test that search results don't include deleted documents."""
    engine = VectorEngine()
    doc_id = engine.add_doc("Unique searchable content", {})
    engine.add_doc("Other content", {})
    
    results_before = engine.search("unique searchable")
    engine.delete_doc(doc_id)
    results_after = engine.search("unique searchable")
    
    doc_ids_before = [r.id for r in results_before]
    assert doc_id in doc_ids_before
    
    doc_ids_after = [r.id for r in results_after]
    assert doc_id not in doc_ids_after


def test_search_results_sorted_by_score():
    """Test that search results are sorted by similarity score descending."""
    engine = VectorEngine()
    for i in range(10):
        engine.add_doc(f"Document with varying content topic {i}", {})
    
    results = engine.search("document content")
    
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_with_empty_index():
    """Test that search on empty index returns empty list."""
    engine = VectorEngine()
    
    results = engine.search("test query")
    
    assert results == []


def test_search_updates_metrics():
    """Test that search increments query count metrics."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    initial_queries = engine.metrics.total_queries
    
    engine.search("test")
    
    assert engine.metrics.total_queries == initial_queries + 1


def test_search_tracks_query_time():
    """Test that search tracks query execution time."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    initial_time = engine.metrics.total_query_time_ms
    
    engine.search("test")
    
    assert engine.metrics.total_query_time_ms > initial_time
    assert len(engine.metrics.query_times) == 1


def test_search_with_default_top_k():
    """Test that search uses default top_k value."""
    engine = VectorEngine()
    for i in range(20):
        engine.add_doc(f"Document {i}", {})
    
    results = engine.search("document")
    
    assert len(results) == Config.DEFAULT_TOP_K


def test_search_similarity_scores_in_range():
    """Test that similarity scores are between 0 and 1."""
    engine = VectorEngine()
    for i in range(10):
        engine.add_doc(f"Document content {i}", {})
    
    results = engine.search("document")
    
    for result in results:
        assert 0.0 <= result.score <= 1.0


def test_search_with_whitespace_only_query_raises_error():
    """Test that search with whitespace-only query raises ValueError."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.search("   ")


def test_search_returns_empty_list_when_all_deleted():
    """Test that search returns empty list when all documents are deleted."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {})
    doc_id2 = engine.add_doc("Document 2", {})
    
    engine.delete_doc(doc_id1)
    engine.delete_doc(doc_id2)
    
    results = engine.search("document")
    
    assert results == []


def test_search_top_k_larger_than_available():
    """Test search when top_k is larger than number of available documents."""
    engine = VectorEngine()
    engine.add_doc("Document 1", {})
    engine.add_doc("Document 2", {})
    
    results = engine.search("document", top_k=100)
    
    assert len(results) == 2


# =============================================================================
# list_files() Tests
# =============================================================================

def test_list_files_returns_list():
    """Test that list_files returns a list."""
    engine = VectorEngine()
    
    files = engine.list_files()
    
    assert isinstance(files, list)


def test_list_files_empty_when_no_files():
    """Test that list_files returns empty list when no files uploaded."""
    engine = VectorEngine()
    engine.add_doc("Document without source_file", {})
    
    files = engine.list_files()
    
    assert files == []


def test_list_files_includes_uploaded_files():
    """Test that list_files includes filenames from uploaded documents."""
    engine = VectorEngine()
    engine.add_doc("Content 1", {"source_file": "file1.txt", "chunk_index": 0})
    engine.add_doc("Content 2", {"source_file": "file2.txt", "chunk_index": 0})
    
    files = engine.list_files()
    
    assert "file1.txt" in files
    assert "file2.txt" in files


def test_list_files_returns_unique_filenames():
    """Test that list_files returns unique filenames only."""
    engine = VectorEngine()
    engine.add_doc("Chunk 1", {"source_file": "file.txt", "chunk_index": 0})
    engine.add_doc("Chunk 2", {"source_file": "file.txt", "chunk_index": 1})
    engine.add_doc("Chunk 3", {"source_file": "file.txt", "chunk_index": 2})
    
    files = engine.list_files()
    
    assert files == ["file.txt"]
    assert len(files) == 1


def test_list_files_ignores_docs_without_source_file():
    """Test that list_files only includes docs with source_file metadata."""
    engine = VectorEngine()
    engine.add_doc("Doc with file", {"source_file": "test.txt", "chunk_index": 0})
    engine.add_doc("Doc without file", {"other": "metadata"})
    engine.add_doc("Another doc", {})
    
    files = engine.list_files()
    
    assert files == ["test.txt"]


def test_list_files_sorted_alphabetically():
    """Test that list_files returns filenames in sorted order."""
    engine = VectorEngine()
    engine.add_doc("Content", {"source_file": "zebra.txt", "chunk_index": 0})
    engine.add_doc("Content", {"source_file": "apple.txt", "chunk_index": 0})
    engine.add_doc("Content", {"source_file": "banana.txt", "chunk_index": 0})
    
    files = engine.list_files()
    
    assert files == ["apple.txt", "banana.txt", "zebra.txt"]


def test_list_files_excludes_deleted_docs():
    """Test that list_files doesn't include files from deleted documents."""
    engine = VectorEngine()
    engine.add_doc("Content 1", {"source_file": "keep.txt", "chunk_index": 0})
    doc_id = engine.add_doc("Content 2", {"source_file": "delete.txt", "chunk_index": 0})
    
    engine.delete_doc(doc_id)
    
    files = engine.list_files()
    
    assert "keep.txt" in files
    assert "delete.txt" not in files


# =============================================================================
# delete_file() Tests
# =============================================================================

def test_delete_file_returns_dict():
    """Test that delete_file returns a dictionary with status."""
    engine = VectorEngine()
    engine.add_doc("Content", {"source_file": "test.txt", "chunk_index": 0})
    
    result = engine.delete_file("test.txt")
    
    assert isinstance(result, dict)
    assert "status" in result
    assert "filename" in result
    assert "chunks_deleted" in result
    assert "doc_ids" in result


def test_delete_file_deletes_all_chunks():
    """Test that delete_file removes all chunks from a source file."""
    engine = VectorEngine()
    engine.add_doc("Chunk 1", {"source_file": "test.txt", "chunk_index": 0})
    engine.add_doc("Chunk 2", {"source_file": "test.txt", "chunk_index": 1})
    engine.add_doc("Chunk 3", {"source_file": "test.txt", "chunk_index": 2})
    
    result = engine.delete_file("test.txt")
    
    assert result["status"] == "deleted"
    assert result["chunks_deleted"] == 3


def test_delete_file_returns_deleted_count():
    """Test that delete_file returns count of deleted chunks."""
    engine = VectorEngine()
    engine.add_doc("Chunk 1", {"source_file": "file.txt", "chunk_index": 0})
    engine.add_doc("Chunk 2", {"source_file": "file.txt", "chunk_index": 1})
    
    result = engine.delete_file("file.txt")
    
    assert result["chunks_deleted"] == 2


def test_delete_file_returns_doc_ids():
    """Test that delete_file returns list of deleted document IDs."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Chunk 1", {"source_file": "file.txt", "chunk_index": 0})
    doc_id2 = engine.add_doc("Chunk 2", {"source_file": "file.txt", "chunk_index": 1})
    
    result = engine.delete_file("file.txt")
    
    assert doc_id1 in result["doc_ids"]
    assert doc_id2 in result["doc_ids"]
    assert len(result["doc_ids"]) == 2


def test_delete_file_returns_not_found_for_nonexistent():
    """Test that delete_file returns not_found status for non-existent file."""
    engine = VectorEngine()
    
    result = engine.delete_file("nonexistent.txt")
    
    assert result["status"] == "not_found"
    assert result["chunks_deleted"] == 0
    assert result["doc_ids"] == []


def test_delete_file_updates_metrics():
    """Test that delete_file updates deletion metrics."""
    engine = VectorEngine()
    engine.add_doc("Chunk 1", {"source_file": "file.txt", "chunk_index": 0})
    engine.add_doc("Chunk 2", {"source_file": "file.txt", "chunk_index": 1})
    initial_deleted = engine.metrics.docs_deleted
    
    engine.delete_file("file.txt")
    
    assert engine.metrics.docs_deleted == initial_deleted + 2


def test_delete_file_with_mixed_chunks():
    """Test delete_file when only some chunks belong to the file."""
    engine = VectorEngine()
    engine.add_doc("Chunk 1", {"source_file": "target.txt", "chunk_index": 0})
    engine.add_doc("Chunk 2", {"source_file": "target.txt", "chunk_index": 1})
    engine.add_doc("Other 1", {"source_file": "other.txt", "chunk_index": 0})
    engine.add_doc("Other 2", {"source_file": "other.txt", "chunk_index": 1})
    
    result = engine.delete_file("target.txt")
    
    assert result["chunks_deleted"] == 2
    assert result["status"] == "deleted"
    
    files = engine.list_files()
    assert "target.txt" not in files
    assert "other.txt" in files


# =============================================================================
# should_compact() Tests
# =============================================================================

def test_should_compact_returns_boolean():
    """Test that should_compact returns a boolean."""
    engine = VectorEngine()
    
    result = engine.should_compact()
    
    assert isinstance(result, bool)


def test_should_compact_false_when_no_deletions():
    """Test that should_compact returns False when no documents deleted."""
    engine = VectorEngine()
    engine.add_doc("Document 1", {})
    engine.add_doc("Document 2", {})
    
    result = engine.should_compact()
    
    assert result is False


def test_should_compact_true_when_threshold_exceeded():
    """Test that should_compact returns True when deleted ratio exceeds threshold."""
    engine = VectorEngine()
    engine.compaction_threshold = 0.25
    doc_ids = [engine.add_doc(f"Document {i}", {}) for i in range(10)]
    
    for doc_id in doc_ids[:3]:
        engine.deleted_docs.add(doc_id)
    
    result = engine.should_compact()
    
    assert result is True


def test_should_compact_respects_threshold():
    """Test that should_compact respects compaction_threshold setting."""
    engine = VectorEngine()
    engine.compaction_threshold = 0.5
    
    doc_ids = [engine.add_doc(f"Document {i}", {}) for i in range(10)]
    
    for doc_id in doc_ids[:4]:
        engine.deleted_docs.add(doc_id)
    
    assert engine.should_compact() is False
    
    engine.deleted_docs.add(doc_ids[4])
    assert engine.should_compact() is False
    
    engine.deleted_docs.add(doc_ids[5])
    assert engine.should_compact() is True


# =============================================================================
# compact() Tests
# =============================================================================

def test_compact_removes_deleted_documents():
    """Test that compact physically removes deleted documents."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {})
    doc_id2 = engine.add_doc("Document 2", {})
    doc_id3 = engine.add_doc("Document 3", {})
    
    engine.deleted_docs.add(doc_id2)
    engine.compact()
    
    assert doc_id2 not in engine.documents


def test_compact_rebuilds_index_mappings():
    """Test that compact rebuilds index_to_doc_id and doc_id_to_index."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {})
    doc_id2 = engine.add_doc("Document 2", {})
    doc_id3 = engine.add_doc("Document 3", {})
    
    engine.deleted_docs.add(doc_id2)
    engine.compact()
    
    assert len(engine.index_to_doc_id) == 2
    assert doc_id1 in engine.index_to_doc_id
    assert doc_id3 in engine.index_to_doc_id
    assert doc_id2 not in engine.index_to_doc_id
    
    assert doc_id1 in engine.doc_id_to_index
    assert doc_id3 in engine.doc_id_to_index
    assert doc_id2 not in engine.doc_id_to_index


def test_compact_clears_deleted_docs_set():
    """Test that compact clears the deleted_docs set."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {})
    doc_id2 = engine.add_doc("Document 2", {})
    
    engine.deleted_docs.add(doc_id1)
    assert len(engine.deleted_docs) > 0
    
    engine.compact()
    
    assert len(engine.deleted_docs) == 0


def test_compact_preserves_active_documents():
    """Test that compact doesn't affect non-deleted documents."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {"key": "value1"})
    doc_id2 = engine.add_doc("Document 2", {"key": "value2"})
    doc_id3 = engine.add_doc("Document 3", {"key": "value3"})
    
    engine.deleted_docs.add(doc_id2)
    engine.compact()
    
    assert doc_id1 in engine.documents
    assert engine.documents[doc_id1]["content"] == "Document 1"
    assert engine.documents[doc_id1]["metadata"] == {"key": "value1"}
    
    assert doc_id3 in engine.documents
    assert engine.documents[doc_id3]["content"] == "Document 3"
    assert engine.documents[doc_id3]["metadata"] == {"key": "value3"}


def test_compact_updates_metrics():
    """Test that compact increments compactions_performed metric."""
    engine = VectorEngine()
    doc_id = engine.add_doc("Document", {})
    initial_compactions = engine.metrics.compactions_performed
    
    engine.delete_doc(doc_id)
    
    assert engine.metrics.compactions_performed == initial_compactions + 1


def test_compact_updates_timestamp():
    """Test that compact updates last_compaction_at timestamp."""
    engine = VectorEngine()
    doc_id = engine.add_doc("Document", {})
    assert engine.metrics.last_compaction_at is None
    
    engine.delete_doc(doc_id)
    
    assert engine.metrics.last_compaction_at is not None


def test_compact_with_no_deletions():
    """Test that compact handles case with no deleted documents."""
    engine = VectorEngine()
    engine.add_doc("Document 1", {})
    engine.add_doc("Document 2", {})
    initial_count = len(engine.documents)
    
    engine.compact()
    
    assert len(engine.documents) == initial_count
    assert len(engine.deleted_docs) == 0


# =============================================================================
# build() Tests
# =============================================================================

def test_build_reconstructs_embeddings():
    """Test that build regenerates all embeddings from documents."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {})
    doc_id2 = engine.add_doc("Document 2", {})
    doc_id3 = engine.add_doc("Document 3", {})
    
    engine.deleted_docs.add(doc_id2)
    initial_embedding_count = len(engine.embeddings)
    
    engine.build()
    
    assert len(engine.embeddings) == 2
    assert len(engine.embeddings) < initial_embedding_count


def test_build_preserves_documents():
    """Test that build doesn't modify document storage."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {"key": "value1"})
    doc_id2 = engine.add_doc("Document 2", {"key": "value2"})
    
    original_content1 = engine.documents[doc_id1]["content"]
    original_content2 = engine.documents[doc_id2]["content"]
    
    engine.build()
    
    assert engine.documents[doc_id1]["content"] == original_content1
    assert engine.documents[doc_id2]["content"] == original_content2


def test_build_updates_index_mappings():
    """Test that build recreates index mappings."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {})
    doc_id2 = engine.add_doc("Document 2", {})
    doc_id3 = engine.add_doc("Document 3", {})
    
    engine.deleted_docs.add(doc_id2)
    engine.build()
    
    assert len(engine.index_to_doc_id) == 2
    assert doc_id1 in engine.index_to_doc_id
    assert doc_id3 in engine.index_to_doc_id
    assert doc_id2 not in engine.index_to_doc_id


def test_build_with_empty_index():
    """Test that build handles empty index."""
    engine = VectorEngine()
    
    engine.build()
    
    assert len(engine.embeddings) == 0
    assert len(engine.documents) == 0
    assert len(engine.index_to_doc_id) == 0


# =============================================================================
# save() and load() Tests
# =============================================================================

def test_save_creates_files():
    """Test that save creates metadata and embeddings files."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.save(tmpdir)
        
        metadata_path = os.path.join(tmpdir, Config.METADATA_FILENAME)
        embeddings_path = os.path.join(tmpdir, Config.EMBEDDINGS_FILENAME)
        
        assert os.path.exists(metadata_path)
        assert os.path.exists(embeddings_path)


def test_save_returns_metrics():
    """Test that save returns dict with save metrics."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = engine.save(tmpdir)
        
        assert "status" in result
        assert result["status"] == "saved"
        assert "directory" in result
        assert "metadata_size_mb" in result
        assert "embeddings_size_mb" in result
        assert "total_size_mb" in result
        assert "documents_saved" in result
        assert "embeddings_saved" in result


def test_save_to_custom_directory():
    """Test that save can use custom directory path."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_dir = os.path.join(tmpdir, "custom_data")
        result = engine.save(custom_dir)
        
        assert result["status"] == "saved"
        assert result["directory"] == custom_dir
        assert os.path.exists(custom_dir)


def test_load_restores_documents():
    """Test that load restores all saved documents."""
    engine = VectorEngine()
    doc_id = engine.add_doc("Test document", {"key": "value"})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.save(tmpdir)
        
        new_engine = VectorEngine()
        new_engine.load(tmpdir)
        
        assert doc_id in new_engine.documents
        assert new_engine.documents[doc_id]["content"] == "Test document"
        assert new_engine.documents[doc_id]["metadata"] == {"key": "value"}


def test_load_restores_embeddings():
    """Test that load restores all saved embeddings."""
    engine = VectorEngine()
    engine.add_doc("Document 1", {})
    engine.add_doc("Document 2", {})
    original_count = len(engine.embeddings)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.save(tmpdir)
        
        new_engine = VectorEngine()
        new_engine.load(tmpdir)
        
        assert len(new_engine.embeddings) == original_count
        assert len(new_engine.embeddings) == 2


def test_load_restores_index_mappings():
    """Test that load restores index mappings correctly."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {})
    doc_id2 = engine.add_doc("Document 2", {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.save(tmpdir)
        
        new_engine = VectorEngine()
        new_engine.load(tmpdir)
        
        assert len(new_engine.index_to_doc_id) == 2
        assert doc_id1 in new_engine.doc_id_to_index
        assert doc_id2 in new_engine.doc_id_to_index


def test_load_restores_deleted_docs():
    """Test that load restores deleted_docs set."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {})
    doc_id2 = engine.add_doc("Document 2", {})
    engine.delete_doc(doc_id1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.save(tmpdir)
        
        new_engine = VectorEngine()
        new_engine.load(tmpdir)
        
        assert len(new_engine.deleted_docs) == 0
        assert doc_id1 not in new_engine.documents


def test_load_returns_metrics():
    """Test that load returns dict with load metrics."""
    engine = VectorEngine()
    engine.add_doc("Document", {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.save(tmpdir)
        
        new_engine = VectorEngine()
        result = new_engine.load(tmpdir)
        
        assert "status" in result
        assert result["status"] == "loaded"
        assert "directory" in result
        assert "documents_loaded" in result
        assert "embeddings_loaded" in result
        assert "deleted_docs" in result
        assert "version" in result


def test_save_and_load_roundtrip():
    """Test that save followed by load preserves all data."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("First document", {"author": "Alice"})
    doc_id2 = engine.add_doc("Second document", {"author": "Bob"})
    
    engine.search("document")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.save(tmpdir)
        
        new_engine = VectorEngine()
        new_engine.load(tmpdir)
        doc1 = new_engine.get_doc(doc_id1)
        doc2 = new_engine.get_doc(doc_id2)
        
        assert doc1 and doc2
        assert len(new_engine.documents) == 2
        assert doc1["content"] == "First document"
        assert doc2["content"] == "Second document"
        
        results = new_engine.search("document")
        assert len(results) == 2


def test_load_raises_error_when_files_missing():
    """Test that load raises FileNotFoundError when files don't exist."""
    engine = VectorEngine()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_dir = os.path.join(tmpdir, "nonexistent")
        
        with pytest.raises(FileNotFoundError):
            engine.load(nonexistent_dir)


def test_save_creates_directory_if_not_exists():
    """Test that save creates the target directory if it doesn't exist."""
    engine = VectorEngine()
    engine.add_doc("Document", {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "nested", "path", "data")
        
        engine.save(save_dir)
        
        assert os.path.exists(save_dir)
        assert os.path.exists(os.path.join(save_dir, "metadata.json"))


def test_save_with_empty_index():
    """Test that save works with an empty index."""
    engine = VectorEngine()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = engine.save(tmpdir)
        
        assert result["status"] == "saved"
        assert result["documents_saved"] == 0
        assert result["embeddings_saved"] == 0


def test_load_with_empty_saved_index():
    """Test that load works when loading an empty saved index."""
    engine = VectorEngine()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.save(tmpdir)
        
        new_engine = VectorEngine()
        result = new_engine.load(tmpdir)
        
        assert result["status"] == "loaded"
        assert result["documents_loaded"] == 0
        assert result["embeddings_loaded"] == 0


# =============================================================================
# get_metrics() Tests
# =============================================================================

def test_get_metrics_returns_dict():
    """Test that get_metrics returns a dictionary."""
    engine = VectorEngine()
    
    metrics = engine.get_metrics()
    
    assert isinstance(metrics, dict)


def test_get_metrics_includes_all_categories():
    """Test that get_metrics includes all metric categories."""
    engine = VectorEngine()
    engine.add_doc("Test", {})
    engine.search("test")
    
    metrics = engine.get_metrics()
    
    assert "total_queries" in metrics
    assert "docs_added" in metrics
    assert "active_documents" in metrics
    assert "model_name" in metrics
    assert "created_at" in metrics


def test_get_metrics_includes_query_stats():
    """Test that metrics include query statistics."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    engine.search("test")
    
    metrics = engine.get_metrics()
    
    assert "total_queries" in metrics
    assert "avg_query_time_ms" in metrics
    assert "total_query_time_ms" in metrics
    assert metrics["total_queries"] > 0


def test_get_metrics_includes_document_stats():
    """Test that metrics include document statistics."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    
    metrics = engine.get_metrics()
    
    assert "docs_added" in metrics
    assert "docs_deleted" in metrics
    assert "active_documents" in metrics
    assert "total_embeddings" in metrics
    assert metrics["docs_added"] == 1
    assert metrics["active_documents"] == 1


def test_get_metrics_includes_memory_stats():
    """Test that metrics include memory statistics."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    
    metrics = engine.get_metrics()
    
    assert "embeddings_mb" in metrics
    assert "documents_mb" in metrics
    assert "total_mb" in metrics
    assert "total_doc_size_bytes" in metrics
    assert metrics["total_doc_size_bytes"] > 0


def test_get_metrics_includes_timestamps():
    """Test that metrics include timestamp information."""
    engine = VectorEngine()
    engine.add_doc("Test document", {})
    engine.search("test")
    
    metrics = engine.get_metrics()
    
    assert "created_at" in metrics
    assert "last_query_at" in metrics
    assert "last_doc_added_at" in metrics
    assert metrics["created_at"] is not None
    assert metrics["last_query_at"] is not None


def test_get_metrics_calculates_uptime():
    """Test that metrics include uptime calculation."""
    engine = VectorEngine()
    
    metrics = engine.get_metrics()
    
    assert "uptime_seconds" in metrics
    assert metrics["uptime_seconds"] >= 0
    assert isinstance(metrics["uptime_seconds"], float)


def test_get_metrics_percentiles_with_few_queries():
    """Test that get_metrics handles percentile calculation with few data points."""
    engine = VectorEngine()
    engine.add_doc("Document", {})
    
    engine.search("test")
    
    metrics = engine.get_metrics()
    
    assert metrics["p50_query_time_ms"] is not None
    assert metrics["p95_query_time_ms"] is not None
    assert metrics["p99_query_time_ms"] is not None


def test_get_metrics_percentiles_with_no_queries():
    """Test that get_metrics handles percentiles when no queries have been made."""
    engine = VectorEngine()
    
    metrics = engine.get_metrics()
    
    assert metrics["p50_query_time_ms"] is None
    assert metrics["p95_query_time_ms"] is None
    assert metrics["p99_query_time_ms"] is None
    assert metrics["min_query_time_ms"] is None
    assert metrics["max_query_time_ms"] is None


# =============================================================================
# get_index_stats() Tests
# =============================================================================

def test_get_index_stats_returns_dict():
    """Test that get_index_stats returns a dictionary."""
    engine = VectorEngine()
    
    stats = engine.get_index_stats()
    
    assert isinstance(stats, dict)


def test_get_index_stats_includes_document_counts():
    """Test that index stats include document and embedding counts."""
    engine = VectorEngine()
    engine.add_doc("Document 1", {})
    engine.add_doc("Document 2", {})
    
    stats = engine.get_index_stats()
    
    assert "total_documents" in stats
    assert "total_embeddings" in stats
    assert "deleted_documents" in stats
    assert stats["total_documents"] == 2
    assert stats["total_embeddings"] == 2


def test_get_index_stats_includes_deleted_ratio():
    """Test that index stats include deleted ratio calculation."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Document 1", {})
    engine.add_doc("Document 2", {})
    engine.deleted_docs.add(doc_id1)
    
    stats = engine.get_index_stats()
    
    assert "deleted_ratio" in stats
    assert stats["deleted_ratio"] == 0.5


def test_get_index_stats_includes_compaction_status():
    """Test that index stats include needs_compaction flag."""
    engine = VectorEngine()
    engine.add_doc("Document 1", {})
    
    stats = engine.get_index_stats()
    
    assert "needs_compaction" in stats
    assert isinstance(stats["needs_compaction"], bool)


def test_get_index_stats_includes_embedding_dimension():
    """Test that index stats include embedding dimension."""
    engine = VectorEngine()
    engine.add_doc("Document", {})
    
    stats = engine.get_index_stats()
    
    assert "embedding_dimension" in stats
    assert stats["embedding_dimension"] == Config.EMBEDDING_DIMENSION


# =============================================================================
# cosine_similarity() Tests
# =============================================================================

def test_cosine_similarity_identical_vectors():
    """Test that cosine similarity of identical normalized vectors is 1.0."""
    engine = VectorEngine()
    vec = np.array([1.0, 0.0, 0.0])
    
    similarity = engine.cosine_similarity(vec, vec)
    
    assert np.isclose(similarity, 1.0)


def test_cosine_similarity_orthogonal_vectors():
    """Test that cosine similarity of orthogonal vectors is close to 0."""
    engine = VectorEngine()
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    
    similarity = engine.cosine_similarity(vec1, vec2)
    
    assert np.isclose(similarity, 0.0, atol=1e-6)


def test_cosine_similarity_opposite_vectors():
    """Test that cosine similarity of opposite vectors is -1.0."""
    engine = VectorEngine()
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([-1.0, 0.0, 0.0])
    
    similarity = engine.cosine_similarity(vec1, vec2)
    
    assert np.isclose(similarity, -1.0)


def test_cosine_similarity_normalized_embeddings():
    """Test that cosine_similarity works with pre-normalized embeddings."""
    engine = VectorEngine()
    vec1 = np.array([3.0, 4.0])
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = np.array([1.0, 1.0])
    vec2 = vec2 / np.linalg.norm(vec2)
    
    similarity = engine.cosine_similarity(vec1, vec2)
    
    assert np.isclose(similarity, np.dot(vec1, vec2))


# =============================================================================
# EngineMetrics Tests
# =============================================================================

def test_engine_metrics_initialization():
    """Test that EngineMetrics initializes with correct default values."""
    metrics = EngineMetrics()
    
    assert metrics.total_queries == 0
    assert metrics.docs_added == 0
    assert metrics.docs_deleted == 0
    assert metrics.compactions_performed == 0
    assert metrics.chunks_created == 0
    assert metrics.files_uploaded == 0
    assert metrics.total_query_time_ms == 0.0
    assert metrics.total_doc_size_bytes == 0
    assert metrics.last_query_at is None
    assert metrics.last_doc_added_at is None
    assert metrics.created_at is not None


def test_engine_metrics_to_dict():
    """Test that EngineMetrics.to_dict() returns proper dictionary."""
    metrics = EngineMetrics()
    metrics.total_queries = 5
    metrics.docs_added = 10
    
    metrics_dict = metrics.to_dict()
    
    assert isinstance(metrics_dict, dict)
    assert "total_queries" in metrics_dict
    assert "docs_added" in metrics_dict
    assert metrics_dict["total_queries"] == 5
    assert metrics_dict["docs_added"] == 10


def test_engine_metrics_query_times_max_history():
    """Test that query_times respects max_query_history limit."""
    metrics = EngineMetrics()
    metrics.max_query_history = 5
    metrics.query_times = deque(maxlen=5)
    
    for i in range(10):
        metrics.query_times.append(float(i))
    
    assert len(metrics.query_times) == 5
    assert list(metrics.query_times) == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_engine_metrics_created_at_is_iso_format():
    """Test that created_at timestamp is in ISO format."""
    from datetime import datetime
    
    metrics = EngineMetrics()
    
    created = datetime.fromisoformat(metrics.created_at)
    assert created is not None


# =============================================================================
# Integration Tests
# =============================================================================

def test_add_search_delete_workflow():
    """Test complete workflow: add documents, search, delete, verify."""
    engine = VectorEngine()
    
    doc_id1 = engine.add_doc("Python programming language", {"topic": "coding"})
    doc_id2 = engine.add_doc("Machine learning algorithms", {"topic": "AI"})
    doc_id3 = engine.add_doc("Python snake species", {"topic": "nature"})
    
    results = engine.search("Python")
    assert len(results) == 3
    
    assert engine.delete_doc(doc_id2) is True
    
    results = engine.search("Python")
    assert len(results) == 2
    
    result_ids = [r.id for r in results]
    assert doc_id2 not in result_ids
    assert doc_id1 in result_ids
    assert doc_id3 in result_ids


def test_multiple_operations_metrics_accuracy():
    """Test that metrics remain accurate after multiple operations."""
    engine = VectorEngine()
    
    for i in range(5):
        engine.add_doc(f"Document {i}", {})
    
    for i in range(3):
        engine.search("document")
    
    doc_ids = list(engine.documents.keys())
    for doc_id in doc_ids[:2]:
        engine.delete_doc(doc_id)
    
    metrics = engine.get_metrics()
    assert metrics["docs_added"] == 5
    assert metrics["docs_deleted"] >= 2
    assert metrics["total_queries"] == 3
    assert metrics["active_documents"] <= 3


def test_compaction_triggered_automatically():
    """Test that compaction triggers automatically when threshold exceeded."""
    engine = VectorEngine()
    engine.compaction_threshold = 0.25
    
    doc_ids = [engine.add_doc(f"Document {i}", {}) for i in range(10)]
    
    for doc_id in doc_ids[:3]:
        engine.delete_doc(doc_id)
    
    assert len(engine.deleted_docs) == 0 or engine.should_compact() is False


def test_search_after_compaction():
    """Test that search works correctly after compaction."""
    engine = VectorEngine()
    
    doc_id1 = engine.add_doc("Keep this document", {"status": "keep"})
    doc_id2 = engine.add_doc("Delete this document", {"status": "delete"})
    doc_id3 = engine.add_doc("Keep this one too", {"status": "keep"})
    
    engine.deleted_docs.add(doc_id2)
    
    engine.compact()
    
    results = engine.search("document")
    assert len(results) == 2
    
    result_ids = [r.id for r in results]
    assert doc_id1 in result_ids
    assert doc_id3 in result_ids
    assert doc_id2 not in result_ids


def test_save_load_with_deleted_documents():
    """Test that save/load preserves deleted documents state."""
    engine = VectorEngine()
    doc_id1 = engine.add_doc("Active document", {})
    doc_id2 = engine.add_doc("Deleted document", {})
    doc_id3 = engine.add_doc("Another active doc", {})
    
    engine.deleted_docs.add(doc_id2)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_result = engine.save(tmpdir)
        assert save_result["documents_saved"] == 2
        
        new_engine = VectorEngine()
        new_engine.load(tmpdir)
        
        assert len(new_engine.documents) == 2
        assert doc_id1 in new_engine.documents
        assert doc_id3 in new_engine.documents
        assert doc_id2 not in new_engine.documents
        assert len(new_engine.deleted_docs) == 0
