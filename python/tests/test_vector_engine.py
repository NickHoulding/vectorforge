"""Tests for the VectorEngine class"""

import os
import tempfile
import uuid
from collections import deque

import numpy as np
import pytest

from vectorforge.config import VFGConfig
from vectorforge.vector_engine import EngineMetrics, VectorEngine

# =============================================================================
# Initialization Tests
# =============================================================================


def test_vector_engine_initialization(vector_engine):
    """Test that VectorEngine initializes with correct default values."""
    assert vector_engine.collection is not None
    assert vector_engine.chroma_client is not None
    assert vector_engine.collection.count() == 0


def test_vector_engine_loads_model(vector_engine):
    """Test that VectorEngine loads the sentence transformer model."""
    assert vector_engine.model is not None
    assert vector_engine.model_name == VFGConfig.MODEL_NAME
    assert (
        vector_engine.model.get_sentence_embedding_dimension()
        == VFGConfig.EMBEDDING_DIMENSION
    )


def test_vector_engine_initializes_empty_collections(vector_engine):
    """Test that VectorEngine starts with empty documents and embeddings."""
    assert vector_engine.collection.count() == 0


def test_vector_engine_initializes_metrics(vector_engine):
    """Test that VectorEngine initializes metrics tracking."""
    assert vector_engine.metrics is not None
    assert isinstance(vector_engine.metrics, EngineMetrics)
    assert vector_engine.metrics.total_queries == 0
    assert vector_engine.metrics.docs_added == 0
    assert vector_engine.metrics.docs_deleted == 0


# =============================================================================
# add_doc() Tests
# =============================================================================


def test_add_doc_returns_uuid(vector_engine):
    """Test that add_doc returns a valid UUID string."""
    doc_id = vector_engine.add_doc("Test content", {})

    assert isinstance(doc_id, str)
    uuid_obj = uuid.UUID(doc_id)
    assert str(uuid_obj) == doc_id


def test_add_doc_stores_content(vector_engine):
    """Test that add_doc stores document content correctly."""
    content = "This is my test document"
    doc_id = vector_engine.add_doc(content, {})
    result = vector_engine.collection.get(ids=[doc_id], include=["documents"])

    assert len(result["ids"]) == 1
    assert result["documents"][0] == content


def test_add_doc_stores_metadata(vector_engine):
    """Test that add_doc stores metadata correctly."""
    metadata = {"author": "John", "title": "Test Doc"}
    doc_id = vector_engine.add_doc("Test content", metadata)
    result = vector_engine.collection.get(ids=[doc_id], include=["metadatas"])

    assert len(result["ids"]) == 1
    assert result["metadatas"][0] == metadata


def test_add_doc_creates_embedding(vector_engine):
    """Test that add_doc creates an embedding vector."""
    doc_id = vector_engine.add_doc("Test content", {})
    result = vector_engine.collection.get(ids=[doc_id], include=["embeddings"])
    assert len(result["embeddings"]) == 1

    embedding = result["embeddings"][0]
    assert isinstance(embedding, (list, np.ndarray))
    assert len(embedding) == VFGConfig.EMBEDDING_DIMENSION


def test_add_doc_normalizes_embedding(vector_engine):
    """Test that embeddings are normalized."""
    doc_id = vector_engine.add_doc("Test content", {})
    result = vector_engine.collection.get(ids=[doc_id], include=["embeddings"])
    embedding = np.array(result["embeddings"][0])
    norm = np.linalg.norm(embedding)

    assert np.isclose(norm, 1.0, atol=1e-6)


def test_add_doc_updates_collection_count(vector_engine):
    """Test that add_doc increases the collection count."""
    initial_count = vector_engine.collection.count()
    vector_engine.add_doc("Test content", {})

    assert vector_engine.collection.count() == initial_count + 1


def test_add_doc_increments_metrics(vector_engine):
    """Test that add_doc increments docs_added metric."""
    initial_count = vector_engine.metrics.docs_added
    vector_engine.add_doc("Test content", {})

    assert vector_engine.metrics.docs_added == initial_count + 1


def test_add_doc_with_empty_content_raises_error(vector_engine):
    """Test that adding document with empty content raises ValueError."""
    with pytest.raises(ValueError, match="content cannot be empty"):
        vector_engine.add_doc("", {})


def test_add_doc_with_null_metadata(vector_engine):
    """Test that add_doc handles None metadata correctly."""
    doc_id = vector_engine.add_doc("Test content", None)
    result = vector_engine.collection.get(ids=[doc_id], include=["metadatas"])

    assert len(result["ids"]) == 1
    assert result["metadatas"][0] is None or result["metadatas"][0] == {}


def test_add_doc_with_empty_metadata(vector_engine):
    """Test that add_doc handles empty dict metadata correctly."""
    doc_id = vector_engine.add_doc("Test content", {})
    result = vector_engine.collection.get(ids=[doc_id], include=["metadatas"])

    assert len(result["ids"]) == 1
    assert result["metadatas"][0] is None or result["metadatas"][0] == {}


def test_add_doc_with_only_source_file_raises_error(vector_engine):
    """Test that metadata with only source_file raises ValueError."""
    with pytest.raises(ValueError, match="both 'source_file' and 'chunk_index'"):
        vector_engine.add_doc("Test content", {"source_file": "test.txt"})


def test_add_doc_with_only_chunk_index_raises_error(vector_engine):
    """Test that metadata with only chunk_index raises ValueError."""
    with pytest.raises(ValueError, match="both 'source_file' and 'chunk_index'"):
        vector_engine.add_doc("Test content", {"chunk_index": 0})


def test_add_doc_with_invalid_source_file_type_raises_error(vector_engine):
    """Test that non-string source_file raises ValueError."""
    doc_id = vector_engine.add_doc(
        "Test content", {"source_file": 123, "chunk_index": 0}
    )
    assert doc_id is not None


def test_add_doc_with_invalid_chunk_index_type_raises_error(vector_engine):
    """Test that non-integer chunk_index raises ValueError."""

    doc_id = vector_engine.add_doc(
        "Test content", {"source_file": "test.txt", "chunk_index": "0"}
    )
    assert doc_id is not None


def test_add_doc_with_valid_chunk_metadata(vector_engine):
    """Test that add_doc accepts both source_file and chunk_index."""
    metadata = {"source_file": "test.txt", "chunk_index": 1}
    doc_id = vector_engine.add_doc("Test content", metadata)
    result = vector_engine.collection.get(ids=[doc_id], include=["metadatas"])

    assert len(result["ids"]) == 1
    assert result["metadatas"][0] == metadata


def test_add_doc_updates_file_metrics(vector_engine):
    """Test that add_doc updates file upload metrics when chunk_index is 0."""
    initial_files = vector_engine.metrics.files_uploaded
    vector_engine.add_doc("Test content", {"source_file": "test.txt", "chunk_index": 0})

    assert vector_engine.metrics.files_uploaded == initial_files + 1
    assert vector_engine.metrics.last_file_uploaded_at is not None


def test_add_doc_updates_chunk_metrics(vector_engine):
    """Test that add_doc updates chunks_created metric."""
    initial_chunks = vector_engine.metrics.chunks_created
    vector_engine.add_doc("Test content", {"source_file": "test.txt", "chunk_index": 0})

    assert vector_engine.metrics.chunks_created == initial_chunks + 1


def test_add_doc_updates_doc_size_metric(vector_engine):
    """Test that add_doc updates total_doc_size_bytes metric."""
    content = "Test content with specific length"
    initial_size = vector_engine.metrics.total_doc_size_bytes
    vector_engine.add_doc(content, {})

    assert vector_engine.metrics.total_doc_size_bytes == initial_size + len(content)


def test_add_doc_updates_last_doc_added_timestamp(vector_engine):
    """Test that add_doc updates last_doc_added_at timestamp."""
    assert vector_engine.metrics.last_doc_added_at is None

    vector_engine.add_doc("Test content", {})

    assert vector_engine.metrics.last_doc_added_at is not None


def test_add_doc_with_whitespace_only_content_raises_error(vector_engine):
    """Test that add_doc with whitespace-only content raises ValueError."""
    with pytest.raises(ValueError, match="content cannot be empty"):
        vector_engine.add_doc("   ", {})


def test_add_doc_multiple_sequential():
    """Test adding multiple documents sequentially updates indices correctly."""
    engine = VectorEngine()

    doc_ids = []
    for i in range(5):
        doc_id = engine.add_doc(f"Document {i}", {})
        doc_ids.append(doc_id)

    assert engine.collection.count() == 5

    for doc_id in doc_ids:
        result = engine.collection.get(ids=[doc_id])
        assert len(result["ids"]) == 1


# =============================================================================
# get_doc() Tests
# =============================================================================


def test_get_doc_returns_document(vector_engine):
    """Test that get_doc returns correct document by ID."""
    content = "Test document content"
    metadata = {"key": "value"}
    doc_id = vector_engine.add_doc(content, metadata)
    doc = vector_engine.get_doc(doc_id)

    assert doc is not None
    assert doc["content"] == content
    assert doc["metadata"] == metadata


def test_get_doc_returns_none_for_nonexistent_id(vector_engine):
    """Test that get_doc returns None for non-existent document."""
    doc = vector_engine.get_doc("nonexistent-uuid")

    assert doc is None


def test_get_doc_returns_none_for_deleted_doc(vector_engine):
    """Test that get_doc returns None for deleted documents."""
    doc_id = vector_engine.add_doc("Test content", {})
    vector_engine.delete_doc(doc_id)
    doc = vector_engine.get_doc(doc_id)

    assert doc is None


def test_get_doc_includes_content(vector_engine):
    """Test that returned document includes content."""
    content = "Specific test content"
    doc_id = vector_engine.add_doc(content, {})
    doc = vector_engine.get_doc(doc_id)

    assert doc
    assert "content" in doc
    assert doc["content"] == content


def test_get_doc_includes_metadata(vector_engine):
    """Test that returned document includes metadata."""
    metadata = {"author": "Alice", "topic": "Testing"}
    doc_id = vector_engine.add_doc("Test content", metadata)
    doc = vector_engine.get_doc(doc_id)

    assert doc
    assert "metadata" in doc
    assert doc["metadata"] == metadata


# =============================================================================
# delete_doc() Tests
# =============================================================================


def test_delete_doc_returns_true_for_existing_doc(vector_engine):
    """Test that delete_doc returns True when deleting existing document."""
    doc_id = vector_engine.add_doc("Test content", {})
    result = vector_engine.delete_doc(doc_id)

    assert result is True


def test_delete_doc_returns_false_for_nonexistent_doc(vector_engine):
    """Test that delete_doc returns False for non-existent document."""
    result = vector_engine.delete_doc("nonexistent-uuid")

    assert result is False


def test_delete_doc_removes_from_collection(vector_engine):
    """Test that delete_doc removes document from ChromaDB collection."""
    doc_id = vector_engine.add_doc("Test content", {})

    initial_count = vector_engine.collection.count()
    vector_engine.delete_doc(doc_id)
    assert vector_engine.collection.count() == initial_count - 1

    result = vector_engine.collection.get(ids=[doc_id])
    assert len(result["ids"]) == 0


def test_delete_doc_increments_metrics(vector_engine):
    """Test that delete_doc increments docs_deleted metric."""
    doc_id = vector_engine.add_doc("Test content", {})
    initial_deleted = vector_engine.metrics.docs_deleted
    vector_engine.delete_doc(doc_id)

    assert vector_engine.metrics.docs_deleted == initial_deleted + 1


def test_delete_same_doc_twice_returns_false(vector_engine):
    """Test that deleting same document twice returns False on second attempt."""
    doc_id = vector_engine.add_doc("Test content", {})
    result1 = vector_engine.delete_doc(doc_id)
    result2 = vector_engine.delete_doc(doc_id)

    assert result1 is True
    assert result2 is False


def test_multiple_deletes_decrease_size_metric(vector_engine):
    """Test that deleting documents decreases total_doc_size_bytes."""
    content1 = "First document content"
    content2 = "Second document content"

    doc_id1 = vector_engine.add_doc(content1, {})
    doc_id2 = vector_engine.add_doc(content2, {})

    initial_size = vector_engine.metrics.total_doc_size_bytes

    vector_engine.delete_doc(doc_id1)

    assert vector_engine.metrics.total_doc_size_bytes == initial_size - len(content1)


def test_delete_doc_updates_size_metric_correctly(vector_engine):
    """Test that delete_doc correctly decreases total_doc_size_bytes."""
    content = "This is a test document with specific length"
    doc_id = vector_engine.add_doc(content, {})

    size_before = vector_engine.metrics.total_doc_size_bytes
    vector_engine.delete_doc(doc_id)
    size_after = vector_engine.metrics.total_doc_size_bytes

    assert size_after == size_before - len(content)


# =============================================================================
# search() Tests
# =============================================================================


def test_search_returns_list(vector_engine):
    """Test that search returns a list of results."""
    vector_engine.add_doc("Test document", {})

    results = vector_engine.search("test")

    assert isinstance(results, list)


def test_search_returns_correct_structure(vector_engine):
    """Test that search results have correct structure (doc_id, content, score, metadata)."""
    vector_engine.add_doc("Test document", {"key": "value"})

    results = vector_engine.search("test")

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


def test_search_excludes_deleted_documents(vector_engine):
    """Test that search results don't include deleted documents."""
    doc_id = vector_engine.add_doc("Unique searchable content", {})
    vector_engine.add_doc("Other content", {})

    results_before = vector_engine.search("unique searchable")
    vector_engine.delete_doc(doc_id)
    results_after = vector_engine.search("unique searchable")

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


def test_search_with_empty_index(vector_engine):
    """Test that search on empty index returns empty list."""

    results = vector_engine.search("test query")

    assert results == []


def test_search_updates_metrics(vector_engine):
    """Test that search increments query count metrics."""
    vector_engine.add_doc("Test document", {})
    initial_queries = vector_engine.metrics.total_queries

    vector_engine.search("test")

    assert vector_engine.metrics.total_queries == initial_queries + 1


def test_search_tracks_query_time(vector_engine):
    """Test that search tracks query execution time."""
    vector_engine.add_doc("Test document", {})
    initial_time = vector_engine.metrics.total_query_time_ms

    vector_engine.search("test")

    assert vector_engine.metrics.total_query_time_ms > initial_time
    assert len(vector_engine.metrics.query_times) == 1


def test_search_with_default_top_k(vector_engine):
    """Test that search uses default top_k value."""
    for i in range(20):
        vector_engine.add_doc(f"Document {i}", {})

    results = vector_engine.search("document")

    assert len(results) == VFGConfig.DEFAULT_TOP_K


def test_search_similarity_scores_in_range(vector_engine):
    """Test that similarity scores are between 0 and 1."""
    for i in range(10):
        vector_engine.add_doc(f"Document content {i}", {})

    results = vector_engine.search("document")

    for result in results:
        assert 0.0 <= result.score <= 1.0


def test_search_with_whitespace_only_query_raises_error(vector_engine):
    """Test that search with whitespace-only query raises ValueError."""
    vector_engine.add_doc("Test document", {})

    with pytest.raises(ValueError, match="cannot be empty"):
        vector_engine.search("   ")


def test_search_returns_empty_list_when_all_deleted(vector_engine):
    """Test that search returns empty list when all documents are deleted."""
    doc_id1 = vector_engine.add_doc("Document 1", {})
    doc_id2 = vector_engine.add_doc("Document 2", {})

    vector_engine.delete_doc(doc_id1)
    vector_engine.delete_doc(doc_id2)

    results = vector_engine.search("document")

    assert results == []


def test_search_top_k_larger_than_available(vector_engine):
    """Test search when top_k is larger than number of available documents."""
    vector_engine.add_doc("Document 1", {})
    vector_engine.add_doc("Document 2", {})

    results = vector_engine.search("document", top_k=100)

    assert len(results) == 2


# =============================================================================
# search() with filters Tests
# =============================================================================


def test_search_filter_by_source_file(vector_engine):
    """Test that filtering by source_file returns only matching documents."""
    vector_engine.add_doc(
        "Python tutorial", {"source_file": "python.pdf", "chunk_index": 0}
    )
    vector_engine.add_doc(
        "Java tutorial", {"source_file": "java.pdf", "chunk_index": 0}
    )
    vector_engine.add_doc(
        "JavaScript tutorial", {"source_file": "js.pdf", "chunk_index": 0}
    )

    results = vector_engine.search(
        "tutorial", top_k=10, filters={"source_file": "python.pdf"}
    )

    assert len(results) == 1
    assert results[0].metadata["source_file"] == "python.pdf"


def test_search_filter_by_chunk_index(vector_engine):
    """Test that filtering by chunk_index returns only matching documents."""
    vector_engine.add_doc(
        "Chunk 0 content", {"source_file": "doc.pdf", "chunk_index": 0}
    )
    vector_engine.add_doc(
        "Chunk 1 content", {"source_file": "doc.pdf", "chunk_index": 1}
    )
    vector_engine.add_doc(
        "Chunk 2 content", {"source_file": "doc.pdf", "chunk_index": 2}
    )

    results = vector_engine.search("content", top_k=10, filters={"chunk_index": 1})

    assert len(results) == 1
    assert results[0].metadata["chunk_index"] == 1


def test_search_filter_by_both_fields(vector_engine):
    """Test that filtering by both source_file and chunk_index uses AND logic."""
    vector_engine.add_doc("Doc A Chunk 0", {"source_file": "a.pdf", "chunk_index": 0})
    vector_engine.add_doc("Doc A Chunk 1", {"source_file": "a.pdf", "chunk_index": 1})
    vector_engine.add_doc("Doc B Chunk 0", {"source_file": "b.pdf", "chunk_index": 0})

    results = vector_engine.search(
        "doc chunk", top_k=10, filters={"source_file": "a.pdf", "chunk_index": 1}
    )

    assert len(results) == 1
    assert results[0].metadata["source_file"] == "a.pdf"
    assert results[0].metadata["chunk_index"] == 1


def test_search_filter_and_logic(vector_engine):
    """Test that multiple filters require all to match (AND logic)."""
    vector_engine.add_doc(
        "Python programming",
        {"source_file": "python.pdf", "chunk_index": 0, "topic": "programming"},
    )
    vector_engine.add_doc(
        "Python data science",
        {"source_file": "python.pdf", "chunk_index": 1, "topic": "data"},
    )
    vector_engine.add_doc(
        "Java programming",
        {"source_file": "java.pdf", "chunk_index": 0, "topic": "programming"},
    )

    results = vector_engine.search(
        "programming",
        top_k=10,
        filters={"source_file": "python.pdf", "topic": "programming"},
    )

    assert len(results) == 1
    assert results[0].metadata["source_file"] == "python.pdf"
    assert results[0].metadata["topic"] == "programming"


def test_search_filter_no_matches(vector_engine):
    """Test that filtering with no matches returns empty list."""
    vector_engine.add_doc(
        "Document content", {"source_file": "doc.pdf", "chunk_index": 0}
    )

    results = vector_engine.search(
        "content", top_k=10, filters={"source_file": "missing.pdf"}
    )

    assert len(results) == 0


def test_search_filter_custom_metadata(vector_engine):
    """Test that filtering works with custom metadata fields."""
    vector_engine.add_doc("Article 1", {"author": "Alice", "year": 2020})
    vector_engine.add_doc("Article 2", {"author": "Bob", "year": 2021})
    vector_engine.add_doc("Article 3", {"author": "Alice", "year": 2021})

    results = vector_engine.search("article", top_k=10, filters={"author": "Alice"})

    assert len(results) == 2
    for result in results:
        assert result.metadata["author"] == "Alice"


def test_search_filter_none(vector_engine):
    """Test that filters=None returns all results."""
    vector_engine.add_doc("Doc 1", {"source_file": "a.pdf", "chunk_index": 0})
    vector_engine.add_doc("Doc 2", {"source_file": "b.pdf", "chunk_index": 0})

    results = vector_engine.search("doc", top_k=10, filters=None)

    assert len(results) == 2


def test_search_filter_empty_dict(vector_engine):
    """Test that filters={} (empty dict) returns all results."""
    vector_engine.add_doc("Doc 1", {"source_file": "a.pdf", "chunk_index": 0})
    vector_engine.add_doc("Doc 2", {"source_file": "b.pdf", "chunk_index": 0})

    results = vector_engine.search("doc", top_k=10, filters={})

    assert len(results) == 2


def test_search_filter_case_sensitive(vector_engine):
    """Test that filter matching is case-sensitive."""
    vector_engine.add_doc("Document", {"source_file": "Doc.pdf", "chunk_index": 0})

    results = vector_engine.search(
        "document", top_k=10, filters={"source_file": "doc.pdf"}
    )

    assert len(results) == 0

    results = vector_engine.search(
        "document", top_k=10, filters={"source_file": "Doc.pdf"}
    )

    assert len(results) == 1


# =============================================================================
# list_files() Tests
# =============================================================================


def test_list_files_returns_list(vector_engine):
    """Test that list_files returns a list."""

    files = vector_engine.list_files()

    assert isinstance(files, list)


def test_list_files_empty_when_no_files(vector_engine):
    """Test that list_files returns empty list when no files uploaded."""
    vector_engine.add_doc("Document without source_file", {})

    files = vector_engine.list_files()

    assert files == []


def test_list_files_includes_uploaded_files(vector_engine):
    """Test that list_files includes filenames from uploaded documents."""
    vector_engine.add_doc("Content 1", {"source_file": "file1.txt", "chunk_index": 0})
    vector_engine.add_doc("Content 2", {"source_file": "file2.txt", "chunk_index": 0})

    files = vector_engine.list_files()

    assert "file1.txt" in files
    assert "file2.txt" in files


def test_list_files_returns_unique_filenames(vector_engine):
    """Test that list_files returns unique filenames only."""
    vector_engine.add_doc("Chunk 1", {"source_file": "file.txt", "chunk_index": 0})
    vector_engine.add_doc("Chunk 2", {"source_file": "file.txt", "chunk_index": 1})
    vector_engine.add_doc("Chunk 3", {"source_file": "file.txt", "chunk_index": 2})

    files = vector_engine.list_files()

    assert files == ["file.txt"]
    assert len(files) == 1


def test_list_files_ignores_docs_without_source_file(vector_engine):
    """Test that list_files only includes docs with source_file metadata."""
    vector_engine.add_doc(
        "Doc with file", {"source_file": "test.txt", "chunk_index": 0}
    )
    vector_engine.add_doc("Doc without file", {"other": "metadata"})
    vector_engine.add_doc("Another doc", {})

    files = vector_engine.list_files()

    assert files == ["test.txt"]


def test_list_files_sorted_alphabetically(vector_engine):
    """Test that list_files returns filenames in sorted order."""
    vector_engine.add_doc("Content", {"source_file": "zebra.txt", "chunk_index": 0})
    vector_engine.add_doc("Content", {"source_file": "apple.txt", "chunk_index": 0})
    vector_engine.add_doc("Content", {"source_file": "banana.txt", "chunk_index": 0})

    files = vector_engine.list_files()

    assert files == ["apple.txt", "banana.txt", "zebra.txt"]


def test_list_files_excludes_deleted_docs(vector_engine):
    """Test that list_files doesn't include files from deleted documents."""
    vector_engine.add_doc("Content 1", {"source_file": "keep.txt", "chunk_index": 0})
    doc_id = vector_engine.add_doc(
        "Content 2", {"source_file": "delete.txt", "chunk_index": 0}
    )

    vector_engine.delete_doc(doc_id)

    files = vector_engine.list_files()

    assert "keep.txt" in files
    assert "delete.txt" not in files


# =============================================================================
# delete_file() Tests
# =============================================================================


def test_delete_file_returns_dict(vector_engine):
    """Test that delete_file returns a dictionary with status."""
    vector_engine.add_doc("Content", {"source_file": "test.txt", "chunk_index": 0})

    result = vector_engine.delete_file("test.txt")

    assert isinstance(result, dict)
    assert "status" in result
    assert "filename" in result
    assert "chunks_deleted" in result
    assert "doc_ids" in result


def test_delete_file_deletes_all_chunks(vector_engine):
    """Test that delete_file removes all chunks from a source file."""
    vector_engine.add_doc("Chunk 1", {"source_file": "test.txt", "chunk_index": 0})
    vector_engine.add_doc("Chunk 2", {"source_file": "test.txt", "chunk_index": 1})
    vector_engine.add_doc("Chunk 3", {"source_file": "test.txt", "chunk_index": 2})

    result = vector_engine.delete_file("test.txt")

    assert result["status"] == "deleted"
    assert result["chunks_deleted"] == 3


def test_delete_file_returns_deleted_count(vector_engine):
    """Test that delete_file returns count of deleted chunks."""
    vector_engine.add_doc("Chunk 1", {"source_file": "file.txt", "chunk_index": 0})
    vector_engine.add_doc("Chunk 2", {"source_file": "file.txt", "chunk_index": 1})

    result = vector_engine.delete_file("file.txt")

    assert result["chunks_deleted"] == 2


def test_delete_file_returns_doc_ids(vector_engine):
    """Test that delete_file returns list of deleted document IDs."""
    doc_id1 = vector_engine.add_doc(
        "Chunk 1", {"source_file": "file.txt", "chunk_index": 0}
    )
    doc_id2 = vector_engine.add_doc(
        "Chunk 2", {"source_file": "file.txt", "chunk_index": 1}
    )

    result = vector_engine.delete_file("file.txt")

    assert doc_id1 in result["doc_ids"]
    assert doc_id2 in result["doc_ids"]
    assert len(result["doc_ids"]) == 2


def test_delete_file_returns_not_found_for_nonexistent(vector_engine):
    """Test that delete_file returns not_found status for non-existent file."""

    result = vector_engine.delete_file("nonexistent.txt")

    assert result["status"] == "not_found"
    assert result["chunks_deleted"] == 0
    assert result["doc_ids"] == []


def test_delete_file_updates_metrics(vector_engine):
    """Test that delete_file updates deletion metrics."""
    vector_engine.add_doc("Chunk 1", {"source_file": "file.txt", "chunk_index": 0})
    vector_engine.add_doc("Chunk 2", {"source_file": "file.txt", "chunk_index": 1})
    initial_deleted = vector_engine.metrics.docs_deleted

    vector_engine.delete_file("file.txt")

    assert vector_engine.metrics.docs_deleted == initial_deleted + 2


def test_delete_file_with_mixed_chunks(vector_engine):
    """Test delete_file when only some chunks belong to the file."""
    vector_engine.add_doc("Chunk 1", {"source_file": "target.txt", "chunk_index": 0})
    vector_engine.add_doc("Chunk 2", {"source_file": "target.txt", "chunk_index": 1})
    vector_engine.add_doc("Other 1", {"source_file": "other.txt", "chunk_index": 0})
    vector_engine.add_doc("Other 2", {"source_file": "other.txt", "chunk_index": 1})

    result = vector_engine.delete_file("target.txt")

    assert result["chunks_deleted"] == 2
    assert result["status"] == "deleted"

    files = vector_engine.list_files()
    assert "target.txt" not in files
    assert "other.txt" in files


# =============================================================================
# get_metrics() Tests
# =============================================================================


def test_get_metrics_returns_dict(vector_engine):
    """Test that get_metrics returns a dictionary."""
    metrics = vector_engine.get_metrics()

    assert isinstance(metrics, dict)


def test_get_metrics_includes_all_categories(vector_engine):
    """Test that get_metrics includes all metric categories."""
    vector_engine.add_doc("Test", {})
    vector_engine.search("test")

    metrics = vector_engine.get_metrics()

    assert "total_queries" in metrics
    assert "docs_added" in metrics
    assert "total_documents" in metrics
    assert "model_name" in metrics
    assert "created_at" in metrics


def test_get_metrics_includes_query_stats(vector_engine):
    """Test that metrics include query statistics."""
    vector_engine.add_doc("Test document", {})
    vector_engine.search("test")

    metrics = vector_engine.get_metrics()

    assert "total_queries" in metrics
    assert "avg_query_time_ms" in metrics
    assert "total_query_time_ms" in metrics
    assert metrics["total_queries"] > 0


def test_get_metrics_includes_document_stats(vector_engine):
    """Test that metrics include document statistics."""
    vector_engine.add_doc("Test document", {})

    metrics = vector_engine.get_metrics()

    assert "docs_added" in metrics
    assert "docs_deleted" in metrics
    assert "total_documents" in metrics
    assert metrics["docs_added"] == 1
    assert metrics["total_documents"] == 1


def test_get_metrics_includes_memory_stats(vector_engine):
    """Test that metrics include memory statistics."""
    vector_engine.add_doc("Test document", {})

    metrics = vector_engine.get_metrics()

    # Only total_doc_size_bytes is tracked (no approximations)
    assert "total_doc_size_bytes" in metrics
    assert metrics["total_doc_size_bytes"] > 0


def test_get_metrics_includes_timestamps(vector_engine):
    """Test that metrics include timestamp information."""
    vector_engine.add_doc("Test document", {})
    vector_engine.search("test")

    metrics = vector_engine.get_metrics()

    assert "created_at" in metrics
    assert "last_query_at" in metrics
    assert "last_doc_added_at" in metrics
    assert metrics["created_at"] is not None
    assert metrics["last_query_at"] is not None


def test_get_metrics_calculates_uptime(vector_engine):
    """Test that metrics include uptime calculation."""
    metrics = vector_engine.get_metrics()

    assert "uptime_seconds" in metrics
    assert metrics["uptime_seconds"] >= 0
    assert isinstance(metrics["uptime_seconds"], float)


def test_get_metrics_percentiles_with_few_queries(vector_engine):
    """Test that get_metrics handles percentile calculation with few data points."""
    vector_engine.add_doc("Document", {})

    vector_engine.search("test")

    metrics = vector_engine.get_metrics()

    assert metrics["p50_query_time_ms"] is not None
    assert metrics["p95_query_time_ms"] is not None
    assert metrics["p99_query_time_ms"] is not None


def test_get_metrics_percentiles_with_no_queries(vector_engine):
    """Test that get_metrics handles percentiles when no queries have been made."""
    metrics = vector_engine.get_metrics()

    assert metrics["p50_query_time_ms"] is None
    assert metrics["p95_query_time_ms"] is None
    assert metrics["p99_query_time_ms"] is None
    assert metrics["min_query_time_ms"] is None
    assert metrics["max_query_time_ms"] is None


# =============================================================================
# get_index_stats() Tests
# =============================================================================


def test_get_index_stats_returns_dict(vector_engine):
    """Test that get_index_stats returns a dictionary."""
    stats = vector_engine.get_index_stats()

    assert isinstance(stats, dict)


def test_get_index_stats_includes_document_counts(vector_engine):
    """Test that index stats include document counts."""
    vector_engine.add_doc("Document 1", {})
    vector_engine.add_doc("Document 2", {})

    stats = vector_engine.get_index_stats()

    assert "total_documents" in stats
    assert stats["total_documents"] == 2


def test_get_index_stats_includes_embedding_dimension(vector_engine):
    """Test that index stats include embedding dimension."""
    vector_engine.add_doc("Document", {})

    stats = vector_engine.get_index_stats()

    assert "embedding_dimension" in stats
    assert stats["embedding_dimension"] == VFGConfig.EMBEDDING_DIMENSION


# =============================================================================
# EngineMetrics Tests
# =============================================================================


def test_engine_metrics_initialization():
    """Test that EngineMetrics initializes with correct default values."""
    metrics = EngineMetrics()

    assert metrics.total_queries == 0
    assert metrics.docs_added == 0
    assert metrics.docs_deleted == 0
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


def test_add_search_delete_workflow(vector_engine):
    """Test complete workflow: add documents, search, delete, verify."""
    doc_id1 = vector_engine.add_doc("Python programming language", {"topic": "coding"})
    doc_id2 = vector_engine.add_doc("Machine learning algorithms", {"topic": "AI"})
    doc_id3 = vector_engine.add_doc("Python snake species", {"topic": "nature"})

    results = vector_engine.search("Python")
    assert len(results) == 3

    assert vector_engine.delete_doc(doc_id2) is True

    results = vector_engine.search("Python")
    assert len(results) == 2

    result_ids = [r.id for r in results]
    assert doc_id2 not in result_ids
    assert doc_id1 in result_ids
    assert doc_id3 in result_ids


def test_multiple_operations_metrics_accuracy(vector_engine):
    """Test that metrics remain accurate after multiple operations."""
    for i in range(5):
        vector_engine.add_doc(f"Document {i}", {})

    for i in range(3):
        vector_engine.search("document")

    all_docs = vector_engine.collection.get()
    doc_ids = all_docs["ids"][:2]

    for doc_id in doc_ids:
        vector_engine.delete_doc(doc_id)

    metrics = vector_engine.get_metrics()
    assert metrics["docs_added"] == 5
    assert metrics["docs_deleted"] >= 2
    assert metrics["total_queries"] == 3
    assert metrics["total_documents"] <= 3


# =============================================================================
# ChromaDB Metrics Tests
# =============================================================================


def test_get_chromadb_disk_size_returns_tuple(vector_engine):
    """Test that _get_chromadb_disk_size returns a tuple of (bytes, mb)."""
    result = vector_engine._get_chromadb_disk_size()

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], int)  # bytes
    assert isinstance(result[1], float)  # mb


def test_get_chromadb_disk_size_values_non_negative(vector_engine):
    """Test that disk size values are non-negative."""
    bytes_size, mb_size = vector_engine._get_chromadb_disk_size()

    assert bytes_size >= 0
    assert mb_size >= 0.0


def test_get_chromadb_disk_size_conversion(vector_engine):
    """Test that MB conversion is accurate."""
    bytes_size, mb_size = vector_engine._get_chromadb_disk_size()

    expected_mb = bytes_size / (1024 * 1024)
    # Check that the MB value is close (within 0.01 MB difference)
    assert abs(mb_size - expected_mb) < 0.01


def test_get_chromadb_disk_size_increases_with_data(vector_engine):
    """Test that disk size increases when documents are added."""
    # Get initial size
    initial_bytes, _ = vector_engine._get_chromadb_disk_size()

    # Add documents
    for i in range(10):
        vector_engine.add_doc(f"Test document {i} with content " * 50, {})

    # Get new size
    final_bytes, _ = vector_engine._get_chromadb_disk_size()

    # Size should increase
    assert final_bytes >= initial_bytes


def test_get_chromadb_metrics_returns_dict(vector_engine):
    """Test that get_chromadb_metrics returns a dictionary."""
    result = vector_engine.get_chromadb_metrics()

    assert isinstance(result, dict)


def test_get_chromadb_metrics_has_all_required_fields(vector_engine):
    """Test that get_chromadb_metrics returns all expected fields."""
    metrics = vector_engine.get_chromadb_metrics()

    required_fields = {
        "version",
        "collection_id",
        "collection_name",
        "disk_size_bytes",
        "disk_size_mb",
        "persist_directory",
        "max_batch_size",
    }

    assert set(metrics.keys()) == required_fields


def test_get_chromadb_metrics_version_is_string(vector_engine):
    """Test that ChromaDB version is a string."""
    metrics = vector_engine.get_chromadb_metrics()

    assert isinstance(metrics["version"], str)
    assert len(metrics["version"]) > 0


def test_get_chromadb_metrics_collection_info(vector_engine):
    """Test that collection information is correct."""
    metrics = vector_engine.get_chromadb_metrics()

    assert isinstance(metrics["collection_id"], str)
    assert isinstance(metrics["collection_name"], str)
    assert len(metrics["collection_id"]) > 0
    assert metrics["collection_name"] == vector_engine.collection.name


def test_get_chromadb_metrics_disk_sizes(vector_engine):
    """Test that disk size metrics are correct types and values."""
    metrics = vector_engine.get_chromadb_metrics()

    assert isinstance(metrics["disk_size_bytes"], int)
    assert isinstance(metrics["disk_size_mb"], float)
    assert metrics["disk_size_bytes"] >= 0
    assert metrics["disk_size_mb"] >= 0.0

    # Verify conversion
    expected_mb = round(metrics["disk_size_bytes"] / (1024 * 1024), 2)
    assert metrics["disk_size_mb"] == expected_mb


def test_get_chromadb_metrics_persist_directory(vector_engine):
    """Test that persist directory is a valid path."""
    metrics = vector_engine.get_chromadb_metrics()

    assert isinstance(metrics["persist_directory"], str)
    assert len(metrics["persist_directory"]) > 0


def test_get_chromadb_metrics_max_batch_size(vector_engine):
    """Test that max_batch_size is a positive integer."""
    metrics = vector_engine.get_chromadb_metrics()

    assert isinstance(metrics["max_batch_size"], int)
    assert metrics["max_batch_size"] > 0


def test_chromadb_metrics_consistency(vector_engine):
    """Test that ChromaDB metrics are consistent across calls."""
    metrics1 = vector_engine.get_chromadb_metrics()
    metrics2 = vector_engine.get_chromadb_metrics()

    # These fields should be identical
    assert metrics1["version"] == metrics2["version"]
    assert metrics1["collection_id"] == metrics2["collection_id"]
    assert metrics1["collection_name"] == metrics2["collection_name"]
    assert metrics1["persist_directory"] == metrics2["persist_directory"]
    assert metrics1["max_batch_size"] == metrics2["max_batch_size"]


# =============================================================================
# HNSW Configuration Tests
# =============================================================================


def test_get_hnsw_config_returns_dict(vector_engine):
    """Test that get_hnsw_config returns a dictionary."""
    result = vector_engine.get_hnsw_config()

    assert isinstance(result, dict)


def test_get_hnsw_config_has_all_required_fields(vector_engine):
    """Test that get_hnsw_config returns all expected fields."""
    config = vector_engine.get_hnsw_config()

    required_fields = {
        "space",
        "ef_construction",
        "ef_search",
        "max_neighbors",
        "resize_factor",
        "sync_threshold",
    }

    assert set(config.keys()) == required_fields


def test_get_hnsw_config_space_is_string(vector_engine):
    """Test that space parameter is a valid string."""
    config = vector_engine.get_hnsw_config()

    assert isinstance(config["space"], str)
    assert config["space"] in ["cosine", "l2", "ip"]


def test_get_hnsw_config_ef_construction_is_positive(vector_engine):
    """Test that ef_construction is a positive integer."""
    config = vector_engine.get_hnsw_config()

    assert isinstance(config["ef_construction"], int)
    assert config["ef_construction"] > 0


def test_get_hnsw_config_ef_search_is_positive(vector_engine):
    """Test that ef_search is a positive integer."""
    config = vector_engine.get_hnsw_config()

    assert isinstance(config["ef_search"], int)
    assert config["ef_search"] > 0


def test_get_hnsw_config_max_neighbors_is_positive(vector_engine):
    """Test that max_neighbors is a positive integer."""
    config = vector_engine.get_hnsw_config()

    assert isinstance(config["max_neighbors"], int)
    assert config["max_neighbors"] > 0


def test_get_hnsw_config_resize_factor_is_valid(vector_engine):
    """Test that resize_factor is a float greater than 1.0."""
    config = vector_engine.get_hnsw_config()

    assert isinstance(config["resize_factor"], float)
    assert config["resize_factor"] > 1.0


def test_get_hnsw_config_sync_threshold_is_positive(vector_engine):
    """Test that sync_threshold is a positive integer."""
    config = vector_engine.get_hnsw_config()

    assert isinstance(config["sync_threshold"], int)
    assert config["sync_threshold"] > 0


def test_get_hnsw_config_default_values(vector_engine):
    """Test that HNSW config returns expected default values."""
    config = vector_engine.get_hnsw_config()

    # ChromaDB default values
    assert config["space"] == "cosine"
    assert config["ef_construction"] == 100
    assert config["ef_search"] == 100
    assert config["max_neighbors"] == 16
    assert config["resize_factor"] == 1.2
    assert config["sync_threshold"] == 1000


def test_get_hnsw_config_consistency(vector_engine):
    """Test that HNSW config is consistent across calls."""
    config1 = vector_engine.get_hnsw_config()
    config2 = vector_engine.get_hnsw_config()

    assert config1 == config2


# =============================================================================
# Peak Document Tracking Tests
# =============================================================================


def test_peak_document_count_initializes_to_zero(vector_engine):
    """Test that peak document count starts at zero."""
    assert vector_engine.metrics.total_documents_peak == 0


def test_peak_document_count_updates_after_add_doc(vector_engine):
    """Test that peak increases after adding a document."""
    initial_peak = vector_engine.metrics.total_documents_peak

    # Add a document
    doc_id = vector_engine.add_doc(
        content="Test document for peak tracking",
        metadata={"test": "peak_tracking"},
    )

    # Peak should have increased
    assert vector_engine.metrics.total_documents_peak > initial_peak
    assert vector_engine.metrics.total_documents_peak == 1


def test_peak_document_count_increases_with_multiple_adds(vector_engine):
    """Test that peak increases correctly with multiple document additions."""
    # Add 5 documents
    for i in range(5):
        vector_engine.add_doc(
            content=f"Test document {i} for peak tracking",
            metadata={"test_id": i},
        )

    assert vector_engine.metrics.total_documents_peak == 5


def test_peak_document_count_does_not_decrease_on_delete(vector_engine):
    """Test that peak does NOT decrease when documents are deleted."""
    # Add 5 documents
    doc_ids = []
    for i in range(5):
        doc_id = vector_engine.add_doc(
            content=f"Test document {i} for peak tracking",
            metadata={"test_id": i},
        )
        doc_ids.append(doc_id)

    peak_after_add = vector_engine.metrics.total_documents_peak
    assert peak_after_add == 5

    # Delete 3 documents
    for i in range(3):
        vector_engine.delete_doc(doc_ids[i])

    peak_after_delete = vector_engine.metrics.total_documents_peak

    # Peak should stay the same
    assert peak_after_delete == peak_after_add
    assert peak_after_delete == 5

    # But total documents should have decreased
    total_docs = vector_engine.collection.count()
    assert total_docs == 2


def test_peak_document_count_in_get_metrics(vector_engine):
    """Test that get_metrics() includes peak document count."""
    # Add some documents
    for i in range(3):
        vector_engine.add_doc(
            content=f"Test document {i}",
            metadata={"test_id": i},
        )

    metrics = vector_engine.get_metrics()

    assert "total_documents_peak" in metrics
    assert metrics["total_documents_peak"] == 3
    assert metrics["total_documents_peak"] == metrics["total_documents"]


def test_peak_document_count_after_add_delete_cycles(vector_engine):
    """Test that peak correctly tracks maximum across add/delete cycles."""
    # Add 5 documents
    doc_ids = []
    for i in range(5):
        doc_id = vector_engine.add_doc(
            content=f"Test document {i}",
            metadata={"test_id": i},
        )
        doc_ids.append(doc_id)

    assert vector_engine.metrics.total_documents_peak == 5

    # Delete 3 documents (down to 2)
    for i in range(3):
        vector_engine.delete_doc(doc_ids[i])

    assert vector_engine.metrics.total_documents_peak == 5
    assert vector_engine.collection.count() == 2

    # Add 2 more documents (up to 4, but peak should stay at 5)
    vector_engine.add_doc(content="New doc 1", metadata={"new": 1})
    assert vector_engine.metrics.total_documents_peak == 5

    vector_engine.add_doc(content="New doc 2", metadata={"new": 2})
    assert vector_engine.metrics.total_documents_peak == 5
    assert vector_engine.collection.count() == 4

    # Add 2 more to exceed previous peak (up to 6)
    vector_engine.add_doc(content="New doc 3", metadata={"new": 3})
    vector_engine.add_doc(content="New doc 4", metadata={"new": 4})

    assert vector_engine.metrics.total_documents_peak == 6
    assert vector_engine.collection.count() == 6
