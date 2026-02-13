"""Tests for Python/C++ integration"""

import numpy as np
import pytest

from vectorforge.vectorforge_cpp import cosine_similarity_batch

# =============================================================================
# CPP Integration Tests
# =============================================================================


def test_cosine_similarity_batch_identical_vectors(client):
    """Test cosine_similarity_batch with identical vectors."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result = cosine_similarity_batch(query, docs)

    assert result.shape == (1,)
    assert np.isclose(result[0], 1.0)


def test_cosine_similarity_batch_orthogonal_vectors(client):
    """Test cosine_similarity_batch with orthogonal vectors."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    result = cosine_similarity_batch(query, docs)

    assert result.shape == (1,)
    assert np.isclose(result[0], 0.0, atol=1e-6)


def test_cosine_similarity_batch_multiple_documents(client):
    """Test cosine_similarity_batch with multiple documents."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32
    )
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    result = cosine_similarity_batch(query, docs)

    assert result.shape == (3,)
    assert np.isclose(result[0], 1.0)
    assert np.isclose(result[1], 0.0, atol=1e-6)
    assert np.isclose(result[2], 0.707, atol=0.01)


def test_cosine_similarity_batch_realistic_embeddings(client):
    """Test cosine_similarity_batch with realistic embedding dimensions."""
    np.random.seed(42)
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)
    docs = np.random.randn(10, 384).astype(np.float32)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    result = cosine_similarity_batch(query, docs)

    assert result.shape == (10,)
    assert result.dtype == np.float32
    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_cosine_similarity_batch_numerical_equivalence(client):
    """Test that C++ results match Python implementation."""
    np.random.seed(123)
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)
    docs = np.random.randn(100, 384).astype(np.float32)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    cpp_result = cosine_similarity_batch(query, docs)

    py_result = []
    for i in range(docs.shape[0]):
        score = float(np.dot(query, docs[i]))
        py_result.append(score)
    py_result = np.array(py_result, dtype=np.float32)

    assert np.allclose(cpp_result, py_result, rtol=1e-5, atol=1e-7)


def test_cosine_similarity_batch_invalid_query_dimensions(client):
    """Test error handling for invalid query dimensions."""
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    docs = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="Invalid array dimensions"):
        cosine_similarity_batch(query, docs)


def test_cosine_similarity_batch_invalid_docs_dimensions(client):
    """Test error handling for invalid document dimensions."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    docs = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="Invalid array dimensions"):
        cosine_similarity_batch(query, docs)


def test_cosine_similarity_batch_dimension_mismatch(client):
    """Test error handling for dimension mismatch between query and documents."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    docs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="Dimension mismatch"):
        cosine_similarity_batch(query, docs)
