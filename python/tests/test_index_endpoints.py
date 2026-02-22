"""Tests for index management endpoints"""

import os

import pytest

from vectorforge.config import VFGConfig

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


# =============================================================================
# Index Test Fixtures
# =============================================================================


@pytest.fixture
def stats(client):
    """Reusable stats index data fixture"""
    resp = client.get("/index/stats")
    assert resp.status_code == 200
    return resp.json()


# =============================================================================
# Index Stats Tests
# =============================================================================


def test_index_stats_returns_200(client):
    """Test that GET /index/stats returns 200 status."""
    resp = client.get("/index/stats")
    assert resp.status_code == 200


def test_index_stats_returns_status(stats):
    """Test that index stats includes status."""
    assert "status" in stats
    assert isinstance(stats["status"], str)


def test_index_stats_returns_total_documents(stats):
    """Test that index stats includes total documents count."""
    assert "total_documents" in stats
    assert isinstance(stats["total_documents"], int)


def test_index_stats_returns_embedding_dimension(stats):
    """Test that index stats includes embedding dimension."""
    assert "embedding_dimension" in stats
    assert isinstance(stats["embedding_dimension"], int)


def test_index_stats_with_empty_index(client):
    """Test index stats when index is empty."""
    stats = client.get("/index/stats").json()

    assert stats["total_documents"] == 0
    assert stats["embedding_dimension"] == VFGConfig.EMBEDDING_DIMENSION


def test_index_stats_after_adding_documents(client):
    """Test that stats update correctly after adding documents."""
    client.post("/doc/add", json={"content": "test doc 1", "metadata": {}})
    client.post("/doc/add", json={"content": "test doc 2", "metadata": {}})

    stats = client.get("/index/stats").json()
    expected_total_docs = 2
    assert stats["total_documents"] == expected_total_docs


def test_index_stats_after_document_deletion(client, multiple_added_docs):
    """Test that stats update correctly after document deletion."""
    doc_id = multiple_added_docs[0]
    client.delete(f"/doc/{doc_id}")

    stats = client.get("/index/stats").json()
    expected_total_docs = 19
    assert stats["total_documents"] == expected_total_docs


def test_index_stats_embedding_dimension_is_384(client):
    """Test that embedding_dimension matches configured model dimension."""
    stats = client.get("/index/stats").json()
    assert stats["embedding_dimension"] == VFGConfig.EMBEDDING_DIMENSION


def test_index_stats_multiple_deletions_immediate_removal(client, multiple_added_docs):
    """Test that multiple deletions are immediately reflected in stats."""
    for i in range(4):
        client.delete(f"/doc/{multiple_added_docs[i]}")

    stats = client.get("/index/stats").json()

    expected_total_docs = 16
    assert stats["total_documents"] == expected_total_docs
