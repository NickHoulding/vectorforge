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


# =============================================================================
# HNSW Configuration Tests
# =============================================================================


def test_index_stats_includes_hnsw_config(stats):
    """Test that index stats includes HNSW configuration."""
    assert "hnsw_config" in stats
    assert isinstance(stats["hnsw_config"], dict)


def test_hnsw_config_has_space(stats):
    """Test that HNSW config includes distance metric."""
    hnsw = stats["hnsw_config"]
    assert "space" in hnsw
    assert isinstance(hnsw["space"], str)
    assert hnsw["space"] in ["cosine", "l2", "ip"]


def test_hnsw_config_has_ef_construction(stats):
    """Test that HNSW config includes ef_construction parameter."""
    hnsw = stats["hnsw_config"]
    assert "ef_construction" in hnsw
    assert isinstance(hnsw["ef_construction"], int)
    assert hnsw["ef_construction"] > 0


def test_hnsw_config_has_ef_search(stats):
    """Test that HNSW config includes ef_search parameter."""
    hnsw = stats["hnsw_config"]
    assert "ef_search" in hnsw
    assert isinstance(hnsw["ef_search"], int)
    assert hnsw["ef_search"] > 0


def test_hnsw_config_has_max_neighbors(stats):
    """Test that HNSW config includes max_neighbors parameter."""
    hnsw = stats["hnsw_config"]
    assert "max_neighbors" in hnsw
    assert isinstance(hnsw["max_neighbors"], int)
    assert hnsw["max_neighbors"] > 0


def test_hnsw_config_has_resize_factor(stats):
    """Test that HNSW config includes resize_factor parameter."""
    hnsw = stats["hnsw_config"]
    assert "resize_factor" in hnsw
    assert isinstance(hnsw["resize_factor"], (int, float))
    assert hnsw["resize_factor"] > 1.0


def test_hnsw_config_has_sync_threshold(stats):
    """Test that HNSW config includes sync_threshold parameter."""
    hnsw = stats["hnsw_config"]
    assert "sync_threshold" in hnsw
    assert isinstance(hnsw["sync_threshold"], int)
    assert hnsw["sync_threshold"] > 0


def test_hnsw_config_all_fields_present(stats):
    """Test that all HNSW config fields are present."""
    hnsw = stats["hnsw_config"]
    expected_fields = {
        "space",
        "ef_construction",
        "ef_search",
        "max_neighbors",
        "resize_factor",
        "sync_threshold",
    }
    actual_fields = set(hnsw.keys())
    assert actual_fields == expected_fields


def test_hnsw_config_default_values(client):
    """Test that HNSW config returns expected default values."""
    stats = client.get("/index/stats").json()
    hnsw = stats["hnsw_config"]

    # These are ChromaDB's default values
    assert hnsw["space"] == "cosine"
    assert hnsw["ef_construction"] == 100
    assert hnsw["ef_search"] == 100
    assert hnsw["max_neighbors"] == 16
    assert hnsw["resize_factor"] == 1.2
    assert hnsw["sync_threshold"] == 1000


def test_hnsw_config_consistent_across_calls(client):
    """Test that HNSW config is consistent across multiple calls."""
    stats1 = client.get("/index/stats").json()
    stats2 = client.get("/index/stats").json()

    assert stats1["hnsw_config"] == stats2["hnsw_config"]
