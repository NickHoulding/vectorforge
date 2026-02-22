"""Tests for index management endpoints"""

import os

import pytest

from vectorforge import __version__
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


@pytest.fixture
def save_data(client):
    """Reusable index save data fixture"""
    resp = client.post(f"/index/save", params={"directory": TEST_DATA_PATH})
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def load_data(client):
    """Reusable index load data fixture"""
    resp = client.post(f"/index/load", params={"directory": TEST_DATA_PATH})
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
    assert stats["total_documents"] == 2


def test_index_stats_after_document_deletion(client, multiple_added_docs):
    """Test that stats update correctly after document deletion."""
    doc_id = multiple_added_docs[0]
    client.delete(f"/doc/{doc_id}")

    stats = client.get("/index/stats").json()
    assert stats["total_documents"] == 19


def test_index_stats_embedding_dimension_is_384(client):
    """Test that embedding_dimension matches configured model dimension."""
    stats = client.get("/index/stats").json()
    assert stats["embedding_dimension"] == VFGConfig.EMBEDDING_DIMENSION


def test_index_stats_multiple_deletions_immediate_removal(client, multiple_added_docs):
    """Test that multiple deletions are immediately reflected in stats."""
    for i in range(4):
        client.delete(f"/doc/{multiple_added_docs[i]}")

    stats = client.get("/index/stats").json()

    assert stats["total_documents"] == 16  # 20 - 4 = 16


# =============================================================================
# Index Build Tests
# =============================================================================


def test_index_after_deletions(client, multiple_added_docs):
    """Test index stats after deletions."""
    for i in range(5):
        client.delete(f"/doc/{multiple_added_docs[i]}")

    expected_total_docs = 15
    stats = client.get("/index/stats").json()
    assert stats["total_documents"] == expected_total_docs


# =============================================================================
# Index Save Tests
# =============================================================================


def test_index_save_returns_200(client):
    """Test that POST /index/save returns 200 status.

    ChromaDB auto-persists, so save() is informational only.
    """
    resp = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    assert resp.status_code == 200


def test_index_save_includes_status(save_data):
    """Test that save response includes status field."""
    assert "status" in save_data
    assert isinstance(save_data["status"], str)


def test_index_save_includes_directory(save_data):
    """Test that save response includes directory path."""
    assert "directory" in save_data
    assert isinstance(save_data["directory"], str)


def test_index_save_includes_document_count(save_data):
    """Test that save response includes number of documents saved."""
    assert "documents_saved" in save_data
    assert isinstance(save_data["documents_saved"], int)


def test_index_save_includes_version(save_data):
    """Test that save response includes version information."""
    assert "version" in save_data
    assert isinstance(save_data["version"], str)


def test_index_save_with_documents(client, multiple_added_docs):
    """Test saving index with actual documents."""
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()

    assert data["documents_saved"] == 20


def test_index_save_empty_index(client):
    """Test saving an empty index."""
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()

    assert data["documents_saved"] == 0


def test_index_save_basic(client, multiple_added_docs):
    """Test that save reflects current documents."""
    for i in range(5):
        client.delete(f"/doc/{multiple_added_docs[i]}")

    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()

    expected_saved = 15
    assert data["documents_saved"] == expected_saved


def test_index_save_after_deletions(client, multiple_added_docs):
    """Test saving after deletions."""
    for i in range(6):
        client.delete(f"/doc/{multiple_added_docs[i]}")

    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()

    expected_saved = 14
    assert data["documents_saved"] == expected_saved


def test_index_save_overwrites_existing_files(client, multiple_added_docs):
    """Test that save reflects current state after deletions.

    Note: ChromaDB's SQLite backend may not immediately reduce file size after
    deletions due to database vacuuming behavior. We verify the document count
    is accurate rather than expecting immediate size reduction.
    """
    response1 = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    docs_before = response1.json()["documents_saved"]
    assert docs_before == 20

    for i in range(10):
        client.delete(f"/doc/{multiple_added_docs[i]}")

    response2 = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data2 = response2.json()

    assert data2["documents_saved"] == 10


def test_index_save_version_matches_app_version(client):
    """Test that saved version matches application version."""
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()
    assert data["version"] == __version__


# =============================================================================
# Index Load Tests
# =============================================================================


def test_index_load_returns_200(client):
    """Test that POST /index/load returns 200 status.

    ChromaDB auto-loads on init, so load() is informational only.
    """
    resp = client.post("/index/load")
    assert resp.status_code == 200


def test_index_load_restores_documents(client, multiple_added_docs):
    """Test that documents are accessible (ChromaDB auto-loads)."""
    resp = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    assert resp.status_code == 200

    resp = client.post("/index/load", params={"directory": TEST_DATA_PATH})
    assert resp.status_code == 200

    for doc_id in multiple_added_docs:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 200


def test_index_load_includes_status(load_data):
    """Test that load response includes status field."""
    assert "status" in load_data
    assert isinstance(load_data["status"], str)


def test_index_load_includes_directory(load_data):
    """Test that load response includes directory path."""
    assert "directory" in load_data
    assert isinstance(load_data["directory"], str)


def test_index_load_includes_document_count(load_data):
    """Test that load response includes number of documents loaded."""
    assert "documents_loaded" in load_data
    assert isinstance(load_data["documents_loaded"], int)


def test_index_load_includes_version(load_data):
    """Test that load response includes version information."""
    assert "version" in load_data
    assert isinstance(load_data["version"], str)


def test_index_save_and_load_roundtrip(client, multiple_added_docs):
    """Test that saving and loading preserves data (ChromaDB auto-persists)."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    initial_metrics = resp.json()

    resp = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    assert resp.status_code == 200

    resp = client.post("/index/load", params={"directory": TEST_DATA_PATH})
    assert resp.status_code == 200

    resp = client.get("/metrics")
    assert resp.status_code == 200
    loaded_metrics = resp.json()

    assert (
        loaded_metrics["index"]["total_documents"]
        == initial_metrics["index"]["total_documents"]
    )

    assert (
        loaded_metrics["usage"]["documents_added"]
        == initial_metrics["usage"]["documents_added"]
    )
    assert (
        loaded_metrics["usage"]["documents_deleted"]
        == initial_metrics["usage"]["documents_deleted"]
    )
    assert (
        loaded_metrics["usage"]["chunks_created"]
        == initial_metrics["usage"]["chunks_created"]
    )
    assert (
        loaded_metrics["usage"]["files_uploaded"]
        == initial_metrics["usage"]["files_uploaded"]
    )

    assert (
        loaded_metrics["system"]["model_name"]
        == initial_metrics["system"]["model_name"]
    )
    assert (
        loaded_metrics["system"]["model_dimension"]
        == initial_metrics["system"]["model_dimension"]
    )
    assert loaded_metrics["system"]["version"] == initial_metrics["system"]["version"]

    for doc_id in multiple_added_docs:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 200

    search_resp = client.post("/search", json={"query": "test", "top_k": 5})
    assert search_resp.status_code == 200
    assert len(search_resp.json()["results"]) > 0


def test_index_load_basic(client, multiple_added_docs):
    """Test that load works correctly."""
    deleted_ids = multiple_added_docs[:3]
    for doc_id in deleted_ids:
        client.delete(f"/doc/{doc_id}")

    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    client.post("/index/load", params={"directory": TEST_DATA_PATH})

    for doc_id in deleted_ids:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 404

    for doc_id in multiple_added_docs[3:]:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 200


def test_index_load_replaces_current_state(client, multiple_added_docs):
    """Test that ChromaDB maintains current state (no file-based replacement).

    ChromaDB auto-persists, so new documents remain after load.
    """
    client.post("/index/save", params={"directory": TEST_DATA_PATH})

    new_doc = client.post(
        "/doc/add", json={"content": "new document after save", "metadata": {}}
    ).json()

    stats_before = client.get("/index/stats").json()
    assert stats_before["total_documents"] == 21

    client.post("/index/load", params={"directory": TEST_DATA_PATH})

    stats_after = client.get("/index/stats").json()
    assert stats_after["total_documents"] == 21

    resp = client.get(f"/doc/{new_doc['id']}")
    assert resp.status_code == 200


def test_index_load_preserves_metadata(client):
    """Test that load preserves document metadata."""
    original_metadata = {"source": "test.txt", "author": "test_user"}
    doc = client.post(
        "/doc/add", json={"content": "test content", "metadata": original_metadata}
    ).json()

    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    client.post("/index/load", params={"directory": TEST_DATA_PATH})

    loaded_doc = client.get(f"/doc/{doc['id']}").json()
    assert loaded_doc["metadata"] == original_metadata


def test_index_load_preserves_content(client):
    """Test that load preserves exact document content."""
    original_content = "This is the exact content that should be preserved"
    doc = client.post(
        "/doc/add", json={"content": original_content, "metadata": {}}
    ).json()

    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    client.post("/index/load", params={"directory": TEST_DATA_PATH})

    loaded_doc = client.get(f"/doc/{doc['id']}").json()
    assert loaded_doc["content"] == original_content


def test_index_load_preserves_search_results(client, multiple_added_docs):
    """Test that search results are consistent after load."""
    search_before = client.post(
        "/search", json={"query": "Python programming", "top_k": 5}
    ).json()

    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    client.post("/index/load", params={"directory": TEST_DATA_PATH})

    search_after = client.post(
        "/search", json={"query": "Python programming", "top_k": 5}
    ).json()

    ids_before = [r["id"] for r in search_before["results"]]
    ids_after = [r["id"] for r in search_after["results"]]
    assert ids_before == ids_after

    for r_before, r_after in zip(search_before["results"], search_after["results"]):
        assert abs(r_before["score"] - r_after["score"]) < 0.0001


def test_index_load_empty_index(client):
    """Test loading an index that was saved when empty."""
    client.post("/index/save", params={"directory": TEST_DATA_PATH})

    resp = client.post("/index/load", params={"directory": TEST_DATA_PATH})
    data = resp.json()

    assert data["documents_loaded"] == 0

    stats = client.get("/index/stats").json()
    assert stats["total_documents"] == 0


def test_index_load_counts_match_save(client, multiple_added_docs):
    """Test that load counts match what was saved."""
    save_resp = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    save_data = save_resp.json()

    load_resp = client.post("/index/load", params={"directory": TEST_DATA_PATH})
    load_data = load_resp.json()

    assert load_data["documents_loaded"] == save_data["documents_saved"]


def test_index_load_multiple_times_idempotent(client, multiple_added_docs):
    """Test that loading the same data multiple times produces same result."""
    client.post("/index/save", params={"directory": TEST_DATA_PATH})

    load1 = client.post("/index/load", params={"directory": TEST_DATA_PATH}).json()
    stats1 = client.get("/index/stats").json()

    load2 = client.post("/index/load", params={"directory": TEST_DATA_PATH}).json()
    stats2 = client.get("/index/stats").json()

    assert load1["documents_loaded"] == load2["documents_loaded"]
    assert stats1 == stats2


def test_index_load_version_information(client):
    """Test that load response includes version of loaded data."""
    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    load_data = client.post("/index/load", params={"directory": TEST_DATA_PATH}).json()
    assert load_data["version"] == __version__


def test_index_load_with_custom_directory(client, multiple_added_docs):
    """Test load with custom directory parameter."""
    custom_dir = os.path.join(TEST_DATA_PATH, "custom_load")

    resp = client.post("/index/load", params={"directory": custom_dir})
    assert resp.status_code == 200

    data = resp.json()
    assert data["directory"] == custom_dir
    assert data["documents_loaded"] == 20


def test_index_load_with_very_long_directory_path(client):
    """Test load with very long directory path (ChromaDB ignores parameter)."""
    long_dir = "a" * (VFGConfig.MAX_PATH_LEN + 1)

    resp = client.post("/index/load", params={"directory": long_dir})
    assert resp.status_code == 200


def test_index_load_preserves_document_ids(client, multiple_added_docs):
    """Test that document IDs remain unchanged after load."""
    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    client.post("/index/load", params={"directory": TEST_DATA_PATH})

    for doc_id in multiple_added_docs:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == doc_id
