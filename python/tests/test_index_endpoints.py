"""Tests for index management endpoints"""

import json
import os
import shutil

import numpy as np
import pytest

from vectorforge import __version__
from vectorforge.config import Config


TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data"
)


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
    resp = client.post(f"/index/save", params={
        "directory": TEST_DATA_PATH
    })
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def load_data(client):
    """Reusable index load data fixture"""
    resp = client.post(f"/index/load", params={
        "directory": TEST_DATA_PATH
    })
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


def test_index_stats_returns_total_embeddings(stats):
    """Test that index stats includes total embeddings count."""
    assert "total_embeddings" in stats
    assert isinstance(stats["total_embeddings"], int)


def test_index_stats_returns_deleted_documents(stats):
    """Test that index stats includes deleted documents count."""
    assert "deleted_documents" in stats
    assert isinstance(stats["deleted_documents"], int)


def test_index_stats_returns_deleted_ratio(stats):
    """Test that index stats includes deleted ratio calculation."""
    assert "deleted_ratio" in stats
    assert isinstance(stats["deleted_ratio"], float)


def test_index_stats_returns_needs_compaction(stats):
    """Test that index stats includes compaction status."""
    assert "needs_compaction" in stats
    assert isinstance(stats["needs_compaction"], bool)


def test_index_stats_returns_embedding_dimension(stats):
    """Test that index stats includes embedding dimension."""
    assert "embedding_dimension" in stats
    assert isinstance(stats["embedding_dimension"], int)


def test_index_stats_with_empty_index(client):
    """Test index stats when index is empty."""
    stats = client.get("/index/stats").json()
    
    assert stats["total_documents"] == 0
    assert stats["total_embeddings"] == 0
    assert stats["deleted_documents"] == 0
    assert stats["deleted_ratio"] == 0.0
    assert stats["needs_compaction"] is False
    assert stats["embedding_dimension"] == Config.EMBEDDING_DIMENSION


def test_index_stats_after_adding_documents(client):
    """Test that stats update correctly after adding documents."""
    client.post("/doc/add", json={"content": "test doc 1", "metadata": {}})
    client.post("/doc/add", json={"content": "test doc 2", "metadata": {}})
    
    stats = client.get("/index/stats").json()
    assert stats["total_documents"] == 2
    assert stats["total_embeddings"] == 2
    assert stats["deleted_documents"] == 0
    assert stats["deleted_ratio"] == 0.0


def test_index_stats_after_document_deletion(client, multiple_added_docs):
    """Test that stats track deleted documents correctly."""
    doc_id = multiple_added_docs[0]
    client.delete(f"/doc/{doc_id}")
    
    stats = client.get("/index/stats").json()
    assert stats["total_documents"] == 20
    assert stats["total_embeddings"] == 20
    assert stats["deleted_documents"] == 1
    assert stats["deleted_ratio"] > 0.0


def test_index_stats_deleted_ratio_calculation(client):
    """Test that deleted_ratio is calculated correctly."""
    doc_ids = []
    for i in range(5):
        resp = client.post("/doc/add", json={"content": f"doc {i}", "metadata": {}})
        doc_ids.append(resp.json()["id"])
    
    client.delete(f"/doc/{doc_ids[0]}")

    stats = client.get("/index/stats").json()
    assert stats["total_embeddings"] == 5
    assert stats["deleted_documents"] == 1
    assert stats["deleted_ratio"] == 0.20


def test_index_stats_needs_compaction_false(client):
    """Test that needs_compaction is False when below threshold."""
    for i in range(10):
        client.post("/doc/add", json={"content": f"doc {i}", "metadata": {}})
    
    stats = client.get("/index/stats").json()
    assert stats["needs_compaction"] is False

def test_index_stats_needs_compaction_true(client, multiple_added_docs):
    """Test that needs_compaction is True when above threshold."""
    for i in range(5):
        client.delete(f"/doc/{multiple_added_docs[i]}")
    
    stats = client.get("/index/stats").json()
    assert stats["deleted_ratio"] == Config.COMPACTION_THRESHOLD
    assert stats["needs_compaction"] is False
    
    client.delete(f"/doc/{multiple_added_docs[5]}")
    
    stats = client.get("/index/stats").json()
    assert stats["deleted_documents"] == 0
    assert stats["deleted_ratio"] == 0.0
    assert stats["needs_compaction"] is False


def test_index_stats_embedding_dimension_is_384(client):
    """Test that embedding_dimension matches configured model dimension."""
    stats = client.get("/index/stats").json()
    assert stats["embedding_dimension"] == Config.EMBEDDING_DIMENSION


def test_index_stats_multiple_deletions_below_threshold(client, multiple_added_docs):
    """Test stats with multiple deletions that stay below compaction threshold."""
    for i in range(4):
        client.delete(f"/doc/{multiple_added_docs[i]}")
    
    stats = client.get("/index/stats").json()
    
    assert stats["total_documents"] == 20
    assert stats["total_embeddings"] == 20
    assert stats["deleted_documents"] == 4
    assert stats["deleted_ratio"] == 0.20
    assert stats["needs_compaction"] is False


def test_index_stats_after_manual_build(client, multiple_added_docs):
    """Test that stats update correctly after manual index build."""
    client.delete(f"/doc/{multiple_added_docs[0]}")
    client.delete(f"/doc/{multiple_added_docs[1]}")
    
    client.post("/index/build")
    
    stats = client.get("/index/stats").json()
    assert stats["deleted_documents"] == 0
    assert stats["deleted_ratio"] == 0.0
    assert stats["total_documents"] == 18
    assert stats["total_embeddings"] == 18
    assert stats["needs_compaction"] is False


def test_index_stats_deleted_ratio_with_zero_embeddings(client):
    """Test that deleted_ratio is 0.0 when no embeddings exist."""
    stats = client.get("/index/stats").json()
    assert stats["total_embeddings"] == 0
    assert stats["deleted_ratio"] == 0.0


# =============================================================================
# Index Build Tests
# =============================================================================

def test_index_build_returns_200(client):
    """Test that POST /index/build returns 200 status."""
    resp = client.post("/index/build")
    assert resp.status_code == 200


def test_index_build_reconstructs_index(client, multiple_added_docs):
    """Test that building the index reconstructs it from documents."""
    for i in range(4):
        resp = client.delete(f"/doc/{multiple_added_docs[i]}")
        assert resp.status_code == 200
    
    stats_before = client.get("/index/stats").json()
    assert stats_before["deleted_documents"] == 4
    assert stats_before["total_embeddings"] == 20
    
    resp = client.post("/index/build")
    assert resp.status_code == 200
    
    stats_after = client.get("/index/stats").json()
    assert stats_after["deleted_documents"] == 0
    assert stats_after["total_embeddings"] == 16
    assert stats_after["total_documents"] == 16
    assert stats_after["deleted_ratio"] == 0.0
    
    search_resp = client.post("/search", json={
        "query": "test",
        "top_k": 20
    })
    results = search_resp.json()["results"]
    
    assert len(results) == 16
    
    result_ids = [r["id"] for r in results]
    for i in range(4):
        assert multiple_added_docs[i] not in result_ids


def test_index_build_returns_stats_in_response(client, multiple_added_docs):
    """Test that build response includes updated index stats."""
    for i in range(2):
        client.delete(f"/doc/{multiple_added_docs[i]}")
    
    response = client.post("/index/build")
    data = response.json()
    
    assert "total_documents" in data
    assert "total_embeddings" in data
    assert "deleted_documents" in data
    assert data["total_documents"] == 18
    assert data["deleted_documents"] == 0


def test_index_build_returns_updated_stats(client, multiple_added_docs):
    """Test that index build returns updated statistics."""
    for i in range(2):
        client.delete(f"/doc/{multiple_added_docs[i]}")
    
    resp = client.get("/index/stats")
    assert resp.status_code == 200
    
    initial_stats = resp.json()

    resp = client.post("/index/build")
    assert resp.status_code == 200

    resp = client.get("/index/stats")
    assert resp.status_code == 200

    updated_stats = resp.json()
    
    assert initial_stats["total_documents"] == updated_stats["total_documents"] + 2
    assert initial_stats["total_embeddings"] == updated_stats["total_embeddings"] + 2
    assert initial_stats["deleted_documents"] == 2
    assert updated_stats["deleted_documents"] == 0
    assert initial_stats["deleted_ratio"] == 0.1
    assert updated_stats["deleted_ratio"] == 0.0
    assert initial_stats["needs_compaction"] == False
    assert updated_stats["needs_compaction"] == False
    assert initial_stats["embedding_dimension"] == Config.EMBEDDING_DIMENSION
    assert updated_stats["embedding_dimension"] == Config.EMBEDDING_DIMENSION


def test_index_build_with_empty_index(client):
    """Test building an index when no documents exist."""
    resp = client.post("/index/build")
    assert resp.status_code == 200


def test_index_build_returns_success_status(client):
    """Test that build response contains 'success' status."""
    response = client.post("/index/build")
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"


def test_index_build_increments_compactions_metric(client, multiple_added_docs):
    """Test that building index increments compactions_performed metric."""
    initial_metrics = client.get("/metrics").json()
    initial_compactions = initial_metrics["usage"]["compactions_performed"]
    
    client.delete(f"/doc/{multiple_added_docs[0]}")
    client.post("/index/build")
    
    updated_metrics = client.get("/metrics").json()
    updated_compactions = updated_metrics["usage"]["compactions_performed"]
    
    assert updated_compactions == initial_compactions + 1


def test_index_build_twice_is_idempotent(client, multiple_added_docs):
    """Test that building twice when no changes doesn't cause issues."""
    response1 = client.post("/index/build")
    assert response1.status_code == 200
    stats1 = client.get("/index/stats").json()
    
    response2 = client.post("/index/build")
    assert response2.status_code == 200
    stats2 = client.get("/index/stats").json()
    
    assert stats1 == stats2


def test_index_build_preserves_document_content(client, added_doc):
    """Test that rebuild doesn't alter document content."""
    doc_id = added_doc["id"]
    before_response = client.get(f"/doc/{doc_id}")
    before_content = before_response.json()["content"]
    
    client.post("/index/build")
    
    after_response = client.get(f"/doc/{doc_id}")
    after_content = after_response.json()["content"]
    
    assert after_content == before_content


def test_index_build_preserves_document_metadata(client, added_doc):
    """Test that rebuild preserves document metadata."""
    doc_id = added_doc["id"]
    before_response = client.get(f"/doc/{doc_id}")
    before_metadata = before_response.json()["metadata"]
    
    client.post("/index/build")
    
    after_response = client.get(f"/doc/{doc_id}")
    after_metadata = after_response.json()["metadata"]
    
    assert after_metadata == before_metadata


def test_index_build_after_deleting_all_documents(client, multiple_added_docs):
    """Test rebuilding after all documents are deleted."""
    for doc_id in multiple_added_docs:
        client.delete(f"/doc/{doc_id}")
    
    response = client.post("/index/build")
    assert response.status_code == 200
    
    stats = client.get("/index/stats").json()
    assert stats["total_documents"] == 0
    assert stats["total_embeddings"] == 0
    assert stats["deleted_documents"] == 0


def test_index_build_preserves_search_scores(client, multiple_added_docs):
    """Test that rebuild doesn't significantly change search result scores."""
    before_response = client.post("/search", json={
        "query": "Python programming",
        "top_k": 5
    })
    before_scores = [r["score"] for r in before_response.json()["results"]]
    
    client.post("/index/build")
    
    after_response = client.post("/search", json={
        "query": "Python programming",
        "top_k": 5
    })
    after_scores = [r["score"] for r in after_response.json()["results"]]
    
    for before, after in zip(before_scores, after_scores):
        assert abs(before - after) < 0.0001


def test_index_build_with_only_deleted_documents(client, multiple_added_docs):
    """Test building when all documents are marked deleted but still in index."""
    for doc_id in multiple_added_docs:
        client.delete(f"/doc/{doc_id}")
    
    response = client.post("/index/build")
    assert response.status_code == 200
    
    stats = client.get("/index/stats").json()
    assert stats["total_documents"] == 0
    assert stats["total_embeddings"] == 0
    assert stats["deleted_documents"] == 0


def test_index_build_response_structure(client, multiple_added_docs):
    """Test that build response contains all expected fields."""
    client.delete(f"/doc/{multiple_added_docs[0]}")
    
    response = client.post("/index/build")
    data = response.json()
    
    required_fields = [
        "status",
        "total_documents", 
        "total_embeddings",
        "deleted_documents",
        "deleted_ratio",
        "needs_compaction",
        "embedding_dimension"
    ]
    
    for field in required_fields:
        assert field in data


def test_index_build_at_compaction_threshold(client, multiple_added_docs):
    """Test building when deleted ratio is exactly at threshold."""
    for i in range(5):
        client.delete(f"/doc/{multiple_added_docs[i]}")
    
    stats_before = client.get("/index/stats").json()
    assert stats_before["deleted_ratio"] == Config.COMPACTION_THRESHOLD
    
    response = client.post("/index/build")
    assert response.status_code == 200
    
    stats_after = client.get("/index/stats").json()
    assert stats_after["total_documents"] == 15
    assert stats_after["deleted_documents"] == 0


def test_index_build_updates_index_mappings(client, multiple_added_docs):
    """Test that build correctly updates internal index-to-doc-id mappings."""
    deleted_ids = multiple_added_docs[:3]
    for doc_id in deleted_ids:
        client.delete(f"/doc/{doc_id}")
    
    client.post("/index/build")
    
    remaining_ids = multiple_added_docs[3:]
    for doc_id in remaining_ids:
        response = client.get(f"/doc/{doc_id}")
        assert response.status_code == 200
        assert response.json()["id"] == doc_id
    
    for doc_id in deleted_ids:
        response = client.get(f"/doc/{doc_id}")
        assert response.status_code == 404


# =============================================================================
# Index Save Tests
# =============================================================================

def test_index_save_returns_200(client):
    """Test that POST /index/save returns 200 status."""
    resp = client.post("/index/save", params={
        "directory": TEST_DATA_PATH
    })
    assert resp.status_code == 200


def test_index_save_creates_directory_if_not_exists(client):
    """Test that save creates the target directory if it doesn't exist."""
    shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)
    assert not os.path.exists(TEST_DATA_PATH)

    resp = client.post("/index/save", params={
        "directory": TEST_DATA_PATH
    })
    assert resp.status_code == 200

    assert os.path.exists(TEST_DATA_PATH)


def test_index_save_persists_to_disk(save_data):
    """Test that saving index creates files on disk."""
    files = os.listdir(save_data["directory"])
    assert Config.METADATA_FILENAME in files
    assert Config.EMBEDDINGS_FILENAME in files


def test_index_save_returns_save_metrics(save_data):
    """Test that save response includes metrics like file sizes."""
    assert "metadata_size_mb" in save_data
    assert "embeddings_size_mb" in save_data
    assert "total_size_mb" in save_data
    assert isinstance(save_data["metadata_size_mb"], float)
    assert isinstance(save_data["embeddings_size_mb"], float)
    assert isinstance(save_data["total_size_mb"], float)


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


def test_index_save_includes_embeddings_count(save_data):
    """Test that save response includes embeddings_saved count."""
    assert "embeddings_saved" in save_data
    assert isinstance(save_data["embeddings_saved"], int)


def test_index_save_includes_version(save_data):
    """Test that save response includes version information."""
    assert "version" in save_data
    assert isinstance(save_data["version"], str)


def test_index_save_with_documents(client, multiple_added_docs):
    """Test saving index with actual documents."""
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()
    
    assert data["documents_saved"] == 20
    assert data["embeddings_saved"] == 20
    assert data["metadata_size_mb"] > 0
    assert data["embeddings_size_mb"] > 0


def test_index_save_empty_index(client):
    """Test saving an empty index."""
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()
    
    assert data["documents_saved"] == 0
    assert data["embeddings_saved"] == 0
    assert os.path.exists(os.path.join(
        TEST_DATA_PATH,
        Config.METADATA_FILENAME
    ))


def test_index_save_excludes_deleted_documents(client, multiple_added_docs):
    """Test that save only persists active documents, not deleted ones."""
    for i in range(5):
        client.delete(f"/doc/{multiple_added_docs[i]}")
    
    stats_before = client.get("/index/stats").json()
    assert stats_before["deleted_documents"] == 5
    
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()
    
    assert data["documents_saved"] == 15
    assert data["embeddings_saved"] == 15
    
    stats_after = client.get("/index/stats").json()
    assert stats_after["deleted_documents"] == 5


def test_index_save_after_compaction(client, multiple_added_docs):
    """Test saving after compaction removes deleted documents from saved data."""
    for i in range(6):
        client.delete(f"/doc/{multiple_added_docs[i]}")
    
    stats = client.get("/index/stats").json()
    assert stats["deleted_documents"] == 0
    
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()
    
    assert data["documents_saved"] == 14
    assert data["embeddings_saved"] == 14


def test_index_save_file_sizes_are_positive(client, multiple_added_docs):
    """Test that file sizes are positive numbers when data exists."""
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()
    
    assert data["metadata_size_mb"] > 0
    assert data["embeddings_size_mb"] > 0
    assert data["total_size_mb"] > 0
    assert data["total_size_mb"] == data["metadata_size_mb"] + data["embeddings_size_mb"]


def test_index_save_total_size_equals_sum(client, multiple_added_docs):
    """Test that total_size_mb equals sum of metadata and embeddings sizes."""
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()
    
    expected_total = data["metadata_size_mb"] + data["embeddings_size_mb"]
    assert abs(data["total_size_mb"] - expected_total) < 0.001


def test_index_save_overwrites_existing_files(client, multiple_added_docs):
    """Test that saving twice overwrites previous save."""
    response1 = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    size1 = response1.json()["total_size_mb"]
    
    for i in range(10):
        client.delete(f"/doc/{multiple_added_docs[i]}")
    
    response2 = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data2 = response2.json()
    
    assert data2["documents_saved"] == 10
    assert data2["total_size_mb"] < size1


def test_index_save_version_matches_app_version(client):
    """Test that saved version matches application version."""
    response = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    data = response.json()
    assert data["version"] == __version__


def test_index_save_with_default_directory(client):
    """Test saving without specifying directory uses default."""
    response = client.post("/index/save")
    assert response.status_code == 200
    
    data = response.json()
    assert "directory" in data
    assert data["directory"] == Config.DEFAULT_DATA_DIR


def test_index_save_creates_valid_json_metadata(client, added_doc):
    """Test that saved metadata.json is valid JSON."""
    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    
    metadata_path = os.path.join(TEST_DATA_PATH, Config.METADATA_FILENAME)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    assert "documents" in metadata
    assert "metrics" in metadata
    assert "version" in metadata


def test_index_save_creates_valid_embeddings_file(client, added_doc):
    """Test that saved embeddings.npz is valid numpy format."""
    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    
    embeddings_path = os.path.join(TEST_DATA_PATH, Config.EMBEDDINGS_FILENAME)
    data = np.load(embeddings_path)
    
    assert "embeddings" in data
    assert len(data["embeddings"]) > 0


def test_index_save_with_very_long_directory_path(client):
    """Test save with directory path over the maximum length."""
    long_dir = "a" * (Config.MAX_PATH_LEN + 1)

    resp = client.post("/index/save", params={"directory": long_dir})
    assert resp.status_code == 400


# =============================================================================
# Index Load Tests
# =============================================================================

def test_index_load_returns_200(client):
    """Test that POST /index/load returns 200 status."""
    resp = client.post("/index/load")
    assert resp.status_code == 200


def test_index_load_restores_documents(client, multiple_added_docs):
    """Test that loading index restores previously saved documents."""
    resp = client.post("/index/save", params={
        "directory": TEST_DATA_PATH
    })
    assert resp.status_code == 200

    resp = client.post("/index/load", params={
        "directory": TEST_DATA_PATH
    })
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


def test_index_load_includes_embeddings_count(load_data):
    """Test that load response includes number of embeddings loaded."""
    assert "embeddings_loaded" in load_data
    assert isinstance(load_data["embeddings_loaded"], int)


def test_index_load_includes_deleted_docs(load_data):
    """Test that load response includes number of deleted docs."""
    assert "deleted_docs" in load_data
    assert isinstance(load_data["deleted_docs"], int)


def test_index_load_includes_version(load_data):
    """Test that load response includes version information."""
    assert "version" in load_data
    assert isinstance(load_data["version"], str)


def test_index_load_from_nonexistent_directory(client):
    """Test that loading from nonexistent directory returns 404."""
    resp = client.post("/index/load", params={
        "directory": "/path/that/does/not/exist"
    })
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()

def test_index_load_with_missing_metadata_file(client):
    """Test that load returns 404 when only metadata.json is missing."""
    resp = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    os.remove(os.path.join(TEST_DATA_PATH, "metadata.json"))
    resp = client.post("/index/load", params={"directory": TEST_DATA_PATH})
    
    assert resp.status_code == 404


def test_index_load_with_missing_embeddings_file(client):
    """Test that load returns 404 when only embeddings.npz is missing."""
    resp = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    os.remove(os.path.join(TEST_DATA_PATH, "embeddings.npz"))
    resp = client.post("/index/load", params={"directory": TEST_DATA_PATH})

    assert resp.status_code == 404


def test_index_save_and_load_roundtrip(client, multiple_added_docs):
    """Test that saving and loading preserves all data correctly."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    initial_metrics = resp.json()
    
    resp = client.post("/index/save", params={
        "directory": TEST_DATA_PATH
    })
    assert resp.status_code == 200

    resp = client.post("/index/load", params={
        "directory": TEST_DATA_PATH
    })
    assert resp.status_code == 200

    resp = client.get("/metrics")
    assert resp.status_code == 200
    loaded_metrics = resp.json()

    assert loaded_metrics["index"]["total_documents"] == initial_metrics["index"]["total_documents"]
    assert loaded_metrics["index"]["total_embeddings"] == initial_metrics["index"]["total_embeddings"]
    assert loaded_metrics["index"]["deleted_documents"] == initial_metrics["index"]["deleted_documents"]
    assert loaded_metrics["index"]["deleted_ratio"] == initial_metrics["index"]["deleted_ratio"]
    assert loaded_metrics["index"]["needs_compaction"] == initial_metrics["index"]["needs_compaction"]
    assert loaded_metrics["index"]["compact_threshold"] == initial_metrics["index"]["compact_threshold"]
    
    assert loaded_metrics["usage"]["documents_added"] == initial_metrics["usage"]["documents_added"]
    assert loaded_metrics["usage"]["documents_deleted"] == initial_metrics["usage"]["documents_deleted"]
    assert loaded_metrics["usage"]["chunks_created"] == initial_metrics["usage"]["chunks_created"]
    assert loaded_metrics["usage"]["files_uploaded"] == initial_metrics["usage"]["files_uploaded"]
    
    assert loaded_metrics["system"]["model_name"] == initial_metrics["system"]["model_name"]
    assert loaded_metrics["system"]["model_dimension"] == initial_metrics["system"]["model_dimension"]
    assert loaded_metrics["system"]["version"] == initial_metrics["system"]["version"]
    
    for doc_id in multiple_added_docs:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 200
        
    search_resp = client.post("/search", json={
        "query": "test",
        "top_k": 5
    })
    assert search_resp.status_code == 200
    assert len(search_resp.json()["results"]) > 0


def test_index_load_excludes_deleted_documents(client, multiple_added_docs):
    """Test that load doesn't restore documents deleted before save."""
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
    """Test that load completely replaces the current index."""
    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    
    new_doc = client.post("/doc/add", json={
        "content": "new document after save",
        "metadata": {}
    }).json()
    
    stats_before = client.get("/index/stats").json()
    assert stats_before["total_documents"] == 21
    
    client.post("/index/load", params={"directory": TEST_DATA_PATH})
    
    stats_after = client.get("/index/stats").json()
    assert stats_after["total_documents"] == 20
    
    resp = client.get(f"/doc/{new_doc['id']}")
    assert resp.status_code == 404


def test_index_load_preserves_metadata(client):
    """Test that load preserves document metadata."""
    original_metadata = {"source": "test.txt", "author": "test_user"}
    doc = client.post("/doc/add", json={
        "content": "test content",
        "metadata": original_metadata
    }).json()
    
    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    client.post("/index/load", params={"directory": TEST_DATA_PATH})
    
    loaded_doc = client.get(f"/doc/{doc['id']}").json()
    assert loaded_doc["metadata"] == original_metadata


def test_index_load_preserves_content(client):
    """Test that load preserves exact document content."""
    original_content = "This is the exact content that should be preserved"
    doc = client.post("/doc/add", json={
        "content": original_content,
        "metadata": {}
    }).json()
    
    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    client.post("/index/load", params={"directory": TEST_DATA_PATH})
    
    loaded_doc = client.get(f"/doc/{doc['id']}").json()
    assert loaded_doc["content"] == original_content


def test_index_load_preserves_search_results(client, multiple_added_docs):
    """Test that search results are consistent after load."""
    search_before = client.post("/search", json={
        "query": "Python programming",
        "top_k": 5
    }).json()
    
    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    client.post("/index/load", params={"directory": TEST_DATA_PATH})
    
    search_after = client.post("/search", json={
        "query": "Python programming",
        "top_k": 5
    }).json()
    
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
    assert data["embeddings_loaded"] == 0
    assert data["deleted_docs"] == 0
    
    stats = client.get("/index/stats").json()
    assert stats["total_documents"] == 0
    assert stats["total_embeddings"] == 0


def test_index_load_counts_match_save(client, multiple_added_docs):
    """Test that load counts match what was saved."""
    save_resp = client.post("/index/save", params={"directory": TEST_DATA_PATH})
    save_data = save_resp.json()
    
    load_resp = client.post("/index/load", params={"directory": TEST_DATA_PATH})
    load_data = load_resp.json()
    
    assert load_data["documents_loaded"] == save_data["documents_saved"]
    assert load_data["embeddings_loaded"] == save_data["embeddings_saved"]


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
    """Test loading from a custom directory path."""
    custom_dir = os.path.join(TEST_DATA_PATH, "custom_load")
    
    client.post("/index/save", params={"directory": custom_dir})
    
    resp = client.post("/index/load", params={"directory": custom_dir})
    assert resp.status_code == 200
    
    data = resp.json()
    assert data["directory"] == custom_dir
    assert data["documents_loaded"] == 20


def test_index_load_with_very_long_directory_path(client):
    """Test load with directory path over the maximum length."""
    long_dir = "a" * (Config.MAX_PATH_LEN + 1)

    resp = client.post("/index/load", params={"directory": long_dir})
    assert resp.status_code == 400

def test_index_load_preserves_document_ids(client, multiple_added_docs):
    """Test that document IDs remain unchanged after load."""
    client.post("/index/save", params={"directory": TEST_DATA_PATH})
    client.post("/index/load", params={"directory": TEST_DATA_PATH})
    
    for doc_id in multiple_added_docs:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == doc_id
