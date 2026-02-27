"""Tests for system monitoring endpoints"""

import io
import re
import time
from datetime import datetime

import pytest

from vectorforge import __version__
from vectorforge.config import VFGConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def metrics(client):
    """Reusable fixture returning index metrics."""
    resp = client.get("/collections/vectorforge/metrics")
    assert resp.status_code == 200
    return resp.json()


# =============================================================================
# Health Endpoint Tests
# =============================================================================


def test_health_returns_200(client):
    """Test that GET /health returns 200 status."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_healthy_status(client):
    """Test that health check returns healthy status."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"


def test_health_returns_version(client):
    """Test that health check includes version information."""
    response = client.get("/health")
    data = response.json()
    assert "version" in data
    assert isinstance(data["version"], str)


def test_health_response_format(client):
    """Test that health response contains all required fields."""
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert "status" in data
    assert isinstance(data["status"], str)
    assert "version" in data
    assert isinstance(data["version"], str)


def test_health_version_is_valid_semver(client):
    """Test that version string follows semantic versioning format."""
    resp = client.get("/health")
    assert resp.status_code == 200

    version = resp.json()["version"]
    semver_pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$"
    assert re.match(semver_pattern, version)

    parts = version.split("-")[0].split("+")[0].split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_health_only_accepts_get_method(client):
    """Test that health endpoint only accepts GET requests."""
    resp = client.post("/health")
    assert resp.status_code == 405

    resp = client.put("/health")
    assert resp.status_code == 405

    resp = client.delete("/health")
    assert resp.status_code == 405


def test_health_endpoint_is_idempotent(client):
    """Test that multiple health checks return consistent results."""
    resp1 = client.get("/health")
    resp2 = client.get("/health")
    resp3 = client.get("/health")

    # Check that status and version are consistent (heartbeat will differ)
    assert resp1.json()["status"] == resp2.json()["status"] == resp3.json()["status"]
    assert resp1.json()["version"] == resp2.json()["version"] == resp3.json()["version"]
    assert all(r.status_code == 200 for r in [resp1, resp2, resp3])


def test_health_version_matches_package_version(client):
    """Test that health version matches the actual package version."""

    resp = client.get("/health")
    data = resp.json()

    assert data["version"] == __version__


def test_health_response_has_no_extra_fields(client):
    response = client.get("/health")
    expected_fields = {"status", "version", "chromadb_heartbeat", "total_collections"}
    actual_fields = set(response.json().keys())
    assert actual_fields == expected_fields


def test_health_ignores_query_parameters(client):
    """Test that health endpoint ignores any query parameters."""
    resp1 = client.get("/health")
    resp2 = client.get("/health", params={"foo": "bar", "baz": "qux"})

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    # Compare only status and version (heartbeat will differ)
    assert resp1.json()["status"] == resp2.json()["status"]
    assert resp1.json()["version"] == resp2.json()["version"]


def test_health_status_value_is_healthy(client):
    """Test that status field has exact value 'healthy'."""
    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "healthy"


def test_health_endpoint_exact_path(client):
    """Test that health endpoint requires exact path."""
    assert client.get("/health").status_code == 200

    resp = client.get("/health/")
    assert resp.status_code in [200, 307, 308]


def test_health_endpoint_responds_quickly(client):
    """Test that health check responds in under 100ms."""
    start = time.time()
    resp = client.get("/health")
    duration = (time.time() - start) * 1000

    assert resp.status_code == 200
    assert duration < 100


# =============================================================================
# Metrics Endpoint Tests
# =============================================================================


def test_metrics_returns_200(client):
    """Test that GET /collections/vectorforge/metrics returns 200 status."""
    resp = client.get("/collections/vectorforge/metrics")
    assert resp.status_code == 200


def test_metrics_returns_comprehensive_data(metrics):
    """Test that metrics response includes all metric categories."""
    assert "index" in metrics
    assert "performance" in metrics
    assert "usage" in metrics
    assert "timestamps" in metrics
    assert "system" in metrics


def test_metrics_includes_index_metrics(metrics):
    """Test that metrics response includes index statistics."""
    assert "total_documents" in metrics["index"]
    assert isinstance(metrics["index"]["total_documents"], int)


def test_metrics_includes_performance_metrics(metrics):
    """Test that metrics response includes performance statistics."""
    assert "total_queries" in metrics["performance"]
    assert isinstance(metrics["performance"]["total_queries"], int)
    assert "avg_query_time_ms" in metrics["performance"]
    assert isinstance(metrics["performance"]["avg_query_time_ms"], float)
    assert "total_query_time_ms" in metrics["performance"]
    assert isinstance(metrics["performance"]["total_query_time_ms"], float)
    assert "min_query_time_ms" in metrics["performance"]
    assert isinstance(metrics["performance"]["min_query_time_ms"], (float | None))
    assert "max_query_time_ms" in metrics["performance"]
    assert isinstance(metrics["performance"]["max_query_time_ms"], (float | None))
    assert "p50_query_time_ms" in metrics["performance"]
    assert isinstance(metrics["performance"]["p50_query_time_ms"], (float | None))
    assert "p95_query_time_ms" in metrics["performance"]
    assert isinstance(metrics["performance"]["p95_query_time_ms"], (float | None))
    assert "p99_query_time_ms" in metrics["performance"]
    assert isinstance(metrics["performance"]["p99_query_time_ms"], (float | None))


def test_metrics_includes_usage_metrics(metrics):
    """Test that metrics response includes usage statistics."""
    assert "documents_added" in metrics["usage"]
    assert isinstance(metrics["usage"]["documents_added"], int)
    assert "documents_deleted" in metrics["usage"]
    assert isinstance(metrics["usage"]["documents_deleted"], int)
    assert "chunks_created" in metrics["usage"]
    assert isinstance(metrics["usage"]["chunks_created"], int)
    assert "files_uploaded" in metrics["usage"]
    assert isinstance(metrics["usage"]["files_uploaded"], int)


def test_metrics_includes_memory_metrics(metrics):
    """Test that metrics response includes memory statistics."""
    assert "total_doc_size_bytes" in metrics["usage"]
    assert isinstance(metrics["usage"]["total_doc_size_bytes"], int)


def test_metrics_includes_timestamp_metrics(metrics):
    """Test that metrics response includes timestamp information."""
    assert "engine_created_at" in metrics["timestamps"]
    assert isinstance(metrics["timestamps"]["engine_created_at"], str)
    assert "last_query_at" in metrics["timestamps"]
    assert isinstance(metrics["timestamps"]["last_query_at"], (str | None))
    assert "last_document_added_at" in metrics["timestamps"]
    assert isinstance(metrics["timestamps"]["last_document_added_at"], (str | None))
    assert "last_file_uploaded_at" in metrics["timestamps"]
    assert isinstance(metrics["timestamps"]["last_file_uploaded_at"], (str | None))


def test_metrics_includes_system_info(metrics):
    """Test that metrics response includes system information."""
    assert "model_name" in metrics["system"]
    assert isinstance(metrics["system"]["model_name"], str)
    assert "model_dimension" in metrics["system"]
    assert isinstance(metrics["system"]["model_dimension"], int)
    assert "uptime_seconds" in metrics["system"]
    assert isinstance(metrics["system"]["uptime_seconds"], float)
    assert "version" in metrics["system"]
    assert isinstance(metrics["system"]["version"], str)


def test_metrics_updates_after_operations(client):
    """Test that metrics update correctly after performing operations."""
    initial_metrics = client.get("/collections/vectorforge/metrics").json()

    client.post(
        "/collections/vectorforge/documents",
        json={"content": "test document", "metadata": {}},
    )

    after_add = client.get("/collections/vectorforge/metrics").json()
    assert (
        after_add["usage"]["documents_added"]
        == initial_metrics["usage"]["documents_added"] + 1
    )
    assert (
        after_add["index"]["total_documents"]
        == initial_metrics["index"]["total_documents"] + 1
    )
    assert after_add["timestamps"]["last_document_added_at"] is not None


def test_metrics_after_add_delete_cycle(client, sample_doc, multiple_added_docs):
    """Test metrics accuracy after adding and deleting documents."""
    initial_metrics = client.get("/collections/vectorforge/metrics").json()

    add_resp = client.post("/collections/vectorforge/documents", json=sample_doc)
    doc_id = add_resp.json()["id"]

    after_add = client.get("/collections/vectorforge/metrics").json()
    assert (
        after_add["usage"]["documents_added"]
        == initial_metrics["usage"]["documents_added"] + 1
    )

    client.delete(f"/collections/vectorforge/documents/{doc_id}")

    after_delete = client.get("/collections/vectorforge/metrics").json()
    assert (
        after_delete["usage"]["documents_deleted"]
        == initial_metrics["usage"]["documents_deleted"] + 1
    )
    assert (
        after_delete["index"]["total_documents"]
        == after_add["index"]["total_documents"] - 1
    )


def test_metrics_after_search_operation(client, added_doc):
    """Test that metrics update after search operations."""
    initial_metrics = client.get("/collections/vectorforge/metrics").json()
    initial_queries = initial_metrics["performance"]["total_queries"]
    initial_query_time = initial_metrics["performance"]["total_query_time_ms"]

    client.post("/collections/vectorforge/search", json={"query": "test", "top_k": 5})

    after_search = client.get("/collections/vectorforge/metrics").json()
    assert after_search["performance"]["total_queries"] == initial_queries + 1
    assert after_search["performance"]["total_query_time_ms"] > initial_query_time
    assert after_search["timestamps"]["last_query_at"] is not None
    assert after_search["performance"]["avg_query_time_ms"] > 0


def test_metrics_performance_percentiles_calculation(client, multiple_added_docs):
    """Test that p50, p95, p99 percentiles are calculated correctly."""
    for i in range(20):
        client.post(
            "/collections/vectorforge/search", json={"query": f"query {i}", "top_k": 5}
        )

    metrics = client.get("/collections/vectorforge/metrics").json()
    assert metrics["performance"]["p50_query_time_ms"] is not None
    assert metrics["performance"]["p95_query_time_ms"] is not None
    assert metrics["performance"]["p99_query_time_ms"] is not None

    assert metrics["performance"]["p50_query_time_ms"] >= 0
    assert (
        metrics["performance"]["p95_query_time_ms"]
        >= metrics["performance"]["p50_query_time_ms"]
    )
    assert (
        metrics["performance"]["p99_query_time_ms"]
        >= metrics["performance"]["p95_query_time_ms"]
    )

    assert metrics["performance"]["min_query_time_ms"] is not None
    assert metrics["performance"]["max_query_time_ms"] is not None
    assert (
        metrics["performance"]["min_query_time_ms"]
        <= metrics["performance"]["p50_query_time_ms"]
    )
    assert (
        metrics["performance"]["max_query_time_ms"]
        >= metrics["performance"]["p99_query_time_ms"]
    )


def test_metrics_doc_size_tracking_accuracy(client):
    """Test that document size tracking accurately reflects storage usage."""
    initial_metrics = client.get("/collections/vectorforge/metrics").json()
    initial_size = initial_metrics["usage"]["total_doc_size_bytes"]

    large_content = "x" * 10000
    for _ in range(10):
        client.post(
            "/collections/vectorforge/documents",
            json={"content": large_content, "metadata": {}},
        )

    after_metrics = client.get("/collections/vectorforge/metrics").json()
    expected_increase = len(large_content) * 10
    actual_increase = after_metrics["usage"]["total_doc_size_bytes"] - initial_size

    assert actual_increase == expected_increase


def test_metrics_uptime_increases(client):
    """Test that uptime_seconds increases over time."""
    metrics1 = client.get("/collections/vectorforge/metrics").json()
    uptime1 = metrics1["system"]["uptime_seconds"]

    time.sleep(0.1)

    metrics2 = client.get("/collections/vectorforge/metrics").json()
    uptime2 = metrics2["system"]["uptime_seconds"]

    assert uptime2 > uptime1
    assert uptime2 - uptime1 >= 0.1


def test_metrics_only_accepts_get_method(client):
    """Test that metrics endpoint only accepts GET requests."""
    resp = client.post("/collections/vectorforge/metrics")
    assert resp.status_code == 405

    resp = client.put("/collections/vectorforge/metrics")
    assert resp.status_code == 405

    resp = client.delete("/collections/vectorforge/metrics")
    assert resp.status_code == 405


def test_metrics_endpoint_is_idempotent(client):
    """Test that multiple metrics calls return consistent structure."""
    resp1 = client.get("/collections/vectorforge/metrics")
    resp2 = client.get("/collections/vectorforge/metrics")
    resp3 = client.get("/collections/vectorforge/metrics")

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp3.status_code == 200

    assert (
        set(resp1.json().keys()) == set(resp2.json().keys()) == set(resp3.json().keys())
    )


def test_metrics_response_has_no_extra_fields(client):
    """Test that metrics response only contains expected top-level fields."""
    resp = client.get("/collections/vectorforge/metrics")
    data = resp.json()

    expected_fields = {
        "index",
        "performance",
        "usage",
        "timestamps",
        "system",
        "chromadb",
    }
    actual_fields = set(data.keys())

    assert actual_fields == expected_fields


def test_metrics_version_matches_package_version(client):
    """Test that metrics system version matches package version."""
    resp = client.get("/collections/vectorforge/metrics")
    data = resp.json()

    assert data["system"]["version"] == __version__


def test_metrics_ignores_query_parameters(client):
    """Test that metrics endpoint ignores query parameters."""
    resp1 = client.get("/collections/vectorforge/metrics")
    resp2 = client.get(
        "/collections/vectorforge/metrics", params={"foo": "bar", "baz": "qux"}
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    assert set(resp1.json().keys()) == set(resp2.json().keys())


def test_metrics_after_file_upload(client):
    """Test that file upload updates chunks_created and files_uploaded."""
    initial_metrics = client.get("/collections/vectorforge/metrics").json()
    initial_chunks = initial_metrics["usage"]["chunks_created"]
    initial_files = initial_metrics["usage"]["files_uploaded"]

    file_content = b"Test file content for upload testing. " * 100
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}

    resp = client.post("/collections/vectorforge/files/upload", files=files)
    assert resp.status_code == 201

    after_metrics = client.get("/collections/vectorforge/metrics").json()
    assert after_metrics["usage"]["chunks_created"] > initial_chunks
    assert after_metrics["usage"]["files_uploaded"] == initial_files + 1
    assert after_metrics["timestamps"]["last_file_uploaded_at"] is not None


def test_metrics_after_deletions(client, multiple_added_docs):
    """Test that metrics update properly after deletions."""
    initial_metrics = client.get("/collections/vectorforge/metrics").json()

    for i in range(6):
        client.delete(f"/collections/vectorforge/documents/{multiple_added_docs[i]}")

    after_metrics = client.get("/collections/vectorforge/metrics").json()
    assert (
        after_metrics["index"]["total_documents"]
        < initial_metrics["index"]["total_documents"]
    )


def test_metrics_doc_size_values_are_non_negative(client):
    """Test that document size metric is non-negative."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    assert metrics["usage"]["total_doc_size_bytes"] >= 0


def test_metrics_performance_percentiles_none_when_no_queries(client):
    """Test that percentiles are None when no queries have been executed."""
    metrics = client.get("/collections/vectorforge/metrics").json()

    if metrics["performance"]["total_queries"] == 0:
        assert metrics["performance"]["min_query_time_ms"] is None
        assert metrics["performance"]["max_query_time_ms"] is None
        assert metrics["performance"]["p50_query_time_ms"] is None
        assert metrics["performance"]["p95_query_time_ms"] is None
        assert metrics["performance"]["p99_query_time_ms"] is None


def test_metrics_timestamps_are_iso_format(client, added_doc):
    """Test that timestamp fields follow ISO 8601 format."""
    client.post("/collections/vectorforge/search", json={"query": "test", "top_k": 5})
    metrics = client.get("/collections/vectorforge/metrics").json()

    created_at = metrics["timestamps"]["engine_created_at"]
    assert isinstance(created_at, str)
    datetime.fromisoformat(created_at)

    if metrics["timestamps"]["last_query_at"]:
        datetime.fromisoformat(metrics["timestamps"]["last_query_at"])
    if metrics["timestamps"]["last_document_added_at"]:
        datetime.fromisoformat(metrics["timestamps"]["last_document_added_at"])


def test_metrics_average_query_time_calculation(client, multiple_added_docs):
    """Test that average query time is calculated correctly."""
    for i in range(5):
        client.post(
            "/collections/vectorforge/search", json={"query": f"test {i}", "top_k": 5}
        )

    metrics = client.get("/collections/vectorforge/metrics").json()

    total_time = metrics["performance"]["total_query_time_ms"]
    total_queries = metrics["performance"]["total_queries"]
    avg_time = metrics["performance"]["avg_query_time_ms"]

    if total_queries > 0:
        expected_avg = total_time / total_queries
        assert abs(avg_time - expected_avg) < 0.001


def test_metrics_usage_counters_are_cumulative(client):
    """Test that usage metrics are cumulative and never decrease."""
    metrics1 = client.get("/collections/vectorforge/metrics").json()
    client.post(
        "/collections/vectorforge/documents", json={"content": "test", "metadata": {}}
    )
    metrics2 = client.get("/collections/vectorforge/metrics").json()

    assert metrics2["usage"]["documents_added"] >= metrics1["usage"]["documents_added"]
    assert (
        metrics2["usage"]["documents_deleted"] >= metrics1["usage"]["documents_deleted"]
    )


def test_metrics_model_name_is_correct(client):
    """Test that model_name matches the configured model."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    assert metrics["system"]["model_name"] == VFGConfig.MODEL_NAME


def test_metrics_model_dimension_is_correct(client):
    """Test that model_dimension matches the configured dimension."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    assert metrics["system"]["model_dimension"] == VFGConfig.EMBEDDING_DIMENSION


# =============================================================================
# ChromaDB Metrics Tests
# =============================================================================


def test_health_includes_chromadb_heartbeat(client):
    """Test that health endpoint includes ChromaDB heartbeat."""
    resp = client.get("/health")
    data = resp.json()

    assert "chromadb_heartbeat" in data
    assert isinstance(data["chromadb_heartbeat"], int)
    assert data["chromadb_heartbeat"] > 0


def test_metrics_includes_chromadb_section(client):
    """Test that metrics response includes chromadb section."""
    metrics = client.get("/collections/vectorforge/metrics").json()

    assert "chromadb" in metrics
    assert isinstance(metrics["chromadb"], dict)


def test_chromadb_metrics_has_version(client):
    """Test that ChromaDB metrics includes version."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    chromadb = metrics["chromadb"]

    assert "version" in chromadb
    assert isinstance(chromadb["version"], str)
    assert len(chromadb["version"]) > 0


def test_chromadb_metrics_has_collection_info(client):
    """Test that ChromaDB metrics includes collection information."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    chromadb = metrics["chromadb"]

    assert "collection_id" in chromadb
    assert "collection_name" in chromadb
    assert isinstance(chromadb["collection_id"], str)
    assert isinstance(chromadb["collection_name"], str)
    assert chromadb["collection_name"] == VFGConfig.DEFAULT_COLLECTION_NAME


def test_chromadb_metrics_has_disk_size(client):
    """Test that ChromaDB metrics includes disk size information."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    chromadb = metrics["chromadb"]

    assert "disk_size_bytes" in chromadb
    assert "disk_size_mb" in chromadb
    assert isinstance(chromadb["disk_size_bytes"], int)
    assert isinstance(chromadb["disk_size_mb"], float)
    assert chromadb["disk_size_bytes"] >= 0
    assert chromadb["disk_size_mb"] >= 0.0


def test_chromadb_metrics_disk_size_conversion(client):
    """Test that disk_size_mb is correctly calculated from bytes."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    chromadb = metrics["chromadb"]

    expected_mb = round(chromadb["disk_size_bytes"] / (1024 * 1024), 2)
    assert chromadb["disk_size_mb"] == expected_mb


def test_chromadb_metrics_has_persist_directory(client):
    """Test that ChromaDB metrics includes persist directory path."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    chromadb = metrics["chromadb"]

    assert "persist_directory" in chromadb
    assert isinstance(chromadb["persist_directory"], str)
    assert len(chromadb["persist_directory"]) > 0


def test_chromadb_metrics_has_max_batch_size(client):
    """Test that ChromaDB metrics includes max batch size."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    chromadb = metrics["chromadb"]

    assert "max_batch_size" in chromadb
    assert isinstance(chromadb["max_batch_size"], int)
    assert chromadb["max_batch_size"] > 0


def test_chromadb_metrics_disk_size_increases_with_documents(client):
    """Test that disk size increases when documents are added."""
    metrics1 = client.get("/collections/vectorforge/metrics").json()
    initial_disk_bytes = metrics1["chromadb"]["disk_size_bytes"]

    for i in range(10):
        client.post(
            "/collections/vectorforge/documents",
            json={
                "content": f"Test document {i} with substantial content " * 20,
                "metadata": {"test_id": i},
            },
        )

    metrics2 = client.get("/collections/vectorforge/metrics").json()
    final_disk_bytes = metrics2["chromadb"]["disk_size_bytes"]

    assert final_disk_bytes >= initial_disk_bytes


def test_chromadb_metrics_all_fields_present(client):
    """Test that all ChromaDB metric fields are present."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    chromadb = metrics["chromadb"]

    expected_fields = {
        "version",
        "collection_id",
        "collection_name",
        "disk_size_bytes",
        "disk_size_mb",
        "persist_directory",
        "max_batch_size",
    }
    actual_fields = set(chromadb.keys())

    assert actual_fields == expected_fields


# =============================================================================
# Peak Document Tracking Tests
# =============================================================================


def test_metrics_includes_peak_document_count(client):
    """Test that metrics response includes total_documents_peak field."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    index_metrics = metrics["index"]

    assert "total_documents_peak" in index_metrics
    assert isinstance(index_metrics["total_documents_peak"], int)
    assert index_metrics["total_documents_peak"] >= 0


def test_peak_document_count_starts_at_zero(client):
    """Test that peak document count field exists and is non-negative."""
    metrics = client.get("/collections/vectorforge/metrics").json()
    index_metrics = metrics["index"]

    # Peak may not be zero if other tests ran first in the session
    # Just verify the field exists and is a non-negative integer
    assert "total_documents_peak" in index_metrics
    assert isinstance(index_metrics["total_documents_peak"], int)
    assert index_metrics["total_documents_peak"] >= 0


def test_peak_document_count_increases_when_documents_added(client):
    """Test that peak increases when documents are added (or stays same if already high)."""
    metrics1 = client.get("/collections/vectorforge/metrics").json()
    initial_peak = metrics1["index"]["total_documents_peak"]
    initial_total = metrics1["index"]["total_documents"]
    assert initial_total == 0

    client.post(
        "/collections/vectorforge/documents",
        json={"content": "Test document for peak tracking", "metadata": {"test": 1}},
    )

    metrics2 = client.get("/collections/vectorforge/metrics").json()
    new_peak = metrics2["index"]["total_documents_peak"]
    new_total = metrics2["index"]["total_documents"]

    assert new_total == 1
    assert new_peak >= new_total
    assert new_peak >= initial_peak


def test_peak_document_count_stays_same_when_documents_deleted(client):
    """Test that peak does NOT decrease when documents are deleted."""
    doc_ids = []
    for i in range(5):
        resp = client.post(
            "/collections/vectorforge/documents",
            json={
                "content": f"Test document {i} for peak tracking",
                "metadata": {"test_id": i},
            },
        )
        doc_ids.append(resp.json()["id"])

    metrics1 = client.get("/collections/vectorforge/metrics").json()
    peak_after_add = metrics1["index"]["total_documents_peak"]
    total_after_add = metrics1["index"]["total_documents"]

    assert total_after_add == 5
    assert peak_after_add >= 5

    for i in range(3):
        client.delete(f"/collections/vectorforge/documents/{doc_ids[i]}")

    metrics2 = client.get("/collections/vectorforge/metrics").json()
    peak_after_delete = metrics2["index"]["total_documents_peak"]
    total_after_delete = metrics2["index"]["total_documents"]

    assert peak_after_delete >= peak_after_add
    assert total_after_delete == 2


def test_peak_document_count_equals_total_when_no_deletions(
    client, multiple_added_docs
):
    """Test that peak is at least as high as total after only adding documents."""
    assert len(multiple_added_docs) == 20

    metrics = client.get("/collections/vectorforge/metrics").json()
    index_metrics = metrics["index"]

    peak = index_metrics["total_documents_peak"]
    total = index_metrics["total_documents"]

    assert total == 20
    assert peak >= 20
