"""Tests for system monitoring endpoints"""

import re
import time

import pytest

from vectorforge import __version__


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def metrics(client):
    """Reusable fixture returning index metrics."""
    resp = client.get("/metrics")
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
    semver_pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$'
    assert re.match(semver_pattern, version)
    
    parts = version.split('-')[0].split('+')[0].split('.')
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
    
    assert resp1.json() == resp2.json() == resp3.json()
    assert all(r.status_code == 200 for r in [resp1, resp2, resp3])


def test_health_version_matches_package_version(client):
    """Test that health version matches the actual package version."""
    
    resp = client.get("/health")
    data = resp.json()
    
    assert data["version"] == __version__


def test_health_response_has_no_extra_fields(client):
    """Test that health response only contains expected fields."""
    resp = client.get("/health")
    data = resp.json()
    
    expected_fields = {"status", "version"}
    actual_fields = set(data.keys())
    
    assert actual_fields == expected_fields


def test_health_ignores_query_parameters(client):
    """Test that health endpoint ignores any query parameters."""
    resp1 = client.get("/health")
    resp2 = client.get("/health", params={
        "foo": "bar",
        "baz": "qux"
    })
    
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json() == resp2.json()


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
    """Test that GET /metrics returns 200 status."""
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_returns_comprehensive_data(metrics):
    """Test that metrics response includes all metric categories."""
    assert "index" in metrics
    assert "performance" in metrics
    assert "usage" in metrics
    assert "memory" in metrics
    assert "timestamps" in metrics
    assert "system" in metrics


def test_metrics_includes_index_metrics(metrics):
    """Test that metrics response includes index statistics."""
    assert "total_documents" in metrics["index"]
    assert isinstance(metrics["index"]["total_documents"], int)
    assert "total_embeddings" in metrics["index"]
    assert isinstance(metrics["index"]["total_embeddings"], int)
    assert "deleted_documents" in metrics["index"]
    assert isinstance(metrics["index"]["deleted_documents"], int)
    assert "deleted_ratio" in metrics["index"]
    assert isinstance(metrics["index"]["deleted_ratio"], float)
    assert "needs_compaction" in metrics["index"]
    assert isinstance(metrics["index"]["needs_compaction"], bool)
    assert "compact_threshold" in metrics["index"]
    assert isinstance(metrics["index"]["compact_threshold"], float)


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
    assert "compactions_performed" in metrics["usage"]
    assert isinstance(metrics["usage"]["compactions_performed"], int)
    assert "chunks_created" in metrics["usage"]
    assert isinstance(metrics["usage"]["chunks_created"], int)
    assert "files_uploaded" in metrics["usage"]
    assert isinstance(metrics["usage"]["files_uploaded"], int)


def test_metrics_includes_memory_metrics(metrics):
    """Test that metrics response includes memory statistics."""
    assert "embeddings_mb" in metrics["memory"]
    assert isinstance(metrics["memory"]["embeddings_mb"], float)
    assert "documents_mb" in metrics["memory"]
    assert isinstance(metrics["memory"]["documents_mb"], float)
    assert "total_mb" in metrics["memory"]
    assert isinstance(metrics["memory"]["total_mb"], float)


def test_metrics_includes_timestamp_metrics(metrics):
    """Test that metrics response includes timestamp information."""
    assert "engine_created_at" in metrics["timestamps"]
    assert isinstance(metrics["timestamps"]["engine_created_at"], str)
    assert "last_query_at" in metrics["timestamps"]
    assert isinstance(metrics["timestamps"]["last_query_at"], (str | None))
    assert "last_document_added_at" in metrics["timestamps"]
    assert isinstance(metrics["timestamps"]["last_document_added_at"], (str | None))
    assert "last_compaction_at" in metrics["timestamps"]
    assert isinstance(metrics["timestamps"]["last_compaction_at"], (str | None))
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
    initial_metrics = client.get("/metrics").json()
    
    client.post("/doc/add", json={"content": "test document", "metadata": {}})
    
    after_add = client.get("/metrics").json()
    assert after_add["usage"]["documents_added"] == initial_metrics["usage"]["documents_added"] + 1
    assert after_add["index"]["total_documents"] == initial_metrics["index"]["total_documents"] + 1
    assert after_add["timestamps"]["last_document_added_at"] is not None


def test_metrics_after_add_delete_cycle(client, sample_doc, multiple_added_docs):
    """Test metrics accuracy after adding and deleting documents."""
    initial_metrics = client.get("/metrics").json()
    
    add_resp = client.post("/doc/add", json=sample_doc)
    doc_id = add_resp.json()["id"]
    
    after_add = client.get("/metrics").json()
    assert after_add["usage"]["documents_added"] == initial_metrics["usage"]["documents_added"] + 1
    
    client.delete(f"/doc/{doc_id}")
    
    after_delete = client.get("/metrics").json()
    assert after_delete["usage"]["documents_deleted"] == initial_metrics["usage"]["documents_deleted"] + 1
    assert after_delete["index"]["deleted_documents"] == initial_metrics["index"]["deleted_documents"] + 1
    
    assert after_delete["index"]["total_embeddings"] == after_add["index"]["total_embeddings"]


def test_metrics_after_search_operation(client, added_doc):
    """Test that metrics update after search operations."""
    initial_metrics = client.get("/metrics").json()
    initial_queries = initial_metrics["performance"]["total_queries"]
    initial_query_time = initial_metrics["performance"]["total_query_time_ms"]
    
    client.post("/search", json={
        "query": "test", 
        "top_k": 5
    })
    
    after_search = client.get("/metrics").json()
    assert after_search["performance"]["total_queries"] == initial_queries + 1
    assert after_search["performance"]["total_query_time_ms"] > initial_query_time
    assert after_search["timestamps"]["last_query_at"] is not None
    assert after_search["performance"]["avg_query_time_ms"] > 0


def test_metrics_performance_percentiles_calculation(client, multiple_added_docs):
    """Test that p50, p95, p99 percentiles are calculated correctly."""
    for i in range(20):
        client.post("/search", json={"query": f"query {i}", "top_k": 5})
    
    metrics = client.get("/metrics").json()
    assert metrics["performance"]["p50_query_time_ms"] is not None
    assert metrics["performance"]["p95_query_time_ms"] is not None
    assert metrics["performance"]["p99_query_time_ms"] is not None
    
    assert metrics["performance"]["p50_query_time_ms"] >= 0
    assert metrics["performance"]["p95_query_time_ms"] >= metrics["performance"]["p50_query_time_ms"]
    assert metrics["performance"]["p99_query_time_ms"] >= metrics["performance"]["p95_query_time_ms"]
    
    assert metrics["performance"]["min_query_time_ms"] is not None
    assert metrics["performance"]["max_query_time_ms"] is not None
    assert metrics["performance"]["min_query_time_ms"] <= metrics["performance"]["p50_query_time_ms"]
    assert metrics["performance"]["max_query_time_ms"] >= metrics["performance"]["p99_query_time_ms"]


def test_metrics_memory_calculation_accuracy(client):
    """Test that memory metrics accurately reflect storage usage."""
    initial_metrics = client.get("/metrics").json()
    initial_memory = initial_metrics["memory"]["total_mb"]
    
    large_content = "x" * 10000  # 10KB of content
    for i in range(10):
        client.post("/doc/add", json={"content": large_content, "metadata": {}})
    
    after_metrics = client.get("/metrics").json()
    assert after_metrics["memory"]["total_mb"] > initial_memory
    assert after_metrics["memory"]["documents_mb"] > 0
    assert after_metrics["memory"]["embeddings_mb"] > 0
    
    expected_total = after_metrics["memory"]["embeddings_mb"] + after_metrics["memory"]["documents_mb"]
    assert abs(after_metrics["memory"]["total_mb"] - expected_total) < 0.001


def test_metrics_index_deleted_ratio_calculation(client, multiple_added_docs):
    """Test that deleted_ratio is calculated correctly."""
    metrics_before = client.get("/metrics").json()
    initial_deleted = metrics_before["index"]["deleted_documents"]
    initial_ratio = metrics_before["index"]["deleted_ratio"]
    
    client.delete(f"/doc/{multiple_added_docs[0]}")
    client.delete(f"/doc/{multiple_added_docs[1]}")
    
    metrics_after = client.get("/metrics").json()
    assert metrics_after["index"]["deleted_documents"] == initial_deleted + 2
    assert metrics_after["index"]["deleted_ratio"] > initial_ratio
    
    if metrics_after["index"]["deleted_ratio"] > metrics_after["index"]["compact_threshold"]:
        assert metrics_after["index"]["needs_compaction"] is True
    else:
        assert metrics_after["index"]["needs_compaction"] is False


def test_metrics_uptime_increases(client):
    """Test that uptime_seconds increases over time."""
    metrics1 = client.get("/metrics").json()
    uptime1 = metrics1["system"]["uptime_seconds"]
    
    time.sleep(0.1)
    
    metrics2 = client.get("/metrics").json()
    uptime2 = metrics2["system"]["uptime_seconds"]
    
    assert uptime2 > uptime1
    assert uptime2 - uptime1 >= 0.1
