"""Tests for the collections CRUD endpoints.

Covers:
    GET    /collections
    POST   /collections
    GET    /collections/{name}
    DELETE /collections/{name}?confirm=true
"""

import io
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from vectorforge.config import VFGConfig

# =============================================================================
# Multi-collection isolation fixtures
# =============================================================================


@pytest.fixture
def col_a(client: TestClient) -> str:
    """Create isolation_col_a and return its name."""
    client.post("/collections", json={"name": "isolation_col_a"})
    return "isolation_col_a"


@pytest.fixture
def col_b(client: TestClient) -> str:
    """Create isolation_col_b and return its name."""
    client.post("/collections", json={"name": "isolation_col_b"})
    return "isolation_col_b"


# =============================================================================
# GET /collections — list collections
# =============================================================================


def test_list_collections_returns_default(client: TestClient) -> None:
    """Default collection always exists; list should report total=1."""
    response = client.get("/collections")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 1
    assert len(data["collections"]) == 1
    assert (
        data["collections"][0]["collection_name"] == VFGConfig.DEFAULT_COLLECTION_NAME
    )


def test_list_collections_includes_new_collection(client: TestClient) -> None:
    """Creating a collection causes it to appear in the list."""
    client.post("/collections", json={"name": "list_test_col"})

    response = client.get("/collections")

    assert response.status_code == 200
    data = response.json()
    names = [c["collection_name"] for c in data["collections"]]
    assert "list_test_col" in names
    assert data["total"] == 2


def test_list_collections_response_shape(client: TestClient) -> None:
    """Each collection entry has the expected fields."""
    response = client.get("/collections")

    assert response.status_code == 200
    col = response.json()["collections"][0]
    expected_fields = [
        "collection_name",
        "id",
        "document_count",
        "created_at",
        "hnsw_config",
    ]

    for field in expected_fields:
        assert field in col


# =============================================================================
# POST /collections — create collection
# =============================================================================


def test_create_collection_returns_201(client: TestClient) -> None:
    """Creating a valid collection returns HTTP 201."""
    response = client.post("/collections", json={"name": "new_col"})
    assert response.status_code == 201


def test_create_collection_response_shape(client: TestClient) -> None:
    """Create response contains status, message, and collection info."""
    response = client.post("/collections", json={"name": "shape_col"})
    data = response.json()

    assert data["status"] == "success"
    assert "created successfully" in data["message"]
    assert data["collection"]["collection_name"] == "shape_col"
    assert data["collection"]["document_count"] == 0


def test_create_collection_with_description(client: TestClient) -> None:
    """Optional description is stored and returned."""
    payload = {"name": "desc_col", "description": "A test collection"}
    response = client.post("/collections", json=payload)

    assert response.status_code == 201
    assert response.json()["collection"]["description"] == "A test collection"


def test_create_collection_with_hnsw_config(client: TestClient) -> None:
    """Custom HNSW config is applied and reflected in the response."""
    payload = {
        "name": "hnsw_col",
        "hnsw_config": {"ef_search": 200, "max_neighbors": 32},
    }
    response = client.post("/collections", json=payload)

    assert response.status_code == 201
    hnsw = response.json()["collection"]["hnsw_config"]
    assert hnsw["ef_search"] == 200
    assert hnsw["max_neighbors"] == 32


def test_create_collection_with_metadata(client: TestClient) -> None:
    """Custom metadata is stored and returned."""
    payload = {"name": "meta_col", "metadata": {"tenant": "acme"}}
    response = client.post("/collections", json=payload)

    assert response.status_code == 201
    assert response.json()["collection"]["metadata"]["tenant"] == "acme"


def test_create_duplicate_collection_returns_409(client: TestClient) -> None:
    """Creating a collection with an existing name returns HTTP 409."""
    client.post("/collections", json={"name": "dup_col"})
    response = client.post("/collections", json={"name": "dup_col"})
    assert response.status_code == 409


def test_create_collection_invalid_name_returns_422(client: TestClient) -> None:
    """Names with special characters fail Pydantic validation (HTTP 422)."""
    response = client.post("/collections", json={"name": "invalid name!"})
    assert response.status_code == 422


def test_create_collection_empty_name_returns_422(client: TestClient) -> None:
    """Empty name fails minimum-length validation (HTTP 422)."""
    response = client.post("/collections", json={"name": ""})
    assert response.status_code == 422


def test_create_collection_name_too_long_returns_422(client: TestClient) -> None:
    """Name exceeding max length fails Pydantic validation (HTTP 422)."""
    long_name = "a" * (VFGConfig.MAX_COLLECTION_NAME_LENGTH + 1)
    response = client.post("/collections", json={"name": long_name})
    assert response.status_code == 422


# =============================================================================
# GET /collections/{name} — get single collection
# =============================================================================


def test_get_default_collection(client: TestClient) -> None:
    """Default collection is always retrievable."""
    response = client.get(f"/collections/{VFGConfig.DEFAULT_COLLECTION_NAME}")

    assert response.status_code == 200
    data = response.json()
    assert data["collection_name"] == VFGConfig.DEFAULT_COLLECTION_NAME


def test_get_collection_response_shape(client: TestClient) -> None:
    """Get-by-name response includes all CollectionInfo fields."""
    response = client.get(f"/collections/{VFGConfig.DEFAULT_COLLECTION_NAME}")

    data = response.json()
    expected_fields = [
        "collection_name",
        "id",
        "document_count",
        "created_at",
        "hnsw_config",
    ]

    for field in expected_fields:
        assert field in data


def test_get_created_collection(client: TestClient) -> None:
    """A freshly created collection can be retrieved by name."""
    client.post("/collections", json={"name": "get_test_col"})
    response = client.get("/collections/get_test_col")
    assert response.status_code == 200
    assert response.json()["collection_name"] == "get_test_col"


def test_get_nonexistent_collection_returns_404(client: TestClient) -> None:
    """Requesting an unknown collection returns HTTP 404."""
    response = client.get("/collections/does_not_exist")
    assert response.status_code == 404


# =============================================================================
# DELETE /collections/{name} — delete collection
# =============================================================================


def test_delete_collection_success(client: TestClient) -> None:
    """Deleting an existing non-default collection with confirm=true returns 200."""
    client.post("/collections", json={"name": "del_col"})
    response = client.delete("/collections/del_col", params={"confirm": True})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["collection_name"] == "del_col"


def test_delete_collection_removes_it_from_list(client: TestClient) -> None:
    """After deletion the collection no longer appears in the list."""
    client.post("/collections", json={"name": "gone_col"})
    client.delete("/collections/gone_col?confirm=true")

    response = client.get("/collections")
    names = [c["collection_name"] for c in response.json()["collections"]]
    assert "gone_col" not in names


def test_delete_collection_without_confirm_returns_400(client: TestClient) -> None:
    """Omitting ?confirm=true is rejected with HTTP 400."""
    client.post("/collections", json={"name": "no_confirm_col"})
    response = client.delete("/collections/no_confirm_col")
    assert response.status_code == 400


def test_delete_collection_confirm_false_returns_400(client: TestClient) -> None:
    """Passing ?confirm=false is rejected with HTTP 400."""
    client.post("/collections", json={"name": "false_confirm_col"})
    response = client.delete(
        "/collections/false_confirm_col", params={"confirm": False}
    )
    assert response.status_code == 400


def test_delete_nonexistent_collection_returns_404(client: TestClient) -> None:
    """Deleting an unknown collection returns HTTP 404."""
    response = client.delete("/collections/ghost_col?confirm=true")
    assert response.status_code == 404


def test_delete_default_collection_returns_409(client: TestClient) -> None:
    """Attempting to delete the default collection returns HTTP 409."""
    response = client.delete(
        f"/collections/{VFGConfig.DEFAULT_COLLECTION_NAME}?confirm=true"
    )
    assert response.status_code == 409


# =============================================================================
# GET /collections — extended list behaviour
# =============================================================================


def test_list_collections_alphabetical_order(client: TestClient) -> None:
    """List response is sorted alphabetically by collection_name."""
    client.post("/collections", json={"name": "zebra_col"})
    client.post("/collections", json={"name": "apple_col"})

    response = client.get("/collections")
    names = [c["collection_name"] for c in response.json()["collections"]]

    assert names == sorted(names)


def test_list_collections_total_matches_length(client: TestClient) -> None:
    """total field always equals len(collections) after multiple creates."""
    client.post("/collections", json={"name": "count_col_1"})
    client.post("/collections", json={"name": "count_col_2"})
    client.post("/collections", json={"name": "count_col_3"})
    data = client.get("/collections").json()

    assert data["total"] == len(data["collections"])
    assert data["total"] == 4


def test_list_collections_document_count_reflects_added_docs(
    client: TestClient,
) -> None:
    """document_count in list response reflects documents added to that collection."""
    client.post("/collections", json={"name": "count_docs_col"})
    client.post(
        "/collections/count_docs_col/documents",
        json={"content": "first doc", "metadata": {}},
    )
    client.post(
        "/collections/count_docs_col/documents",
        json={"content": "second doc", "metadata": {}},
    )

    data = client.get("/collections").json()
    col = next(
        c for c in data["collections"] if c["collection_name"] == "count_docs_col"
    )

    assert col["document_count"] == 2


def test_list_collections_optional_fields_present_when_empty(
    client: TestClient,
) -> None:
    """description and metadata are present in every entry even when not set."""
    client.post("/collections", json={"name": "empty_fields_col"})
    data = client.get("/collections").json()
    col = next(
        c for c in data["collections"] if c["collection_name"] == "empty_fields_col"
    )

    assert "description" in col
    assert "metadata" in col
    assert col["description"] is None
    assert col["metadata"] == {}


# =============================================================================
# POST /collections — extended create behaviour
# =============================================================================


def test_create_collection_max_length_name_succeeds(client: TestClient) -> None:
    """Name at exactly MAX_COLLECTION_NAME_LENGTH characters is accepted."""
    max_name = "a" * VFGConfig.MAX_COLLECTION_NAME_LENGTH
    response = client.post("/collections", json={"name": max_name})
    assert response.status_code == 201


def test_create_collection_min_length_name_succeeds(client: TestClient) -> None:
    """Name at exactly MIN_COLLECTION_NAME_LENGTH characters is accepted.

    Note: ChromaDB itself requires at least 3 characters, so the effective
    minimum accepted end-to-end is 3, regardless of VFGConfig.MIN_COLLECTION_NAME_LENGTH.
    """
    response = client.post("/collections", json={"name": "abc"})
    assert response.status_code == 201


def test_create_collection_name_with_hyphens_and_underscores(
    client: TestClient,
) -> None:
    """Names containing hyphens and underscores are valid."""
    response = client.post("/collections", json={"name": "my-col_1"})
    assert response.status_code == 201
    assert response.json()["collection"]["collection_name"] == "my-col_1"


def test_create_collection_hnsw_defaults_applied_when_not_specified(
    client: TestClient,
) -> None:
    """Default HNSW config is applied when no hnsw_config is supplied."""
    response = client.post("/collections", json={"name": "default_hnsw_col"})

    hnsw = response.json()["collection"]["hnsw_config"]
    assert hnsw["space"] == "cosine"
    assert hnsw["ef_construction"] == 100
    assert hnsw["ef_search"] == 100
    assert hnsw["max_neighbors"] == 16
    assert hnsw["resize_factor"] == 1.2
    assert hnsw["sync_threshold"] == 1000


def test_create_collection_partial_hnsw_config_fills_remaining_defaults(
    client: TestClient,
) -> None:
    """Specifying only some HNSW fields leaves remaining fields at defaults."""
    payload = {"name": "partial_hnsw_col", "hnsw_config": {"ef_search": 200}}
    response = client.post("/collections", json=payload)

    hnsw = response.json()["collection"]["hnsw_config"]
    assert hnsw["ef_search"] == 200

    # Defaults:
    assert hnsw["ef_construction"] == 100
    assert hnsw["max_neighbors"] == 16
    assert hnsw["space"] == "cosine"


def test_create_collection_metadata_round_trips(client: TestClient) -> None:
    """Metadata key/value pairs survive the create → response round-trip."""
    payload = {
        "name": "roundtrip_meta_col",
        "metadata": {"env": "staging", "owner": "alice"},
    }
    response = client.post("/collections", json=payload)

    meta = response.json()["collection"]["metadata"]
    assert meta["env"] == "staging"
    assert meta["owner"] == "alice"


def test_create_collection_metadata_values_coerced_to_string(
    client: TestClient,
) -> None:
    """Numeric metadata values are stored and returned as strings."""
    payload = {"name": "coerce_meta_col", "metadata": {"count": 42, "ratio": 3.14}}
    response = client.post("/collections", json=payload)

    meta = response.json()["collection"]["metadata"]
    assert meta["count"] == "42"
    assert meta["ratio"] == "3.14"


def test_create_collection_too_many_metadata_pairs_returns_400(
    client: TestClient,
) -> None:
    """Metadata with more than MAX_METADATA_PAIRS entries is rejected with 400."""
    too_many = {f"key_{i}": f"val_{i}" for i in range(VFGConfig.MAX_METADATA_PAIRS + 1)}
    response = client.post(
        "/collections", json={"name": "overflow_meta_col", "metadata": too_many}
    )
    assert response.status_code == 400


def test_create_collection_description_at_max_length_succeeds(
    client: TestClient,
) -> None:
    """Description at exactly 500 characters is accepted."""
    response = client.post(
        "/collections",
        json={"name": "max_desc_col", "description": "x" * 500},
    )
    assert response.status_code == 201


def test_create_collection_description_over_max_length_returns_422(
    client: TestClient,
) -> None:
    """Description exceeding 500 characters is rejected with 422."""
    response = client.post(
        "/collections",
        json={"name": "long_desc_col", "description": "x" * 501},
    )
    assert response.status_code == 422


def test_create_collection_message_format(client: TestClient) -> None:
    """Create response message follows the exact expected format."""
    response = client.post("/collections", json={"name": "msg_col"})
    assert response.json()["message"] == "Collection 'msg_col' created successfully"


# =============================================================================
# GET /collections/{name} — extended single-collection behaviour
# =============================================================================


def test_get_collection_document_count_updates_after_add(
    client: TestClient,
) -> None:
    """document_count reflects documents added after collection creation."""
    client.post("/collections", json={"name": "live_count_col"})
    client.post(
        "/collections/live_count_col/documents",
        json={"content": "hello world", "metadata": {}},
    )

    data = client.get("/collections/live_count_col").json()
    assert data["document_count"] == 1


def test_get_collection_created_at_is_valid_iso_timestamp(
    client: TestClient,
) -> None:
    """created_at on a freshly created collection is a valid ISO 8601 timestamp."""
    client.post("/collections", json={"name": "ts_col"})

    data = client.get("/collections/ts_col").json()

    # Should not raise
    parsed = datetime.fromisoformat(data["created_at"])
    assert parsed.year >= 2025


def test_get_collection_metadata_round_trips(client: TestClient) -> None:
    """Metadata set at create time is returned unchanged by GET."""
    client.post(
        "/collections",
        json={"name": "get_meta_col", "metadata": {"env": "prod", "region": "us-east"}},
    )

    data = client.get("/collections/get_meta_col").json()
    assert data["metadata"]["env"] == "prod"
    assert data["metadata"]["region"] == "us-east"


def test_get_collection_hnsw_config_round_trips(client: TestClient) -> None:
    """HNSW config set at create time is returned unchanged by GET."""
    client.post(
        "/collections",
        json={
            "name": "get_hnsw_col",
            "hnsw_config": {"ef_search": 150, "max_neighbors": 24},
        },
    )

    data = client.get("/collections/get_hnsw_col").json()
    assert data["hnsw_config"]["ef_search"] == 150
    assert data["hnsw_config"]["max_neighbors"] == 24


def test_get_collection_description_round_trips(client: TestClient) -> None:
    """Description set at create time is returned unchanged by GET."""
    client.post(
        "/collections",
        json={"name": "get_desc_col", "description": "My test collection"},
    )

    data = client.get("/collections/get_desc_col").json()
    assert data["description"] == "My test collection"


def test_get_collection_hnsw_config_has_all_fields(client: TestClient) -> None:
    """CollectionInfo.hnsw_config contains all six expected fields."""
    client.post("/collections", json={"name": "hnsw_fields_col"})

    hnsw = client.get("/collections/hnsw_fields_col").json()["hnsw_config"]
    expected_fields = [
        "space",
        "ef_construction",
        "ef_search",
        "max_neighbors",
        "resize_factor",
        "sync_threshold",
    ]

    for field in expected_fields:
        assert field in hnsw, f"Missing hnsw_config field: {field}"


def test_get_collection_id_is_non_empty_string(client: TestClient) -> None:
    """CollectionInfo.id is a non-empty string."""
    client.post("/collections", json={"name": "id_check_col"})

    data = client.get("/collections/id_check_col").json()
    assert isinstance(data["id"], str)
    assert len(data["id"]) > 0


# =============================================================================
# DELETE /collections/{name} — extended delete behaviour
# =============================================================================


def test_delete_collection_message_format(client: TestClient) -> None:
    """Delete response message follows the exact expected format."""
    client.post("/collections", json={"name": "msg_del_col"})
    data = client.delete("/collections/msg_del_col?confirm=true").json()
    assert data["message"] == "Collection 'msg_del_col' deleted successfully"


def test_delete_collection_returns_404_on_subsequent_get(
    client: TestClient,
) -> None:
    """GET on a deleted collection returns 404."""
    client.post("/collections", json={"name": "del_then_get_col"})
    client.delete("/collections/del_then_get_col?confirm=true")

    response = client.get("/collections/del_then_get_col")
    assert response.status_code == 404


def test_delete_one_collection_does_not_affect_another(
    client: TestClient,
) -> None:
    """Deleting col_b leaves col_a intact and retrievable."""
    client.post("/collections", json={"name": "survivor_col"})
    client.post("/collections", json={"name": "victim_col"})
    client.post(
        "/collections/survivor_col/documents",
        json={"content": "I should survive", "metadata": {}},
    )

    client.delete("/collections/victim_col?confirm=true")

    response = client.get("/collections/survivor_col")
    assert response.status_code == 200
    assert response.json()["document_count"] == 1


# =============================================================================
# Multi-collection isolation tests
# =============================================================================


def test_document_isolation_between_collections(
    client: TestClient, col_a: str, col_b: str
) -> None:
    """Documents added to col_a are not visible in col_b's document count."""
    client.post(
        f"/collections/{col_a}/documents",
        json={"content": "exclusive to col_a", "metadata": {}},
    )

    col_b_info = client.get(f"/collections/{col_b}").json()
    assert col_b_info["document_count"] == 0


def test_search_isolation_between_collections(
    client: TestClient, col_a: str, col_b: str
) -> None:
    """Search in col_b returns no results when only col_a has documents."""
    client.post(
        f"/collections/{col_a}/documents",
        json={"content": "machine learning and neural networks", "metadata": {}},
    )

    response = client.post(
        f"/collections/{col_b}/search",
        json={"query": "machine learning and neural networks", "top_k": 5},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 0
    assert response.json()["results"] == []


def test_stats_isolation_between_collections(
    client: TestClient, col_a: str, col_b: str
) -> None:
    """Stats endpoint reports independent document counts per collection."""
    for i in range(3):
        client.post(
            f"/collections/{col_a}/documents",
            json={"content": f"col_a document {i}", "metadata": {}},
        )

    stats_a = client.get(f"/collections/{col_a}/stats").json()
    stats_b = client.get(f"/collections/{col_b}/stats").json()

    assert stats_a["total_documents"] == 3
    assert stats_b["total_documents"] == 0


def test_metrics_isolation_between_collections(
    client: TestClient, col_a: str, col_b: str
) -> None:
    """Metrics endpoint reports independent document counts per collection."""
    for i in range(2):
        client.post(
            f"/collections/{col_a}/documents",
            json={"content": f"col_a metrics doc {i}", "metadata": {}},
        )

    metrics_a = client.get(f"/collections/{col_a}/metrics").json()
    metrics_b = client.get(f"/collections/{col_b}/metrics").json()

    assert metrics_a["index"]["total_documents"] == 2
    assert metrics_b["index"]["total_documents"] == 0


def test_file_isolation_between_collections(
    client: TestClient, col_a: str, col_b: str
) -> None:
    """Files uploaded to col_a do not appear in col_b's file list."""
    file_content = b"isolation test file content"
    client.post(
        f"/collections/{col_a}/files/upload",
        files={"file": ("isolation_test.txt", io.BytesIO(file_content), "text/plain")},
    )

    response = client.get(f"/collections/{col_b}/files/list")
    assert response.status_code == 200
    assert "isolation_test.txt" not in response.json()["filenames"]


def test_hnsw_config_isolation_between_collections(
    client: TestClient, col_a: str, col_b: str
) -> None:
    """Updating HNSW config on col_a does not change col_b's config."""
    client.put(
        f"/collections/{col_a}/config/hnsw?confirm=true",
        json={"ef_search": 250},
    )

    col_b_hnsw = client.get(f"/collections/{col_b}").json()["hnsw_config"]
    assert col_b_hnsw["ef_search"] == 100


def test_delete_collection_does_not_bleed_into_sibling(
    client: TestClient, col_a: str, col_b: str
) -> None:
    """Deleting col_b leaves col_a's documents fully intact."""
    for i in range(5):
        client.post(
            f"/collections/{col_a}/documents",
            json={"content": f"col_a survivor doc {i}", "metadata": {}},
        )
    for i in range(5):
        client.post(
            f"/collections/{col_b}/documents",
            json={"content": f"col_b doomed doc {i}", "metadata": {}},
        )

    client.delete(f"/collections/{col_b}?confirm=true")

    stats_a = client.get(f"/collections/{col_a}/stats").json()
    assert stats_a["total_documents"] == 5
