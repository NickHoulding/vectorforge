"""Tests for the collections CRUD endpoints.

Covers:
    GET    /collections
    POST   /collections
    GET    /collections/{name}
    DELETE /collections/{name}?confirm=true
"""

from fastapi.testclient import TestClient

from vectorforge.config import VFGConfig

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
