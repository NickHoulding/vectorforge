"""Tests for document management endpoints"""

from vectorforge.config import VFGConfig

# =============================================================================
# Document Endpoint Tests
# =============================================================================


def test_doc_add_returns_unique_ids_for_multiple_docs(client, sample_doc):
    """Test that adding multiple documents generates unique IDs for each."""
    first_response = client.post("/doc/add", json=sample_doc)
    first_id = first_response.json()["id"]

    second_response = client.post("/doc/add", json=sample_doc)
    second_id = second_response.json()["id"]

    assert first_id != second_id


def test_doc_add_null_metadata(client, sample_doc):
    """Test that adding a document with null metadata returns 400."""
    sample_doc["metadata"] = None
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_empty_metadata(client, sample_doc):
    """Test that adding a document with empty metadata dict succeeds."""
    sample_doc["metadata"] = {}
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "id" in response_data
    assert response_data["status"] == "indexed"


def test_doc_add_large_content(client, sample_doc):
    """Test that adding a document exceeding max content length returns 422."""
    sample_doc["content"] = "a" * (VFGConfig.MAX_CONTENT_LENGTH + 1)
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 422


def test_doc_add_empty_content(client, sample_doc):
    """Test that adding a document with empty content returns 400."""
    sample_doc["content"] = ""
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 422


def test_doc_get_returns_200_with_valid_id(client, added_doc):
    """Test that GET /doc/{id} returns 200 for existing document."""
    response = client.get(f"/doc/{added_doc['id']}")
    assert response.status_code == 200


def test_doc_get_returns_correct_id(client, added_doc):
    """Test that retrieved document has correct ID."""
    response = client.get(f"/doc/{added_doc['id']}")
    data = response.json()
    assert data["id"] == added_doc["id"]


def test_doc_get_returns_matching_content(client, sample_doc):
    """Test that retrieved document content matches original."""
    add_response = client.post("/doc/add", json=sample_doc)
    doc_id = add_response.json()["id"]

    get_response = client.get(f"/doc/{doc_id}")
    data = get_response.json()
    assert data["content"] == sample_doc["content"]


def test_doc_get_preserves_metadata(client, sample_doc):
    """Test that retrieved document preserves metadata."""
    add_response = client.post("/doc/add", json=sample_doc)
    doc_id = add_response.json()["id"]
    get_response = client.get(f"/doc/{doc_id}")

    metadata = get_response.json()["metadata"]
    assert metadata["source_file"] == "test.txt"
    assert metadata["chunk_index"] == 0


def test_doc_get_returns_404_when_not_found(client):
    """Test 404 response when retrieving a non-existent document."""
    response = client.get("/doc/nonexistent-id")
    assert response.status_code == 404


def test_doc_add_returns_201(client, sample_doc):
    """Test that POST /doc/add returns 201 status."""
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_returns_document_id(client, sample_doc):
    """Test that POST /doc/add returns a document ID."""
    response = client.post("/doc/add", json=sample_doc)
    data = response.json()
    assert "id" in data
    assert isinstance(data["id"], str)
    assert len(data["id"]) > 0


def test_doc_add_creates_document_with_id(client, sample_doc):
    """Test that POST /doc/add returns indexed status."""
    response = client.post("/doc/add", json=sample_doc)
    data = response.json()
    assert data["status"] == "indexed"


def test_doc_delete_returns_200(client, added_doc):
    """Test that DELETE /doc/{id} returns 200 status."""
    response = client.delete(f"/doc/{added_doc['id']}")
    assert response.status_code == 200


def test_doc_delete_returns_deleted_status(client, added_doc):
    """Test that DELETE /doc/{id} returns deleted status."""
    response = client.delete(f"/doc/{added_doc['id']}")
    data = response.json()
    assert data["status"] == "deleted"


def test_doc_delete_removes_from_index(client, added_doc):
    """Test that deleted document is no longer retrievable."""
    doc_id = added_doc["id"]
    client.delete(f"/doc/{doc_id}")

    get_response = client.get(f"/doc/{doc_id}")
    assert get_response.status_code == 404


def test_doc_delete_returns_404_when_not_found(client):
    """Test 404 response when deleting a non-existent document."""
    response = client.delete("/doc/nonexistent-id")
    assert response.status_code == 404


def test_doc_add_metadata_with_only_source_file(client, sample_doc):
    """Test that metadata with only 'source_file' (missing 'chunk_index') returns 400."""
    del sample_doc["metadata"]["chunk_index"]
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 400


def test_doc_add_metadata_with_only_chunk_index(client, sample_doc):
    """Test that metadata with only 'chunk_index' (missing 'source_file') returns 400."""
    del sample_doc["metadata"]["source_file"]
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 400


def test_doc_add_with_special_characters_in_content(client, sample_doc):
    """Test that documents with special characters in content are accepted."""
    sample_doc["metadata"]["content"] = "!@#$%^&*()_~-=+,.<>/?;:\"'[]|\\"
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_with_unicode_content(client, sample_doc):
    """Test that documents with unicode characters are properly handled."""
    sample_doc["content"] = (
        "Hello 世界 🌍 "  # Chinese + emoji
        "Héllo Wörld "  # Accented Latin
        "Привет мир "  # Cyrillic
        "مرحبا بالعالم "  # Arabic
        "שלום עולם "  # Hebrew
        "こんにちは世界 "  # Japanese
        "안녕하세요 "  # Korean
        "Γειά σου κόσμε "  # Greek
        "🎉🚀💻🔥"  # Emojis
    )

    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    doc_id = response.json()["id"]
    get_response = client.get(f"/doc/{doc_id}")
    assert get_response.json()["content"] == sample_doc["content"]


def test_doc_add_with_nested_metadata(client, sample_doc):
    """Test that metadata can contain nested objects and arrays."""
    sample_doc["metadata"]["nested_object"] = {
        "location": {"country": "USA", "city": "San Francisco"},
        "dimensions": {"width": 100, "height": 200},
    }
    sample_doc["metadata"]["nested_array"] = ["tag1", "tag2", "tag3"]
    sample_doc["metadata"]["mixed_nested"] = [
        {"name": "item1", "value": 10},
        {"name": "item2", "value": 20},
    ]

    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    doc_id = response.json()["id"]
    get_response = client.get(f"/doc/{doc_id}")
    metadata = get_response.json()["metadata"]

    assert metadata["nested_object"]["location"]["city"] == "San Francisco"
    assert len(metadata["nested_array"]) == 3
    assert metadata["mixed_nested"][0]["value"] == 10


def test_doc_get_deleted_document(client, added_doc):
    """Test that getting a deleted document returns 404."""
    response = client.delete(f"/doc/{added_doc['id']}")
    assert response.status_code == 200

    response = client.get(f"/doc/{added_doc['id']}")
    assert response.status_code == 404


def test_doc_delete_same_document_twice(client, added_doc):
    """Test that deleting the same document twice returns 404 on second attempt."""
    response = client.delete(f"/doc/{added_doc['id']}")
    assert response.status_code == 200

    response = client.delete(f"/doc/{added_doc['id']}")
    assert response.status_code == 404


def test_doc_add_invalid_json_structure(client):
    """Test that POST /doc/add with invalid JSON structure (missing 'content') returns 422."""
    invalid_doc = {"metadata": {"source_file": "test.txt", "chunk_index": 0}}

    response = client.post("/doc/add", json=invalid_doc)
    assert response.status_code == 422


def test_doc_add_content_not_string(client, sample_doc):
    """Test that content field must be a string, not another type (e.g., number)."""
    sample_doc["content"] = 123
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 422


def test_doc_get_with_empty_string_id(client):
    """Test that GET /doc/ with empty string returns 404 or 405."""
    response = client.get("/doc/")
    assert response.status_code == 404


def test_doc_delete_with_empty_string_id(client):
    """Test that DELETE /doc/ with empty string returns 404 or 405."""
    response = client.delete("/doc/")
    assert response.status_code == 404


def test_doc_add_preserves_metadata_fields(client, sample_doc):
    """Test that all metadata fields are preserved after adding a document."""
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    doc_id = response.json()["id"]
    get_response = client.get(f"/doc/{doc_id}")
    assert get_response.status_code == 200

    doc = get_response.json()
    assert "content" in doc
    assert "metadata" in doc
    assert "source_file" in doc["metadata"]
    assert "chunk_index" in doc["metadata"]
    assert doc["content"] == sample_doc["content"]
    assert doc["metadata"]["source_file"] == sample_doc["metadata"]["source_file"]
    assert doc["metadata"]["chunk_index"] == sample_doc["metadata"]["chunk_index"]


def test_doc_deletion_does_not_affect_other_documents(client, sample_doc):
    """Test that deleting one document doesn't affect other documents."""
    doc1_data = {**sample_doc, "content": "Doc 1 content"}
    doc2_data = {**sample_doc, "content": "Doc 2 content"}
    doc3_data = {**sample_doc, "content": "Doc 3 content"}

    doc1_id = client.post("/doc/add", json=doc1_data).json()["id"]
    doc2_id = client.post("/doc/add", json=doc2_data).json()["id"]
    doc3_id = client.post("/doc/add", json=doc3_data).json()["id"]

    delete_response = client.delete(f"/doc/{doc1_id}")
    assert delete_response.status_code == 200

    doc2_response = client.get(f"/doc/{doc2_id}")
    assert doc2_response.status_code == 200
    assert doc2_response.json()["content"] == "Doc 2 content"

    doc3_response = client.get(f"/doc/{doc3_id}")
    assert doc3_response.status_code == 200
    assert doc3_response.json()["content"] == "Doc 3 content"

    doc1_response = client.get(f"/doc/{doc1_id}")
    assert doc1_response.status_code == 404


def test_doc_add_at_exact_length_limit(client, sample_doc):
    """Test that content at exactly MAX_CONTENT_LENGTH characters is accepted."""
    sample_doc["content"] = "a" * VFGConfig.MAX_CONTENT_LENGTH
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_one_char_over_length_limit(client, sample_doc):
    """Test that content at MAX_CONTENT_LENGTH + 1 characters is rejected."""
    sample_doc["content"] = "a" * (VFGConfig.MAX_CONTENT_LENGTH + 1)
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 422


def test_doc_add_with_length_9999(client, sample_doc):
    """Test that content at MAX_CONTENT_LENGTH - 1 characters is accepted."""
    sample_doc["content"] = "a" * (VFGConfig.MAX_CONTENT_LENGTH - 1)
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_full_lifecycle(client, sample_doc):
    """Test complete document lifecycle: add -> get -> delete -> verify deletion."""
    add_response = client.post("/doc/add", json=sample_doc)
    assert add_response.status_code == 201

    doc_id = add_response.json()["id"]
    get_response = client.get(f"/doc/{doc_id}")
    assert get_response.status_code == 200

    get_doc_id = get_response.json()["id"]
    del_response = client.delete(f"/doc/{get_doc_id}")
    assert del_response.status_code == 200

    del_response = client.delete(f"/doc/{get_doc_id}")
    assert del_response.status_code == 404


def test_doc_add_with_whitespace_only_content(client, sample_doc):
    """Test that content with only whitespace characters is handled appropriately."""
    sample_doc["content"] = ""
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 422


def test_doc_add_response_contains_all_required_fields(client, sample_doc):
    """Test that successful add response contains id and status fields."""
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "id" in response_data
    assert "status" in response_data


def test_doc_delete_response_contains_all_required_fields(client, added_doc):
    """Test that successful delete response contains id and status fields."""
    response = client.delete(f"/doc/{added_doc['id']}")
    assert response.status_code == 200

    response_data = response.json()
    assert "id" in response_data
    assert "status" in response_data


def test_doc_get_response_contains_all_required_fields(client, added_doc):
    """Test that successful get response contains id, content, and metadata fields."""
    response = client.get(f"/doc/{added_doc['id']}")
    assert response.status_code == 200

    response_data = response.json()
    assert "id" in response_data
    assert "content" in response_data
    assert "metadata" in response_data


def test_doc_add_with_valid_chunk_metadata(client, sample_doc):
    """Test that documents with both 'source_file' and 'chunk_index' are accepted."""
    assert "source_file" in sample_doc["metadata"]
    assert "chunk_index" in sample_doc["metadata"]

    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_get_with_invalid_uuid_format(client):
    """Test that GET /doc/{id} with malformed UUID returns 404."""
    response = client.get("/doc/malformed-uuid")
    assert response.status_code == 404


def test_doc_delete_with_invalid_uuid_format(client):
    """Test that DELETE /doc/{id} with malformed UUID returns 404."""
    response = client.delete("/doc/malformed-uuid")
    assert response.status_code == 404


def test_doc_add_returns_id_field(client, sample_doc):
    """Test that add response contains 'id' field."""
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "id" in response_data


def test_doc_add_returns_status_field(client, sample_doc):
    """Test that add response contains 'status' field."""
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "status" in response_data


def test_doc_delete_returns_id_field(client, added_doc):
    """Test that delete response contains 'id' field."""
    response = client.delete(f"/doc/{added_doc['id']}")
    assert response.status_code == 200

    response_data = response.json()
    assert "id" in response_data


def test_doc_get_after_multiple_adds(client, sample_doc):
    """Test retrieving specific document after adding multiple documents."""
    doc1_data = {**sample_doc, "content": "Doc 1 content"}
    doc2_data = {**sample_doc, "content": "Doc 2 content"}
    doc3_data = {**sample_doc, "content": "Doc 3 content"}

    doc1_id = client.post("/doc/add", json=doc1_data).json()["id"]
    client.post("/doc/add", json=doc2_data)
    client.post("/doc/add", json=doc3_data)

    response = client.get(f"/doc/{doc1_id}")
    assert response.status_code == 200

    response_data = response.json()
    assert response_data["content"] == doc1_data["content"]


def test_doc_add_increments_docs_added_metric(client, sample_doc):
    """Test that adding a document increments the docs_added metric."""
    initial_metrics = client.get("/index/stats").json()
    initial_count = initial_metrics["total_documents"]

    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    updated_metrics = client.get("/index/stats").json()
    updated_count = updated_metrics["total_documents"]

    assert updated_count == initial_count + 1


def test_doc_delete_increments_docs_deleted_metric(client, added_doc):
    """Test that deleting a document increments the docs_deleted metric."""
    initial_metrics = client.get("/index/stats").json()
    initial_count = initial_metrics["total_documents"]

    response = client.delete(f"/doc/{added_doc['id']}")
    assert response.status_code == 200

    updated_metrics = client.get("/index/stats").json()
    updated_count = updated_metrics["total_documents"]

    assert updated_count == initial_count - 1


def test_doc_add_metadata_with_null_values(client, sample_doc):
    """Test that metadata can contain null values."""
    sample_doc["metadata"]["optional_field"] = None
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_metadata_with_boolean_values(client, sample_doc):
    """Test that metadata can contain boolean values."""
    sample_doc["metadata"]["is_active"] = True
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_with_very_large_metadata(client, sample_doc):
    """Test document with extremely large metadata object."""
    sample_doc["metadata"]["large_field"] = "x" * 100_000
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_with_deeply_nested_metadata(client, sample_doc):
    """Test metadata with deep nesting (10+ levels)."""
    nested = {"level": {}}
    current = nested["level"]

    for i in range(10):
        current[f"level{i}"] = {}
        current = current[f"level{i}"]

    sample_doc["metadata"]["nested"] = nested
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_with_newlines_and_tabs(client, sample_doc):
    """Test content with newlines, tabs, and other whitespace."""
    sample_doc["content"] = "Line 1\nLine 2\tTabbed\r\nWindows newline"
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    doc_id = response.json()["id"]
    get_response = client.get(f"/doc/{doc_id}")
    assert get_response.json()["content"] == sample_doc["content"]


def test_doc_add_with_only_spaces(client, sample_doc):
    """Test content with only space characters (not empty string)."""
    sample_doc["content"] = "     "
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 400


def test_doc_add_with_control_characters(client, sample_doc):
    """Test content with control characters."""
    sample_doc["content"] = "Hello\x00World\x01Test"
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_get_with_special_characters_in_id(client):
    """Test GET with special characters that might break routing."""
    special_ids = [
        "../../../etc/passwd",
        "id%20with%20spaces",
        "id/with/slashes",
        "id?with=query",
    ]
    for special_id in special_ids:
        response = client.get(f"/doc/{special_id}")
        assert response.status_code == 404


def test_doc_get_with_very_long_id(client):
    """Test GET with extremely long ID string."""
    long_id = "a" * VFGConfig.MAX_CONTENT_LENGTH
    response = client.get(f"/doc/{long_id}")
    assert response.status_code == 404


def test_doc_add_preserves_order_of_metadata_keys(client, sample_doc):
    """Test that metadata key order is preserved (if using Python 3.7+)."""
    sample_doc["metadata"] = {
        "z_field": "last",
        "a_field": "first",
        "m_field": "middle",
    }
    response = client.post("/doc/add", json=sample_doc)
    doc_id = response.json()["id"]

    get_response = client.get(f"/doc/{doc_id}")
    metadata_keys = list(get_response.json()["metadata"].keys())
    expected_keys = list(sample_doc["metadata"].keys())
    assert metadata_keys == expected_keys


def test_doc_add_with_missing_metadata_field(client):
    """Test that request without 'metadata' field entirely is handled."""
    doc_without_metadata = {"content": "Test content"}
    response = client.post("/doc/add", json=doc_without_metadata)
    assert response.status_code == 201


def test_doc_add_with_extra_unknown_fields(client, sample_doc):
    """Test that extra fields in request are ignored or rejected."""
    sample_doc["unknown_field"] = "should be ignored"
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_get_after_index_compaction(client, sample_doc):
    """Test document retrieval after index compaction."""
    doc_ids = []
    for i in range(10):
        resp = client.post("/doc/add", json={**sample_doc, "content": f"Doc {i}"})
        doc_ids.append(resp.json()["id"])

    for doc_id in doc_ids[:3]:
        client.delete(f"/doc/{doc_id}")

    for doc_id in doc_ids[3:]:
        response = client.get(f"/doc/{doc_id}")
        assert response.status_code == 200


def test_doc_operations_update_all_relevant_metrics(client, sample_doc):
    """Test that doc operations update metrics comprehensively."""
    initial_metrics = client.get("/metrics").json()
    add_resp = client.post("/doc/add", json=sample_doc)
    doc_id = add_resp.json()["id"]

    after_add = client.get("/metrics").json()
    assert (
        after_add["usage"]["documents_added"]
        == initial_metrics["usage"]["documents_added"] + 1
    )
    assert (
        after_add["index"]["total_documents"]
        > initial_metrics["index"]["total_documents"]
    )

    client.delete(f"/doc/{doc_id}")
    after_delete = client.get("/metrics").json()
    assert (
        after_delete["usage"]["documents_deleted"]
        == initial_metrics["usage"]["documents_deleted"] + 1
    )


def test_doc_add_metadata_with_array_values(client, sample_doc):
    """Test that metadata can contain array values."""
    sample_doc["metadata"]["tags"] = ["python", "api", "testing"]
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_metadata_with_numeric_values(client, sample_doc):
    """Test that metadata can contain integers and floats."""
    sample_doc["metadata"]["count"] = 42
    sample_doc["metadata"]["score"] = 3.14
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201


def test_doc_delete_increments_deleted_count(client, added_doc):
    """Test that delete increments deleted_documents metric."""
    initial_metrics = client.get("/metrics").json()
    initial_deleted = initial_metrics["usage"]["documents_deleted"]

    client.delete(f"/doc/{added_doc['id']}")

    updated_metrics = client.get("/metrics").json()
    assert updated_metrics["usage"]["documents_deleted"] == initial_deleted + 1


def test_doc_get_with_numeric_id(client):
    """Test GET with pure numeric ID."""
    response = client.get("/doc/12345")
    assert response.status_code == 404
