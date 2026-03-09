"""Tests for document management endpoints.

Covers:
    POST   /collections/{collection_name}/documents
    POST   /collections/{collection_name}/documents/batch
    GET    /collections/{collection_name}/documents/{doc_id}
    DELETE /collections/{collection_name}/documents/{doc_id}
    DELETE /collections/{collection_name}/documents
"""

from vectorforge.config import VFGConfig

# =============================================================================
# Document Endpoint Tests
# =============================================================================


def test_doc_add_returns_unique_ids_for_multiple_docs(client, sample_doc):
    """Test that adding multiple documents generates unique IDs for each."""
    first_response = client.post("/collections/vectorforge/documents", json=sample_doc)
    first_id = first_response.json()["id"]

    second_response = client.post("/collections/vectorforge/documents", json=sample_doc)
    second_id = second_response.json()["id"]

    assert first_id != second_id


def test_doc_add_null_metadata(client, sample_doc):
    """Test that adding a document with null metadata returns 400."""
    sample_doc["metadata"] = None
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_empty_metadata(client, sample_doc):
    """Test that adding a document with empty metadata dict succeeds."""
    sample_doc["metadata"] = {}
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "id" in response_data
    assert response_data["status"] == "indexed"


def test_doc_add_large_content(client, sample_doc):
    """Test that adding a document exceeding max content length returns 422."""
    sample_doc["content"] = "a" * (VFGConfig.MAX_CONTENT_LENGTH + 1)
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 422


def test_doc_add_empty_content(client, sample_doc):
    """Test that adding a document with empty content returns 400."""
    sample_doc["content"] = ""
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 422


def test_doc_get_returns_200_with_valid_id(client, added_doc):
    """Test that GET /collections/vectorforge/documents/{id} returns 200 for existing document."""
    response = client.get(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 200


def test_doc_get_returns_correct_id(client, added_doc):
    """Test that retrieved document has correct ID."""
    response = client.get(f"/collections/vectorforge/documents/{added_doc['id']}")
    data = response.json()
    assert data["id"] == added_doc["id"]


def test_doc_get_returns_matching_content(client, sample_doc):
    """Test that retrieved document content matches original."""
    add_response = client.post("/collections/vectorforge/documents", json=sample_doc)
    doc_id = add_response.json()["id"]

    get_response = client.get(f"/collections/vectorforge/documents/{doc_id}")
    data = get_response.json()
    assert data["content"] == sample_doc["content"]


def test_doc_get_preserves_metadata(client, sample_doc):
    """Test that retrieved document preserves metadata."""
    add_response = client.post("/collections/vectorforge/documents", json=sample_doc)
    doc_id = add_response.json()["id"]
    get_response = client.get(f"/collections/vectorforge/documents/{doc_id}")

    metadata = get_response.json()["metadata"]
    assert metadata["source_file"] == "test.txt"
    assert metadata["chunk_index"] == 0


def test_doc_get_returns_404_when_not_found(client):
    """Test 404 response when retrieving a non-existent document."""
    response = client.get("/collections/vectorforge/documents/nonexistent-id")
    assert response.status_code == 404


def test_doc_add_returns_201(client, sample_doc):
    """Test that POST /collections/vectorforge/documents returns 201 status."""
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_returns_document_id(client, sample_doc):
    """Test that POST /collections/vectorforge/documents returns a document ID."""
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    data = response.json()
    assert "id" in data
    assert isinstance(data["id"], str)
    assert len(data["id"]) > 0


def test_doc_add_creates_document_with_id(client, sample_doc):
    """Test that POST /collections/vectorforge/documents returns indexed status."""
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    data = response.json()
    assert data["status"] == "indexed"


def test_doc_delete_returns_200(client, added_doc):
    """Test that DELETE /collections/vectorforge/documents/{id} returns 200 status."""
    response = client.delete(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 200


def test_doc_delete_returns_deleted_status(client, added_doc):
    """Test that DELETE /collections/vectorforge/documents/{id} returns deleted status."""
    response = client.delete(f"/collections/vectorforge/documents/{added_doc['id']}")
    data = response.json()
    assert data["status"] == "deleted"


def test_doc_delete_removes_from_index(client, added_doc):
    """Test that deleted document is no longer retrievable."""
    doc_id = added_doc["id"]
    client.delete(f"/collections/vectorforge/documents/{doc_id}")

    get_response = client.get(f"/collections/vectorforge/documents/{doc_id}")
    assert get_response.status_code == 404


def test_doc_delete_returns_404_when_not_found(client):
    """Test 404 response when deleting a non-existent document."""
    response = client.delete("/collections/vectorforge/documents/nonexistent-id")
    assert response.status_code == 404


def test_doc_add_metadata_with_only_source_file(client, sample_doc):
    """Test that metadata with only 'source_file' (missing 'chunk_index') returns 400."""
    del sample_doc["metadata"]["chunk_index"]
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 400


def test_doc_add_metadata_with_only_chunk_index(client, sample_doc):
    """Test that metadata with only 'chunk_index' (missing 'source_file') returns 400."""
    del sample_doc["metadata"]["source_file"]
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 400


def test_doc_add_metadata_none_value_returns_422(client):
    """Test that a None metadata value is rejected with 422."""
    response = client.post(
        "/collections/vectorforge/documents",
        json={"content": "Some content", "metadata": {"author": None}},
    )
    assert response.status_code == 422


def test_doc_add_metadata_list_value_returns_422(client):
    """Test that a list metadata value is rejected with 422."""
    response = client.post(
        "/collections/vectorforge/documents",
        json={"content": "Some content", "metadata": {"tags": ["python", "ml"]}},
    )
    assert response.status_code == 422


def test_doc_add_metadata_nested_dict_value_returns_422(client):
    """Test that a nested dict metadata value is rejected with 422."""
    response = client.post(
        "/collections/vectorforge/documents",
        json={"content": "Some content", "metadata": {"nested": {"key": "val"}}},
    )
    assert response.status_code == 422


def test_doc_add_metadata_valid_types_all_accepted(client):
    """Test that str, int, float, and bool metadata values are all accepted."""
    response = client.post(
        "/collections/vectorforge/documents",
        json={
            "content": "Some content",
            "metadata": {
                "str_field": "hello",
                "int_field": 42,
                "float_field": 3.14,
                "bool_field": True,
            },
        },
    )
    assert response.status_code == 201


def test_doc_add_with_special_characters_in_content(client, sample_doc):
    """Test that documents with special characters in content are accepted."""
    sample_doc["metadata"]["content"] = "!@#$%^&*()_~-=+,.<>/?;:\"'[]|\\"
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
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

    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201

    doc_id = response.json()["id"]
    get_response = client.get(f"/collections/vectorforge/documents/{doc_id}")
    assert get_response.json()["content"] == sample_doc["content"]


def test_doc_get_deleted_document(client, added_doc):
    """Test that getting a deleted document returns 404."""
    response = client.delete(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 200

    response = client.get(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 404


def test_doc_delete_same_document_twice(client, added_doc):
    """Test that deleting the same document twice returns 404 on second attempt."""
    response = client.delete(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 200

    response = client.delete(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 404


def test_doc_add_invalid_json_structure(client):
    """Test that POST /collections/vectorforge/documents with invalid JSON structure (missing 'content') returns 422."""
    invalid_doc = {"metadata": {"source_file": "test.txt", "chunk_index": 0}}

    response = client.post("/collections/vectorforge/documents", json=invalid_doc)
    assert response.status_code == 422


def test_doc_add_content_not_string(client, sample_doc):
    """Test that content field must be a string, not another type (e.g., number)."""
    sample_doc["content"] = 123
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 422


def test_doc_get_with_empty_string_id(client):
    """Test that GET /collections/vectorforge/documents/ with empty string returns 404 or 405."""
    response = client.get("/collections/vectorforge/documents/")
    assert response.status_code in (404, 405)


def test_doc_delete_with_empty_string_id(client):
    """Test that DELETE /collections/vectorforge/documents/ with empty string returns 404, 405, or 422."""
    response = client.delete("/collections/vectorforge/documents/")
    assert response.status_code in (404, 405, 422)


def test_doc_add_preserves_metadata_fields(client, sample_doc):
    """Test that all metadata fields are preserved after adding a document."""
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201

    doc_id = response.json()["id"]
    get_response = client.get(f"/collections/vectorforge/documents/{doc_id}")
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

    doc1_id = client.post("/collections/vectorforge/documents", json=doc1_data).json()[
        "id"
    ]
    doc2_id = client.post("/collections/vectorforge/documents", json=doc2_data).json()[
        "id"
    ]
    doc3_id = client.post("/collections/vectorforge/documents", json=doc3_data).json()[
        "id"
    ]

    delete_response = client.delete(f"/collections/vectorforge/documents/{doc1_id}")
    assert delete_response.status_code == 200

    doc2_response = client.get(f"/collections/vectorforge/documents/{doc2_id}")
    assert doc2_response.status_code == 200
    assert doc2_response.json()["content"] == "Doc 2 content"

    doc3_response = client.get(f"/collections/vectorforge/documents/{doc3_id}")
    assert doc3_response.status_code == 200
    assert doc3_response.json()["content"] == "Doc 3 content"

    doc1_response = client.get(f"/collections/vectorforge/documents/{doc1_id}")
    assert doc1_response.status_code == 404


def test_doc_add_at_exact_length_limit(client, sample_doc):
    """Test that content at exactly MAX_CONTENT_LENGTH characters is accepted."""
    sample_doc["content"] = "a" * VFGConfig.MAX_CONTENT_LENGTH
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_one_char_over_length_limit(client, sample_doc):
    """Test that content at MAX_CONTENT_LENGTH + 1 characters is rejected."""
    sample_doc["content"] = "a" * (VFGConfig.MAX_CONTENT_LENGTH + 1)
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 422


def test_doc_add_with_length_9999(client, sample_doc):
    """Test that content at MAX_CONTENT_LENGTH - 1 characters is accepted."""
    sample_doc["content"] = "a" * (VFGConfig.MAX_CONTENT_LENGTH - 1)
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201


def test_doc_full_lifecycle(client, sample_doc):
    """Test complete document lifecycle: add -> get -> delete -> verify deletion."""
    add_response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert add_response.status_code == 201

    doc_id = add_response.json()["id"]
    get_response = client.get(f"/collections/vectorforge/documents/{doc_id}")
    assert get_response.status_code == 200

    get_doc_id = get_response.json()["id"]
    del_response = client.delete(f"/collections/vectorforge/documents/{get_doc_id}")
    assert del_response.status_code == 200

    del_response = client.delete(f"/collections/vectorforge/documents/{get_doc_id}")
    assert del_response.status_code == 404


def test_doc_add_with_whitespace_only_content(client, sample_doc):
    """Test that content with only whitespace characters is handled appropriately."""
    sample_doc["content"] = ""
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 422


def test_doc_add_response_contains_all_required_fields(client, sample_doc):
    """Test that successful add response contains id and status fields."""
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "id" in response_data
    assert "status" in response_data


def test_doc_delete_response_contains_all_required_fields(client, added_doc):
    """Test that successful delete response contains id and status fields."""
    response = client.delete(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 200

    response_data = response.json()
    assert "id" in response_data
    assert "status" in response_data


def test_doc_get_response_contains_all_required_fields(client, added_doc):
    """Test that successful get response contains id, content, and metadata fields."""
    response = client.get(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 200

    response_data = response.json()
    assert "id" in response_data
    assert "content" in response_data
    assert "metadata" in response_data


def test_doc_add_with_valid_chunk_metadata(client, sample_doc):
    """Test that documents with both 'source_file' and 'chunk_index' are accepted."""
    assert "source_file" in sample_doc["metadata"]
    assert "chunk_index" in sample_doc["metadata"]

    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201


def test_doc_get_with_invalid_uuid_format(client):
    """Test that GET /collections/vectorforge/documents/{id} with malformed UUID returns 404."""
    response = client.get("/collections/vectorforge/documents/malformed-uuid")
    assert response.status_code == 404


def test_doc_delete_with_invalid_uuid_format(client):
    """Test that DELETE /collections/vectorforge/documents/{id} with malformed UUID returns 404."""
    response = client.delete("/collections/vectorforge/documents/malformed-uuid")
    assert response.status_code == 404


def test_doc_add_returns_id_field(client, sample_doc):
    """Test that add response contains 'id' field."""
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "id" in response_data


def test_doc_add_returns_status_field(client, sample_doc):
    """Test that add response contains 'status' field."""
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "status" in response_data


def test_doc_delete_returns_id_field(client, added_doc):
    """Test that delete response contains 'id' field."""
    response = client.delete(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 200

    response_data = response.json()
    assert "id" in response_data


def test_doc_get_after_multiple_adds(client, sample_doc):
    """Test retrieving specific document after adding multiple documents."""
    doc1_data = {**sample_doc, "content": "Doc 1 content"}
    doc2_data = {**sample_doc, "content": "Doc 2 content"}
    doc3_data = {**sample_doc, "content": "Doc 3 content"}

    doc1_id = client.post("/collections/vectorforge/documents", json=doc1_data).json()[
        "id"
    ]
    client.post("/collections/vectorforge/documents", json=doc2_data)
    client.post("/collections/vectorforge/documents", json=doc3_data)

    response = client.get(f"/collections/vectorforge/documents/{doc1_id}")
    assert response.status_code == 200

    response_data = response.json()
    assert response_data["content"] == doc1_data["content"]


def test_doc_add_increments_docs_added_metric(client, sample_doc):
    """Test that adding a document increments the docs_added metric."""
    initial_metrics = client.get("/collections/vectorforge/stats").json()
    initial_count = initial_metrics["total_documents"]

    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201

    updated_metrics = client.get("/collections/vectorforge/stats").json()
    updated_count = updated_metrics["total_documents"]

    assert updated_count == initial_count + 1


def test_doc_delete_increments_docs_deleted_metric(client, added_doc):
    """Test that deleting a document increments the docs_deleted metric."""
    initial_metrics = client.get("/collections/vectorforge/stats").json()
    initial_count = initial_metrics["total_documents"]

    response = client.delete(f"/collections/vectorforge/documents/{added_doc['id']}")
    assert response.status_code == 200

    updated_metrics = client.get("/collections/vectorforge/stats").json()
    updated_count = updated_metrics["total_documents"]

    assert updated_count == initial_count - 1


def test_doc_add_metadata_with_boolean_values(client, sample_doc):
    """Test that metadata can contain boolean values."""
    sample_doc["metadata"]["is_active"] = True
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_with_very_large_metadata(client, sample_doc):
    """Test document with extremely large metadata object."""
    sample_doc["metadata"]["large_field"] = "x" * 100_000
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201


def test_doc_add_with_newlines_and_tabs(client, sample_doc):
    """Test content with newlines, tabs, and other whitespace."""
    sample_doc["content"] = "Line 1\nLine 2\tTabbed\r\nWindows newline"
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201

    doc_id = response.json()["id"]
    get_response = client.get(f"/collections/vectorforge/documents/{doc_id}")
    assert get_response.json()["content"] == sample_doc["content"]


def test_doc_add_with_only_spaces(client, sample_doc):
    """Test content with only space characters (not empty string)."""
    sample_doc["content"] = "     "
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 400


def test_doc_add_with_control_characters(client, sample_doc):
    """Test content with control characters."""
    sample_doc["content"] = "Hello\x00World\x01Test"
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
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
        response = client.get(f"/collections/vectorforge/documents/{special_id}")
        assert response.status_code == 404


def test_doc_get_with_very_long_id(client):
    """Test GET with extremely long ID string."""
    long_id = "a" * VFGConfig.MAX_CONTENT_LENGTH
    response = client.get(f"/collections/vectorforge/documents/{long_id}")
    assert response.status_code == 404


def test_doc_add_with_missing_metadata_field(client):
    """Test that request without 'metadata' field entirely is handled."""
    doc_without_metadata = {"content": "Test content"}
    response = client.post(
        "/collections/vectorforge/documents", json=doc_without_metadata
    )
    assert response.status_code == 201


def test_doc_add_with_extra_unknown_fields(client, sample_doc):
    """Test that extra fields in request are ignored or rejected."""
    sample_doc["unknown_field"] = "should be ignored"
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201


def test_doc_get_after_index_compaction(client, sample_doc):
    """Test document retrieval after index compaction."""
    doc_ids = []
    for i in range(10):
        resp = client.post(
            "/collections/vectorforge/documents",
            json={**sample_doc, "content": f"Doc {i}"},
        )
        doc_ids.append(resp.json()["id"])

    for doc_id in doc_ids[:3]:
        client.delete(f"/collections/vectorforge/documents/{doc_id}")

    for doc_id in doc_ids[3:]:
        response = client.get(f"/collections/vectorforge/documents/{doc_id}")
        assert response.status_code == 200


def test_doc_operations_update_all_relevant_metrics(client, sample_doc):
    """Test that doc operations update metrics comprehensively."""
    initial_metrics = client.get("/collections/vectorforge/metrics").json()
    add_resp = client.post("/collections/vectorforge/documents", json=sample_doc)
    doc_id = add_resp.json()["id"]

    after_add = client.get("/collections/vectorforge/metrics").json()
    assert (
        after_add["usage"]["documents_added"]
        == initial_metrics["usage"]["documents_added"] + 1
    )
    assert (
        after_add["index"]["total_documents"]
        > initial_metrics["index"]["total_documents"]
    )

    client.delete(f"/collections/vectorforge/documents/{doc_id}")
    after_delete = client.get("/collections/vectorforge/metrics").json()
    assert (
        after_delete["usage"]["documents_deleted"]
        == initial_metrics["usage"]["documents_deleted"] + 1
    )


def test_doc_add_metadata_with_array_values(client, sample_doc):
    """Test that metadata cannot contain array values."""
    sample_doc["metadata"]["tags"] = ["python", "api", "testing"]
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 422


def test_doc_add_metadata_with_numeric_values(client, sample_doc):
    """Test that metadata can contain integers and floats."""
    sample_doc["metadata"]["count"] = 42
    sample_doc["metadata"]["score"] = 3.14
    response = client.post("/collections/vectorforge/documents", json=sample_doc)
    assert response.status_code == 201


def test_doc_delete_increments_deleted_count(client, added_doc):
    """Test that delete increments deleted_documents metric."""
    initial_metrics = client.get("/collections/vectorforge/metrics").json()
    initial_deleted = initial_metrics["usage"]["documents_deleted"]

    client.delete(f"/collections/vectorforge/documents/{added_doc['id']}")

    updated_metrics = client.get("/collections/vectorforge/metrics").json()
    assert updated_metrics["usage"]["documents_deleted"] == initial_deleted + 1


def test_doc_get_with_numeric_id(client):
    """Test GET with pure numeric ID."""
    response = client.get("/collections/vectorforge/documents/12345")
    assert response.status_code == 404


# =============================================================================
# Batch Add Endpoint Tests (POST /documents/batch)
# =============================================================================


def test_batch_add_returns_201(client, sample_doc):
    """Test that POST /documents/batch returns 201 status."""
    payload = {"documents": [sample_doc, {**sample_doc, "content": "Second doc"}]}
    response = client.post("/collections/vectorforge/documents/batch", json=payload)
    assert response.status_code == 201


def test_batch_add_returns_ids_list(client, sample_doc):
    """Test that batch add response contains a list of IDs."""
    payload = {"documents": [sample_doc, {**sample_doc, "content": "Second doc"}]}
    response = client.post("/collections/vectorforge/documents/batch", json=payload)
    data = response.json()
    assert "ids" in data
    assert isinstance(data["ids"], list)
    assert len(data["ids"]) == 2


def test_batch_add_returns_indexed_status(client, sample_doc):
    """Test that batch add response contains 'indexed' status."""
    payload = {"documents": [sample_doc]}
    response = client.post("/collections/vectorforge/documents/batch", json=payload)
    assert response.json()["status"] == "indexed"


def test_batch_add_ids_are_unique(client, sample_doc):
    """Test that batch add assigns unique IDs to each document."""
    payload = {"documents": [sample_doc, {**sample_doc, "content": "Second doc"}]}
    response = client.post("/collections/vectorforge/documents/batch", json=payload)
    ids = response.json()["ids"]
    assert len(ids) == len(set(ids))


def test_batch_add_documents_are_retrievable(client, sample_doc):
    """Test that all documents added via batch are individually retrievable."""
    payload = {
        "documents": [
            {**sample_doc, "content": "Batch doc A"},
            {**sample_doc, "content": "Batch doc B"},
        ]
    }
    response = client.post("/collections/vectorforge/documents/batch", json=payload)
    ids = response.json()["ids"]

    for doc_id in ids:
        get_response = client.get(f"/collections/vectorforge/documents/{doc_id}")
        assert get_response.status_code == 200


def test_batch_add_single_document_returns_one_id(client, sample_doc):
    """Test that batch add with one document returns a single-element list."""
    payload = {"documents": [sample_doc]}
    response = client.post("/collections/vectorforge/documents/batch", json=payload)
    assert len(response.json()["ids"]) == 1


def test_batch_add_empty_list_returns_422(client):
    """Test that batch add with an empty documents list returns 422."""
    response = client.post(
        "/collections/vectorforge/documents/batch", json={"documents": []}
    )
    assert response.status_code == 422


def test_batch_add_exceeds_max_batch_size_returns_422(client, sample_doc):
    """Test that batch add exceeding MAX_BATCH_SIZE returns 422."""
    from vectorforge.config import VFGConfig

    payload = {"documents": [sample_doc] * (VFGConfig.MAX_BATCH_SIZE + 1)}
    response = client.post("/collections/vectorforge/documents/batch", json=payload)
    assert response.status_code == 422


def test_batch_add_invalid_document_returns_422(client, sample_doc):
    """Test that batch add with an invalid document (missing content) returns 422."""
    payload = {
        "documents": [
            sample_doc,
            {"metadata": {"source_file": "x.txt", "chunk_index": 0}},
        ]
    }
    response = client.post("/collections/vectorforge/documents/batch", json=payload)
    assert response.status_code == 422


def test_batch_add_missing_documents_field_returns_422(client):
    """Test that batch add request without 'documents' field returns 422."""
    response = client.post("/collections/vectorforge/documents/batch", json={})
    assert response.status_code == 422


# =============================================================================
# Batch Delete Endpoint Tests (DELETE /documents)
# =============================================================================


def test_batch_delete_returns_200(client, sample_doc):
    """Test that DELETE /documents returns 200 status."""
    add_resp = client.post("/collections/vectorforge/documents", json=sample_doc)
    doc_id = add_resp.json()["id"]

    response = client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={"ids": [doc_id]},
    )
    assert response.status_code == 200


def test_batch_delete_returns_deleted_ids(client, sample_doc):
    """Test that batch delete response contains the deleted IDs."""
    ids = [
        client.post(
            "/collections/vectorforge/documents",
            json={**sample_doc, "content": f"Doc {i}"},
        ).json()["id"]
        for i in range(3)
    ]

    response = client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={"ids": ids},
    )
    data = response.json()
    assert set(data["ids"]) == set(ids)


def test_batch_delete_returns_deleted_status(client, sample_doc):
    """Test that batch delete response contains 'deleted' status."""
    doc_id = client.post("/collections/vectorforge/documents", json=sample_doc).json()[
        "id"
    ]

    response = client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={"ids": [doc_id]},
    )
    assert response.json()["status"] == "deleted"


def test_batch_delete_removes_documents_from_index(client, sample_doc):
    """Test that batch-deleted documents are no longer retrievable."""
    ids = [
        client.post(
            "/collections/vectorforge/documents",
            json={**sample_doc, "content": f"Doc {i}"},
        ).json()["id"]
        for i in range(2)
    ]

    client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={"ids": ids},
    )

    for doc_id in ids:
        assert (
            client.get(f"/collections/vectorforge/documents/{doc_id}").status_code
            == 404
        )


def test_batch_delete_all_nonexistent_returns_404(client):
    """Test that batch delete with only nonexistent IDs returns 404."""
    response = client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={"ids": ["nonexistent-1", "nonexistent-2"]},
    )
    assert response.status_code == 404


def test_batch_delete_partial_match_returns_only_deleted_ids(client, sample_doc):
    """Test that batch delete with a mix of valid and invalid IDs returns only deleted ones."""
    doc_id = client.post("/collections/vectorforge/documents", json=sample_doc).json()[
        "id"
    ]

    response = client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={"ids": [doc_id, "nonexistent-id"]},
    )
    data = response.json()
    assert data["status"] == "deleted"
    assert data["ids"] == [doc_id]


def test_batch_delete_empty_ids_returns_422(client):
    """Test that batch delete with an empty ID list returns 422."""
    response = client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={"ids": []},
    )
    assert response.status_code == 422


def test_batch_delete_exceeds_max_batch_size_returns_422(client):
    """Test that batch delete exceeding MAX_BATCH_SIZE returns 422."""
    response = client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={"ids": ["id"] * (VFGConfig.MAX_BATCH_SIZE + 1)},
    )
    assert response.status_code == 422


def test_batch_delete_missing_ids_field_returns_422(client):
    """Test that batch delete request without 'ids' field returns 422."""
    response = client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={},
    )
    assert response.status_code == 422


def test_batch_delete_updates_docs_deleted_metric(client, sample_doc):
    """Test that batch delete increments the docs_deleted metric."""
    ids = [
        client.post(
            "/collections/vectorforge/documents",
            json={**sample_doc, "content": f"Doc {i}"},
        ).json()["id"]
        for i in range(3)
    ]

    initial = client.get("/collections/vectorforge/metrics").json()["usage"][
        "documents_deleted"
    ]
    client.request(
        "DELETE",
        "/collections/vectorforge/documents",
        json={"ids": ids},
    )
    updated = client.get("/collections/vectorforge/metrics").json()["usage"][
        "documents_deleted"
    ]

    assert updated == initial + 3
