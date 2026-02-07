"""Tests for search endpoints"""

from vectorforge.config import VFGConfig

# =============================================================================
# Search Endpoint Tests
# =============================================================================


def test_search_returns_200(client, added_doc):
    """Test that POST /search returns 200 status."""
    response = client.post("/search", json={"query": "test document"})
    assert response.status_code == 200


def test_search_returns_query_echo(client, added_doc):
    """Test that search response includes original query."""
    response = client.post("/search", json={"query": "test document"})
    data = response.json()
    assert data["query"] == "test document"


def test_search_returns_results_list(client, added_doc):
    """Test that search response contains results list."""
    response = client.post("/search", json={"query": "test document"})
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_returns_relevant_results(client, added_doc):
    """Test that search returns result count."""
    response = client.post("/search", json={"query": "test search"})
    data = response.json()
    assert "count" in data
    assert data["count"] >= 0


def test_search_with_empty_index_returns_empty_results(client):
    """Test that searching an empty index returns empty results."""
    response = client.post("/search", json={"query": "test search"})
    assert response.status_code == 200

    response_data = response.json()
    assert len(response_data["results"]) == 0


def test_search_respects_small_top_k(client, multiple_added_docs):
    """Test that search respects top_k parameter when less than default."""
    response = client.post("/search", json={"query": "test search", "top_k": 5})
    assert len(response.json()["results"]) == 5


def test_search_respects_large_top_k(client, multiple_added_docs):
    """Test that search respects top_k parameter when greater than default."""
    response = client.post("/search", json={"query": "test search", "top_k": 15})
    assert len(response.json()["results"]) == 15


def test_search_with_top_k_zero(client):
    """Test search with top_k set to 0."""
    response = client.post("/search", json={"query": "test search", "top_k": 0})
    assert response.status_code == 422


def test_search_with_negative_top_k(client):
    """Test that negative top_k values are rejected or handled gracefully."""
    response = client.post("/search", json={"query": "test search", "top_k": -1})
    assert response.status_code == 422


def test_search_with_empty_query(client):
    """Test search with an empty query string."""
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 422


def test_search_with_very_long_query(client):
    """Test search with a very long query string."""
    response = client.post(
        "/search", json={"query": "a" * (VFGConfig.MAX_QUERY_LENGTH + 1)}
    )
    assert response.status_code == 422


def test_search_with_reasonable_long_query(client):
    """Test that queries within limit are accepted."""
    response = client.post("/search", json={"query": "a" * VFGConfig.MAX_QUERY_LENGTH})
    assert response.status_code == 200


def test_search_returns_similarity_scores(client, added_doc):
    """Test that search results include similarity scores."""
    response = client.post("/search", json={"query": "a" * VFGConfig.MAX_QUERY_LENGTH})

    results = response.json()["results"]
    assert all("score" in result for result in results)


def test_search_results_sorted_by_score(client, multiple_added_docs):
    """Test that search results are sorted by similarity score in descending order."""
    response = client.post("/search", json={"query": "test search"})

    results = response.json()["results"]
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i + 1]["score"]


def test_search_result_contains_doc_id(client, multiple_added_docs):
    """Test that each search result contains a document ID."""
    response = client.post("/search", json={"query": "test search"})

    results = response.json()["results"]
    assert all("id" in result for result in results)


def test_search_result_contains_content(client):
    """Test that each search result contains document content."""
    response = client.post("/search", json={"query": "test search"})

    results = response.json()["results"]
    assert all(
        ("content" in result and len(result["content"]) > 0) for result in results
    )


def test_search_result_contains_metadata(client, multiple_added_docs):
    """Test that each search result contains document metadata."""
    response = client.post("/search", json={"query": "test search"})

    results = response.json()["results"]
    assert all("metadata" in result for result in results)


def test_search_with_special_characters_in_query(client):
    """Test search with special characters in the query string."""
    response = client.post("/search", json={"query": "!@#$%^&*()_+-=[]{}|;':\",./<>?"})
    assert response.status_code == 200


def test_search_with_edge_case_characters(client):
    """Test search with edge case special characters."""
    edge_cases = [
        "query\twith\ttabs",
        "query\nwith\nnewlines",
        "query\rwith\rcarriage",
        "query  with   spaces",
        "¿¡inverted punctuation!?",
        "emoji query 🔍 search",
        "null\x00byte query",
    ]

    for query in edge_cases:
        response = client.post("/search", json={"query": query})
        assert response.status_code == 200


def test_search_with_unicode_query(client):
    """Test search with unicode characters in the query."""
    response = client.post("/search", json={"query": "Hello 世界 🌍 Café Привет مرحبا"})

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "Hello 世界 🌍 Café Привет مرحبا"
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_finds_semantically_similar_content(client, multiple_added_docs):
    """Test that search finds semantically similar content even with different words."""
    response = client.post(
        "/search",
        json={
            "query": "Python is a type of snake in addition to a coding language",
            "top_k": 5,
        },
    )
    assert response.status_code == 200

    results = response.json()["results"]
    result_contents = [r["content"] for r in results]

    expected_content = (
        "Python is a high-level programming language used for web development"
    )
    assert (
        expected_content in result_contents
    ), f"Expected content not found in top 5 results: {result_contents}"


def test_search_returns_results_regardless_of_relevance(client):
    """Test that search returns top_k results even when poorly matched."""
    client.post(
        "/doc/add",
        json={"content": "Python programming language development", "metadata": {}},
    )
    response = client.post(
        "/search", json={"query": "ancient Egyptian pyramids and pharaohs", "top_k": 1}
    )
    assert response.status_code == 200

    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["score"] < 0.5


def test_search_response_format(client):
    """Test that search response contains all required fields."""
    response = client.post(
        "/search",
        json={
            "query": "test search",
        },
    )

    search_response = response.json()
    assert "query" in search_response
    assert "results" in search_response
    assert "count" in search_response


def test_search_excludes_deleted_documents(client, added_doc):
    """Test that search results don't include deleted documents."""
    response = client.delete(f"/doc/{added_doc['id']}")
    assert response.status_code == 200

    response = client.post("/search", json={"query": "test document"})
    assert response.status_code == 200

    results = response.json()["results"]
    assert len(results) == 0


def test_search_default_top_k_value(client, multiple_added_docs):
    """Test that search uses default top_k when not specified."""
    response = client.post(
        "/search",
        json={
            "query": "test search",
        },
    )
    assert response.status_code == 200

    results = response.json()["results"]
    assert len(results) == 10


def test_search_increments_total_queries_metric(client, added_doc):
    """Test that search increments total_queries metric."""
    initial_metrics = client.get("/metrics").json()
    initial_queries = initial_metrics["performance"]["total_queries"]

    response = client.post(
        "/search",
        json={
            "query": "test search",
        },
    )
    assert response.status_code == 200

    updated_metrics = client.get("/metrics").json()
    updated_queries = updated_metrics["performance"]["total_queries"]
    assert updated_queries == initial_queries + 1


def test_search_updates_total_query_time_metric(client):
    """Test that search updates total_query_time_ms metric."""
    initial_metrics = client.get("/metrics").json()
    initial_total_query_time = initial_metrics["performance"]["total_query_time_ms"]

    response = client.post(
        "/search",
        json={
            "query": "test search",
        },
    )
    assert response.status_code == 200

    updated_metrics = client.get("/metrics").json()
    updated_total_query_time = updated_metrics["performance"]["total_query_time_ms"]
    assert updated_total_query_time >= initial_total_query_time


def test_search_updates_last_query_timestamp(client, added_doc):
    """Test that search updates last_query_at timestamp."""
    response = client.post(
        "/search",
        json={
            "query": "test search",
        },
    )
    assert response.status_code == 200

    initial_metrics = client.get("/metrics").json()
    initial_query_at = initial_metrics["timestamps"]["last_query_at"]

    response = client.post(
        "/search",
        json={
            "query": "test search",
        },
    )
    assert response.status_code == 200

    updated_metrics = client.get("/metrics").json()
    updated_query_at = updated_metrics["timestamps"]["last_query_at"]
    assert updated_query_at > initial_query_at


def test_search_result_score_format(client, added_doc):
    """Test that search result scores are floats."""
    response = client.post(
        "/search",
        json={
            "query": "test search",
        },
    )
    assert response.status_code == 200
    assert isinstance(response.json()["results"][0]["score"], float)


def test_search_query_with_only_whitespace(client):
    """Test search with query containing only whitespace."""
    response = client.post(
        "/search",
        json={
            "query": "          ",
        },
    )
    assert response.status_code == 400


def test_search_with_missing_query_field(client):
    """Test that search request without 'query' field returns 422."""
    response = client.post("/search", json={})
    assert response.status_code == 422


def test_search_with_top_k_exceeding_index_size(client, added_doc):
    """Test search with top_k larger than number of documents in index."""
    response = client.post("/search", json={"query": "test search", "top_k": 100})
    assert response.status_code == 200

    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == added_doc["id"]


def test_search_query_with_mixed_case(client, sample_doc):
    """Test that search is case-insensitive (if applicable)."""
    sample_doc["content"] = "python programming"
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    response = client.post("/search", json={"query": "Python Programming"})
    assert response.status_code == 200

    results = response.json()["results"]
    assert results[0]["content"] == sample_doc["content"]


def test_search_query_strips_leading_trailing_whitespace(client, added_doc):
    """Test that queries with leading/trailing whitespace are handled."""
    response1 = client.post("/search", json={"query": "test"})
    assert response1.status_code == 200

    response2 = client.post("/search", json={"query": "  test  "})
    assert response2.status_code == 200

    response1_data = response1.json()
    response2_data = response2.json()
    assert response1_data["query"] == response2_data["query"]
    assert response1_data["results"] == response2_data["results"]


def test_search_scores_are_between_zero_and_one(client, multiple_added_docs):
    """Test that all similarity scores are in valid range [0, 1]."""
    response = client.post("/search", json={"query": "test"})
    results = response.json()["results"]

    for result in results:
        assert 0.0 <= result["score"] <= 1.0


def test_search_returns_consistent_results_on_repeat(client, added_doc):
    """Test that repeated searches with same query return consistent results."""
    response1 = client.post("/search", json={"query": "test", "top_k": 5})
    response2 = client.post("/search", json={"query": "test", "top_k": 5})
    assert response1.json()["results"] == response2.json()["results"]


def test_search_with_top_k_one(client, multiple_added_docs):
    """Test search with minimum valid top_k value."""
    response = client.post("/search", json={"query": "test", "top_k": 1})
    assert len(response.json()["results"]) == 1


def test_search_with_too_large_top_k(client, added_doc):
    """Test search with extremely large top_k value."""
    response = client.post("/search", json={"query": "test", "top_k": 101})
    assert response.status_code == 422


def test_search_result_content_is_complete(client):
    """Test that returned content matches original document content exactly."""
    content = "This is a specific test document with unique content."
    add_response = client.post("/doc/add", json={"content": content, "metadata": {}})
    doc_id = add_response.json()["id"]

    search_response = client.post("/search", json={"query": content})
    results = search_response.json()["results"]

    our_result = next(r for r in results if r["id"] == doc_id)
    assert our_result["content"] == content


def test_search_preserves_all_metadata_fields(client):
    """Test that search results include all original metadata fields."""
    metadata = {
        "source_file": "test.txt",
        "chunk_index": 0,
        "custom_field": "custom_value",
        "nested": {"key": "value"},
    }
    add_response = client.post(
        "/doc/add", json={"content": "test content", "metadata": metadata}
    )
    assert add_response.status_code == 201

    search_response = client.post("/search", json={"query": "test content"})
    assert search_response.status_code == 200

    result_metadata = search_response.json()["results"][0]["metadata"]
    assert result_metadata == metadata
