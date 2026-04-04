"""Tests for the re-ranking feature.

Covers:
    SearchQuery.resolve_top_n validator (top_n/top_k interaction)
    SearchQuery top_n field-level validation (MIN_TOP_N, MAX_TOP_N)
    VectorEngine._sigmoid (logit-to-score normalisation)
    VectorEngine._rerank (cross-encoder re-scoring and ordering)
    VectorEngine.search with rerank=True (end-to-end engine behaviour)
"""

import pytest
from pydantic import ValidationError

from vectorforge.config import VFGConfig
from vectorforge.models.search import SearchQuery, SearchResult

# =============================================================================
# SearchQuery Model: resolve_top_n Validator
# =============================================================================


def test_search_query_rerank_disabled_top_n_resolves_to_default() -> None:
    """When rerank=False and top_n not given, top_n resolves to DEFAULT_TOP_N."""
    q = SearchQuery(query="test", rerank=False)
    assert q.top_n == VFGConfig.DEFAULT_TOP_N


def test_search_query_rerank_disabled_preserves_explicit_top_n() -> None:
    """When rerank=False and top_n is explicit, the given value is kept."""
    q = SearchQuery(query="test", rerank=False, top_n=3)
    assert q.top_n == 3


def test_search_query_rerank_disabled_stays_false() -> None:
    """rerank flag remains False when explicitly set to False."""
    q = SearchQuery(query="test", rerank=False)
    assert q.rerank is False


def test_search_query_top_k_one_auto_disables_reranking() -> None:
    """When top_k=1, reranking is auto-disabled regardless of rerank flag."""
    q = SearchQuery(query="test", top_k=1, rerank=True)
    assert q.rerank is False


def test_search_query_top_k_one_resolves_top_n_to_default() -> None:
    """When top_k=1 disables reranking, top_n falls back to DEFAULT_TOP_N."""
    q = SearchQuery(query="test", top_k=1, rerank=True)
    assert q.top_n == VFGConfig.DEFAULT_TOP_N


def test_search_query_rerank_true_top_n_auto_clamps_when_top_k_less_than_default() -> (
    None
):
    """When top_k < DEFAULT_TOP_N, auto top_n is clamped to top_k - 1."""
    q = SearchQuery(query="test", top_k=3, rerank=True)
    assert q.top_n == 2  # min(DEFAULT_TOP_N=5, top_k-1=2) = 2


def test_search_query_rerank_true_top_n_uses_default_when_top_k_exceeds_it() -> None:
    """When top_k > DEFAULT_TOP_N, auto top_n resolves to DEFAULT_TOP_N."""
    q = SearchQuery(query="test", top_k=VFGConfig.DEFAULT_TOP_N + 5, rerank=True)
    assert q.top_n == VFGConfig.DEFAULT_TOP_N


def test_search_query_rerank_true_explicit_top_n_less_than_top_k_accepted() -> None:
    """Explicit top_n < top_k is valid when reranking is enabled."""
    q = SearchQuery(query="test", top_k=10, rerank=True, top_n=3)
    assert q.top_n == 3
    assert q.rerank is True


def test_search_query_rerank_true_top_n_equal_to_top_k_raises_validation_error() -> (
    None
):
    """Explicit top_n == top_k raises ValidationError when reranking is enabled."""
    with pytest.raises(ValidationError):
        SearchQuery(query="test", top_k=5, rerank=True, top_n=5)


def test_search_query_rerank_true_top_n_greater_than_top_k_raises_validation_error() -> (
    None
):
    """Explicit top_n > top_k raises ValidationError when reranking is enabled."""
    with pytest.raises(ValidationError):
        SearchQuery(query="test", top_k=5, rerank=True, top_n=6)


def test_search_query_rerank_false_top_n_gte_top_k_does_not_raise() -> None:
    """When rerank=False, top_n >= top_k is not an error (top_n is unused)."""
    q = SearchQuery(query="test", top_k=3, rerank=False, top_n=10)
    assert q.rerank is False
    assert q.top_n == 10


# =============================================================================
# SearchQuery Model: top_n Field Validation
# =============================================================================


def test_search_query_top_n_at_minimum_is_accepted() -> None:
    """top_n equal to MIN_TOP_N is valid."""
    q = SearchQuery(query="test", rerank=False, top_n=VFGConfig.MIN_TOP_N)
    assert q.top_n == VFGConfig.MIN_TOP_N


def test_search_query_top_n_below_minimum_raises_validation_error() -> None:
    """top_n below MIN_TOP_N (0) is rejected by field-level validation."""
    with pytest.raises(ValidationError):
        SearchQuery(query="test", top_n=VFGConfig.MIN_TOP_N - 1)


def test_search_query_top_n_at_maximum_is_accepted() -> None:
    """top_n equal to MAX_TOP_N is valid when reranking disabled."""
    q = SearchQuery(query="test", rerank=False, top_n=VFGConfig.MAX_TOP_N)
    assert q.top_n == VFGConfig.MAX_TOP_N


def test_search_query_top_n_above_maximum_raises_validation_error() -> None:
    """top_n above MAX_TOP_N is rejected by field-level validation."""
    with pytest.raises(ValidationError):
        SearchQuery(query="test", top_n=VFGConfig.MAX_TOP_N + 1)


# =============================================================================
# VectorEngine._sigmoid
# =============================================================================


def test_sigmoid_zero_input_returns_half(vector_engine) -> None:
    """Sigmoid of 0.0 is exactly 0.5."""
    assert vector_engine._sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_large_positive_returns_near_one(vector_engine) -> None:
    """Sigmoid of a very large positive logit approaches 1.0."""
    assert vector_engine._sigmoid(100.0) > 0.999


def test_sigmoid_large_negative_returns_near_zero(vector_engine) -> None:
    """Sigmoid of a very large negative logit approaches 0.0."""
    assert vector_engine._sigmoid(-100.0) < 0.001


def test_sigmoid_output_is_strictly_between_zero_and_one(vector_engine) -> None:
    """Sigmoid output is always in the open interval (0, 1)."""
    for logit in (-10.0, -1.0, 0.0, 1.0, 10.0):
        result = vector_engine._sigmoid(logit)
        assert 0.0 < result < 1.0, f"sigmoid({logit}) = {result} is out of (0, 1)"


def test_sigmoid_is_monotonically_increasing(vector_engine) -> None:
    """Sigmoid is monotonically increasing: higher input → higher output."""
    logits = [-5.0, -1.0, 0.0, 1.0, 5.0]
    scores = [vector_engine._sigmoid(x) for x in logits]
    for i in range(len(scores) - 1):
        assert scores[i] < scores[i + 1]


# =============================================================================
# VectorEngine._rerank
# =============================================================================


def _make_results(n: int, score: float = 0.5) -> list[SearchResult]:
    """Helper: create n SearchResult objects with uniform dummy scores."""
    return [
        SearchResult(id=str(i), content=f"document about topic {i}", score=score)
        for i in range(n)
    ]


def test_rerank_returns_exactly_top_n_results(vector_engine) -> None:
    """_rerank returns at most top_n results."""
    results = _make_results(10)
    reranked = vector_engine._rerank("topic", results, top_n=4)
    assert len(reranked) == 4


def test_rerank_returns_all_when_top_n_exceeds_candidate_count(vector_engine) -> None:
    """_rerank returns all candidates when top_n > len(results)."""
    results = _make_results(3)
    reranked = vector_engine._rerank("topic", results, top_n=10)
    assert len(reranked) == 3


def test_rerank_results_are_sorted_descending_by_score(vector_engine) -> None:
    """_rerank output is sorted by score in descending order."""
    results = _make_results(8)
    reranked = vector_engine._rerank("document", results, top_n=8)
    for i in range(len(reranked) - 1):
        assert reranked[i].score >= reranked[i + 1].score


def test_rerank_scores_are_between_zero_and_one(vector_engine) -> None:
    """All reranked scores are sigmoid-normalised into (0, 1)."""
    results = _make_results(5)
    reranked = vector_engine._rerank("document topic", results, top_n=5)
    for r in reranked:
        assert 0.0 < r.score < 1.0, f"score {r.score} is outside (0, 1)"


def test_rerank_replaces_initial_scores_with_cross_encoder_scores(
    vector_engine,
) -> None:
    """Scores after reranking differ from the initial cosine scores passed in."""
    # Use an extreme initial score that the cross-encoder is extremely unlikely to reproduce
    initial_score = 0.12345
    results = [
        SearchResult(
            id="0", content="Python is a programming language", score=initial_score
        )
    ]
    reranked = vector_engine._rerank("Python programming language", results, top_n=1)
    assert reranked[0].score != pytest.approx(initial_score, abs=1e-4)


def test_rerank_places_most_relevant_document_first_despite_inverted_cosine_scores(
    vector_engine,
) -> None:
    """Cross-encoder corrects ordering even when cosine scores disagree with relevance.

    The irrelevant document (volcanoes) is given a deliberately high initial cosine
    score and the relevant document (Python) a low one. After reranking, the cross-encoder
    should place the Python document first for a Python-related query.
    """
    results = [
        SearchResult(
            id="irrelevant", content="Volcanoes erupt lava and ash", score=0.95
        ),
        SearchResult(
            id="relevant", content="Python is a programming language", score=0.10
        ),
    ]
    reranked = vector_engine._rerank("Python programming language", results, top_n=2)
    assert reranked[0].id == "relevant"


def test_rerank_single_result_returns_one_item(vector_engine) -> None:
    """_rerank handles a single-element candidate list without error."""
    results = [SearchResult(id="0", content="solo document", score=0.9)]
    reranked = vector_engine._rerank("solo", results, top_n=1)
    assert len(reranked) == 1


# =============================================================================
# VectorEngine.search with rerank=True
# =============================================================================


def test_engine_search_with_rerank_returns_top_n_results(vector_engine) -> None:
    """search(rerank=True) returns exactly top_n results."""
    docs = [
        {"content": f"document about science topic {i}", "metadata": {}}
        for i in range(10)
    ]
    vector_engine.add_docs(docs)
    results = vector_engine.search("science", top_k=10, rerank=True, top_n=3)
    assert len(results) == 3


def test_engine_search_with_rerank_scores_are_in_valid_range(vector_engine) -> None:
    """Reranked result scores are all within [0, 1]."""
    docs = [{"content": f"article {i}", "metadata": {}} for i in range(5)]
    vector_engine.add_docs(docs)
    results = vector_engine.search("article", top_k=5, rerank=True, top_n=3)
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_engine_search_with_rerank_results_sorted_descending_by_score(
    vector_engine,
) -> None:
    """Reranked search results are ordered by score descending."""
    docs = [
        {"content": c, "metadata": {}}
        for c in [
            "Python is a programming language",
            "machine learning and neural networks",
            "database management systems",
            "web development and REST APIs",
            "data science with pandas and numpy",
        ]
    ]
    vector_engine.add_docs(docs)
    results = vector_engine.search("Python programming", top_k=5, rerank=True, top_n=4)
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_engine_search_with_rerank_true_raises_when_top_n_equals_top_k(
    vector_engine,
) -> None:
    """search raises ValueError when rerank=True and top_n == top_k."""
    docs = [{"content": f"doc {i}", "metadata": {}} for i in range(5)]
    vector_engine.add_docs(docs)
    with pytest.raises(ValueError, match="top_n"):
        vector_engine.search("doc", top_k=5, rerank=True, top_n=5)


def test_engine_search_with_rerank_true_raises_when_top_n_exceeds_top_k(
    vector_engine,
) -> None:
    """search raises ValueError when rerank=True and top_n > top_k."""
    docs = [{"content": f"doc {i}", "metadata": {}} for i in range(5)]
    vector_engine.add_docs(docs)
    with pytest.raises(ValueError, match="top_n"):
        vector_engine.search("doc", top_k=3, rerank=True, top_n=5)


def test_engine_search_with_rerank_false_does_not_raise_when_top_n_exceeds_top_k(
    vector_engine,
) -> None:
    """search does not raise when rerank=False even if top_n > top_k."""
    docs = [{"content": f"doc {i}", "metadata": {}} for i in range(5)]
    vector_engine.add_docs(docs)
    results = vector_engine.search("doc", top_k=3, rerank=False, top_n=10)
    assert len(results) == 3


def test_engine_search_with_rerank_returns_fewer_results_than_top_k(
    vector_engine,
) -> None:
    """Reranking reduces the result count from top_k to top_n."""
    docs = [{"content": f"document {i}", "metadata": {}} for i in range(10)]
    vector_engine.add_docs(docs)
    top_k, top_n = 10, 3
    reranked = vector_engine.search("document", top_k=top_k, rerank=True, top_n=top_n)
    non_reranked = vector_engine.search("document", top_k=top_k, rerank=False)
    assert len(non_reranked) == top_k
    assert len(reranked) == top_n


def test_engine_search_with_rerank_on_empty_index_returns_empty(vector_engine) -> None:
    """search(rerank=True) on an empty collection returns an empty list."""
    results = vector_engine.search("anything", top_k=10, rerank=True, top_n=5)
    assert results == []
