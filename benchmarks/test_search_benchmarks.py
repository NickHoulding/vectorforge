"""Search performance benchmarks for VectorForge.

Tests query latency across different:
- Index sizes (tiny, small, medium, large, xlarge)
- Query complexities (simple, medium, complex)
- Top-k values (1, 5, 10, 50, 100)
- Warm queries and throughput measurement
- Filtered vs. unfiltered search

Metrics tracked:
- Query latency (mean, min, max, p50, p95, p99)
- Queries per second
- Memory usage during search

Notes on removed tests vs. the pre-ChromaDB suite:
- ``test_search_cold_engine``: removed — it measured model loading time, not search
  latency. Model loading is now handled by CollectionManager and not directly
  benchmarkable through VectorEngine alone.
- ``test_search_after_compaction`` / ``test_search_before_compaction``: removed —
  ChromaDB manages compaction automatically; there is no explicit compaction call
  and no ``deleted_docs`` attribute on VectorEngine.
- Filtered search tests previously called ``VectorEngine()`` directly; they now
  use ``make_ephemeral_engine`` to build properly constructed engines.
"""

from typing import Callable

import pytest

from benchmarks.conftest import (
    SCALES,
    _bulk_populate,
    generate_documents,
)
from vectorforge.vector_engine import VectorEngine

# ============================================================================
# Search Latency - Varying Index Size
# ============================================================================


@pytest.mark.scale(size="tiny")
def test_search_latency_tiny(
    benchmark, engine_tiny: VectorEngine, simple_queries: list[str]
):
    """Benchmark search latency on tiny index (10 docs)."""
    query = simple_queries[0]
    benchmark(engine_tiny.search, query=query, top_k=10)


@pytest.mark.scale(size="small")
def test_search_latency_small(
    benchmark, engine_small: VectorEngine, simple_queries: list[str]
):
    """Benchmark search latency on small index (100 docs)."""
    query = simple_queries[0]
    benchmark(engine_small.search, query=query, top_k=10)


@pytest.mark.scale(size="medium")
def test_search_latency_medium(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark search latency on medium index (1,000 docs)."""
    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=10)


@pytest.mark.scale(size="large")
@pytest.mark.slow
def test_search_latency_large(
    benchmark, engine_large: VectorEngine, simple_queries: list[str]
):
    """Benchmark search latency on large index (10,000 docs)."""
    query = simple_queries[0]
    benchmark(engine_large.search, query=query, top_k=10)


@pytest.mark.scale(size="xlarge")
@pytest.mark.slow
def test_search_latency_xlarge(
    benchmark, engine_xlarge: VectorEngine, simple_queries: list[str]
):
    """Benchmark search latency on extra-large index (50,000 docs)."""
    query = simple_queries[0]
    benchmark(engine_xlarge.search, query=query, top_k=10)


# ============================================================================
# Search Latency - Varying Top-K
# ============================================================================


def test_search_top_k_1(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark search with top_k=1."""
    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=1)


def test_search_top_k_5(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark search with top_k=5."""
    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=5)


def test_search_top_k_10(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark search with top_k=10."""
    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=10)


def test_search_top_k_50(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark search with top_k=50."""
    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=50)


def test_search_top_k_100(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark search with top_k=100."""
    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=100)


# ============================================================================
# Search Latency - Query Complexity
# ============================================================================


def test_search_simple_query(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark search with simple single-word query."""
    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=10)


def test_search_medium_query(
    benchmark, engine_medium: VectorEngine, medium_queries: list[str]
):
    """Benchmark search with medium complexity query."""
    query = medium_queries[0]
    benchmark(engine_medium.search, query=query, top_k=10)


def test_search_complex_query(
    benchmark, engine_medium: VectorEngine, complex_queries: list[str]
):
    """Benchmark search with complex sentence query."""
    query = complex_queries[0]
    benchmark(engine_medium.search, query=query, top_k=10)


# ============================================================================
# Batch Search Performance
# ============================================================================


def test_search_batch_10_queries(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark batch of 10 search queries."""

    def run_batch():
        for query in simple_queries[:10]:
            engine_medium.search(query=query, top_k=10)

    benchmark(run_batch)


def test_search_batch_100_queries(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark batch of 100 search queries."""
    queries = simple_queries * 10

    def run_batch():
        for query in queries:
            engine_medium.search(query=query, top_k=10)

    benchmark(run_batch)


# ============================================================================
# Empty Index Edge Case
# ============================================================================


def test_search_empty_index(
    benchmark, empty_engine: VectorEngine, simple_queries: list[str]
):
    """Benchmark search on empty index."""
    query = simple_queries[0]
    benchmark(empty_engine.search, query=query, top_k=10)


# ============================================================================
# Warm Query Performance
# ============================================================================


def test_search_warm_engine(
    benchmark, engine_small: VectorEngine, simple_queries: list[str]
):
    """Benchmark search on warm engine (several prior queries already executed)."""
    for _ in range(5):
        engine_small.search(query=simple_queries[0], top_k=10)

    query = simple_queries[0]
    benchmark(engine_small.search, query=query, top_k=10)


# ============================================================================
# Throughput Measurement
# ============================================================================


def test_search_throughput_qps(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Measure queries per second (QPS) throughput."""
    query = simple_queries[0]

    benchmark.pedantic(
        engine_medium.search,
        args=(query,),
        kwargs={"top_k": 10},
        iterations=100,
        rounds=5,
    )


# ============================================================================
# Search After Deletions
# ============================================================================


def test_search_after_deletions(
    benchmark,
    make_ephemeral_engine: Callable[[], VectorEngine],
    simple_queries: list[str],
):
    """Benchmark search performance after deleting 25% of documents.

    Populates a medium index, deletes 25% of docs, then benchmarks search.
    Uses collection.get() to retrieve current doc IDs since VectorEngine no
    longer maintains an in-memory ``documents`` dict.
    """
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["medium"]))

    all_ids = engine.collection.get(include=[])["ids"]
    delete_count = len(all_ids) // 4
    for doc_id in all_ids[:delete_count]:
        engine.delete_doc(doc_id)

    query = simple_queries[0]
    benchmark(engine.search, query=query, top_k=10)


# ============================================================================
# Search with Filters - Performance Impact
# ============================================================================


def test_search_with_filters_small_dataset(
    benchmark,
    make_ephemeral_engine: Callable[[], VectorEngine],
    simple_queries: list[str],
):
    """Benchmark filtered search on small dataset (100 docs)."""
    engine = make_ephemeral_engine()

    docs = [
        {
            "content": f"Document about {simple_queries[file_num % len(simple_queries)]}",
            "metadata": {
                "source_file": f"file_{file_num}.pdf",
                "chunk_index": chunk_num,
            },
        }
        for file_num in range(10)
        for chunk_num in range(10)
    ]
    _bulk_populate(engine, docs)

    query = simple_queries[0]
    filters = {"source_file": "file_5.pdf"}
    benchmark(engine.search, query=query, top_k=10, filters=filters)


@pytest.mark.scale(size="medium")
def test_search_with_filters_medium_dataset(
    benchmark,
    make_ephemeral_engine: Callable[[], VectorEngine],
    simple_queries: list[str],
):
    """Benchmark filtered search on medium dataset (1,000 docs)."""
    engine = make_ephemeral_engine()

    docs = [
        {
            "content": f"Document about {simple_queries[file_num % len(simple_queries)]}",
            "metadata": {
                "source_file": f"file_{file_num}.pdf",
                "chunk_index": chunk_num,
            },
        }
        for file_num in range(50)
        for chunk_num in range(20)
    ]
    _bulk_populate(engine, docs)

    query = simple_queries[0]
    filters = {"source_file": "file_25.pdf", "chunk_index": 10}
    benchmark(engine.search, query=query, top_k=10, filters=filters)


@pytest.mark.scale(size="large")
@pytest.mark.slow
def test_search_with_filters_large_dataset(
    benchmark,
    make_ephemeral_engine: Callable[[], VectorEngine],
    simple_queries: list[str],
):
    """Benchmark filtered search on large dataset (10,000 docs)."""
    engine = make_ephemeral_engine()

    docs = [
        {
            "content": f"Document about {simple_queries[file_num % len(simple_queries)]}",
            "metadata": {
                "source_file": f"file_{file_num}.pdf",
                "chunk_index": chunk_num,
                "category": simple_queries[chunk_num % len(simple_queries)],
            },
        }
        for file_num in range(100)
        for chunk_num in range(100)
    ]
    _bulk_populate(engine, docs)

    query = simple_queries[0]
    filters = {
        "source_file": "file_50.pdf",
        "chunk_index": 50,
        "category": simple_queries[5],
    }
    benchmark(engine.search, query=query, top_k=10, filters=filters)
