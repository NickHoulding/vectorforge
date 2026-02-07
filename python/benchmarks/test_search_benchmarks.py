"""Search performance benchmarks for VectorForge.

Tests query latency across different:
- Index sizes (tiny, small, medium, large, xlarge)
- Query complexities (simple, medium, complex)
- Top-k values (1, 5, 10, 50, 100)
- Cold vs warm queries

Metrics tracked:
- Query latency (mean, min, max, p50, p95, p99)
- Queries per second
- Memory usage during search
"""

import pytest

from benchmarks.conftest import SCALES, generate_documents
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
    # Generate 100 queries
    queries = simple_queries * 10  # Reuse queries

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
# Cold vs Warm Performance
# ============================================================================


def test_search_cold_engine(benchmark, simple_queries: list[str]):
    """Benchmark first search on a new engine (cold start)."""

    def run_cold_search():
        # Create fresh engine for each iteration to measure cold start
        engine = VectorEngine()
        docs = generate_documents(SCALES["small"])
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])
        engine.search(query=simple_queries[0], top_k=10)

    # Reduced rounds for cold start test
    benchmark.pedantic(run_cold_search, iterations=3, rounds=3)


def test_search_warm_engine(
    benchmark, engine_small: VectorEngine, simple_queries: list[str]
):
    """Benchmark search on warm engine (model already loaded)."""
    # First, warm up the engine with a few queries
    for _ in range(5):
        engine_small.search(query=simple_queries[0], top_k=10)

    # Now benchmark
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

    # Run for a fixed number of iterations to calculate QPS
    result = benchmark.pedantic(
        engine_medium.search,
        args=(query,),
        kwargs={"top_k": 10},
        iterations=100,
        rounds=5,
    )

    # QPS will be calculated from benchmark stats


# ============================================================================
# Search After Deletions
# ============================================================================


def test_search_after_deletions(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark search performance after deleting 25% of documents."""
    # Delete 25% of documents
    doc_ids = list(engine_medium.documents.keys())
    delete_count = len(doc_ids) // 4
    for doc_id in doc_ids[:delete_count]:
        engine_medium.delete_doc(doc_id)

    # Benchmark search with deleted docs
    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=10)


def test_search_after_compaction(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Benchmark search performance after compaction."""
    # Delete enough docs to trigger compaction
    doc_ids = list(engine_medium.documents.keys())
    delete_count = len(doc_ids) // 3  # Delete 33% to exceed 25% threshold
    for doc_id in doc_ids[:delete_count]:
        engine_medium.delete_doc(doc_id)

    # This should have triggered compaction
    assert (
        len(engine_medium.deleted_docs) == 0
    ), "Compaction should have cleared deleted_docs"

    # Benchmark search after compaction
    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=10)
