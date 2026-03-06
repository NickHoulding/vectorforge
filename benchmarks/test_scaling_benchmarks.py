"""Scaling benchmarks for VectorForge.

Tests performance degradation and scaling behavior:
- Query latency vs index size
- Indexing speed vs index size
- Memory growth patterns
- End-to-end workflows at scale
- Realistic usage scenarios

Metrics tracked:
- Performance degradation curves
- Memory usage at different scales
- Throughput scaling
- Real-world workflow performance

Notes on removed/reframed tests vs. the pre-ChromaDB suite:
- ``test_e2e_workflow_small_scale`` / ``test_e2e_workflow_medium_scale``: removed
  ``engine.save()`` / ``engine.load()`` calls — these methods no longer exist.
  The workflows now just cover indexing + search, which is still meaningful.
- ``test_performance_before_compaction`` / ``test_performance_after_compaction``:
  replaced with ``test_search_before_deletions`` / ``test_search_after_deletions``.
  ChromaDB handles compaction automatically; ``deleted_docs`` and ``documents``
  attributes no longer exist on VectorEngine. The new tests measure whether batch
  deletion meaningfully affects subsequent query latency.
- ``test_top_k_scaling``: ``engine.documents`` is gone; replaced by
  ``engine.collection.count()`` to check available docs before skipping.
- All inline ``VectorEngine()`` calls replaced with ``make_ephemeral_engine()``.
"""

import os
from typing import Callable

import psutil
import pytest

from benchmarks.conftest import (
    SCALES,
    _bulk_populate,
    generate_document,
    generate_documents,
    generate_file_chunk,
)
from vectorforge.vector_engine import VectorEngine

# ============================================================================
# Query Latency Scaling
# ============================================================================


@pytest.mark.parametrize("scale", ["tiny", "small", "medium"])
def test_query_latency_scaling(
    benchmark,
    scale: str,
    simple_queries: list[str],
    make_ephemeral_engine: Callable[[], VectorEngine],
):
    """Measure query latency across different index sizes."""
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES[scale]))

    query = simple_queries[0]
    benchmark(engine.search, query=query, top_k=10)


@pytest.mark.slow
def test_query_latency_scaling_large(
    benchmark,
    simple_queries: list[str],
    make_ephemeral_engine: Callable[[], VectorEngine],
):
    """Measure query latency on large index (10,000 docs)."""
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["large"]))

    query = simple_queries[0]
    benchmark(engine.search, query=query, top_k=10)


# ============================================================================
# Indexing Speed Scaling
# ============================================================================


@pytest.mark.parametrize("scale", ["tiny", "small", "medium"])
def test_indexing_speed_scaling(
    benchmark,
    scale: str,
    make_ephemeral_engine: Callable[[], VectorEngine],
):
    """Measure the marginal cost of adding one more doc at different index sizes."""
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES[scale]))

    new_doc = generate_document(99999)

    def add_doc():
        engine.add_doc(new_doc["content"], new_doc["metadata"])

    benchmark(add_doc)


# ============================================================================
# Memory Scaling
# ============================================================================


@pytest.mark.parametrize("doc_count", [100, 500])
def test_memory_scaling(
    benchmark,
    doc_count: int,
    make_ephemeral_engine: Callable[[], VectorEngine],
):
    """Measure RSS memory usage growth at different index sizes."""

    def build_index_and_measure():
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        engine = make_ephemeral_engine()
        for doc in generate_documents(doc_count):
            engine.add_doc(doc["content"], doc["metadata"])

        mem_after = process.memory_info().rss / (1024 * 1024)
        return mem_after - mem_before

    benchmark.pedantic(build_index_and_measure, iterations=1, rounds=2)


@pytest.mark.slow
@pytest.mark.parametrize("doc_count", [1000, 5000, 10000])
def test_memory_scaling_large(
    benchmark,
    doc_count: int,
    make_ephemeral_engine: Callable[[], VectorEngine],
):
    """Measure RSS memory usage growth at large index sizes."""

    def build_index_and_measure():
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        engine = make_ephemeral_engine()
        for doc in generate_documents(doc_count):
            engine.add_doc(doc["content"], doc["metadata"])

        mem_after = process.memory_info().rss / (1024 * 1024)
        return mem_after - mem_before

    benchmark.pedantic(build_index_and_measure, iterations=1, rounds=2)


# ============================================================================
# Batch Size Scaling
# ============================================================================


@pytest.mark.parametrize("batch_size", [10, 50, 100, 500])
def test_batch_insertion_scaling(
    benchmark,
    batch_size: int,
    make_ephemeral_engine: Callable[[], VectorEngine],
):
    """Measure batch insertion performance at different batch sizes."""
    docs = generate_documents(batch_size)

    def batch_insert():
        engine = make_ephemeral_engine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=1, rounds=2)


@pytest.mark.slow
def test_batch_insertion_scaling_1000(
    benchmark,
    make_ephemeral_engine: Callable[[], VectorEngine],
):
    """Measure batch insertion performance at 1,000 documents."""
    docs = generate_documents(1000)

    def batch_insert():
        engine = make_ephemeral_engine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=1, rounds=2)


# ============================================================================
# Top-K Scaling
# ============================================================================


@pytest.mark.parametrize("top_k", [1, 5, 10, 50, 100, 500])
def test_top_k_scaling(
    benchmark,
    engine_medium: VectorEngine,
    top_k: int,
    simple_queries: list[str],
):
    """Measure search performance with different top_k values."""
    query = simple_queries[0]

    if top_k > engine_medium.collection.count():
        pytest.skip(f"Not enough documents for top_k={top_k}")

    benchmark(engine_medium.search, query=query, top_k=top_k)


# ============================================================================
# End-to-End Workflow Scaling
# ============================================================================


def test_e2e_workflow_small_scale(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """End-to-end workflow: build small index then run repeated searches."""
    docs = generate_documents(100)

    def workflow():
        engine = make_ephemeral_engine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

        for _ in range(10):
            engine.search("test query", top_k=10)

    benchmark.pedantic(workflow, iterations=1, rounds=3)


@pytest.mark.slow
def test_e2e_workflow_medium_scale(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """End-to-end workflow: build medium index then run repeated searches."""
    docs = generate_documents(1000)

    def workflow():
        engine = make_ephemeral_engine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

        for _ in range(10):
            engine.search("test query", top_k=10)

    benchmark.pedantic(workflow, iterations=1, rounds=2)


# ============================================================================
# Realistic Use Case Scenarios
# ============================================================================


def test_realistic_document_search_workflow(benchmark, engine_medium: VectorEngine):
    """Simulate realistic document search workflow."""
    queries = [
        "machine learning algorithms",
        "data science techniques",
        "python programming",
        "neural networks",
        "database optimization",
    ]

    def search_workflow():
        results_list = []
        for query in queries:
            results = engine_medium.search(query, top_k=5)
            results_list.extend(results)

    benchmark(search_workflow)


def test_realistic_file_upload_workflow(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Simulate realistic file upload and processing workflow."""

    def upload_workflow():
        engine = make_ephemeral_engine()
        for file_idx in range(3):
            filename = f"document_{file_idx}.pdf"
            for chunk_idx in range(20):
                chunk = generate_file_chunk(
                    chunk_id=file_idx * 20 + chunk_idx,
                    source_file=filename,
                    chunk_index=chunk_idx,
                )
                engine.add_doc(chunk["content"], chunk["metadata"])

            engine.search("test", top_k=5)

    benchmark.pedantic(upload_workflow, iterations=1, rounds=3)


def test_realistic_crud_workflow(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Simulate realistic CRUD operations workflow."""

    def crud_workflow():
        engine = make_ephemeral_engine()

        docs = generate_documents(100)
        doc_ids = []
        for doc in docs:
            doc_id = engine.add_doc(doc["content"], doc["metadata"])
            doc_ids.append(doc_id)

        for _ in range(10):
            engine.search("test query", top_k=5)

        for i in range(10):
            engine.delete_doc(doc_ids[i])
            new_doc = generate_document(1000 + i)
            engine.add_doc(new_doc["content"], new_doc["metadata"])

        for i in range(10, 30):
            engine.delete_doc(doc_ids[i])

        engine.search("final query", top_k=10)

    benchmark.pedantic(crud_workflow, iterations=1, rounds=3)


# ============================================================================
# Deletion Impact on Search Performance
# ============================================================================


def test_search_before_deletions(
    benchmark,
    make_ephemeral_engine: Callable[[], VectorEngine],
    simple_queries: list[str],
):
    """Baseline: search latency on a medium index before any deletions."""
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["medium"]))

    query = simple_queries[0]
    benchmark(engine.search, query=query, top_k=10)


def test_search_after_bulk_deletions(
    benchmark,
    make_ephemeral_engine: Callable[[], VectorEngine],
    simple_queries: list[str],
):
    """Search latency on a medium index after deleting 30% of documents.

    Uses ``collection.get()`` to retrieve current IDs since VectorEngine no
    longer maintains an in-memory ``documents`` dict.
    """
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["medium"]))

    all_ids = engine.collection.get(include=[])["ids"]
    delete_count = len(all_ids) * 3 // 10
    for doc_id in all_ids[:delete_count]:
        engine.delete_doc(doc_id)

    query = simple_queries[0]
    benchmark(engine.search, query=query, top_k=10)


# ============================================================================
# Concurrent Query Simulation
# ============================================================================


def test_sequential_concurrent_queries(benchmark, engine_medium: VectorEngine):
    """Simulate concurrent queries (sequential execution)."""
    queries = [f"query {i}" for i in range(20)]

    def run_queries():
        for query in queries:
            engine_medium.search(query, top_k=10)

    benchmark(run_queries)


# ============================================================================
# Long-Running Session
# ============================================================================


@pytest.mark.slow
def test_long_running_session(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Simulate long-running session with mixed operations."""

    def long_session():
        engine = make_ephemeral_engine()

        docs = generate_documents(1000)
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

        all_ids = engine.collection.get(include=[])["ids"]

        for i in range(50):
            engine.search(f"query {i}", top_k=5)

            if i % 5 == 0:
                new_doc = generate_document(2000 + i)
                engine.add_doc(new_doc["content"], new_doc["metadata"])

            if i % 10 == 0 and i < len(all_ids):
                engine.delete_doc(all_ids[i])

    benchmark.pedantic(long_session, iterations=1, rounds=2)
