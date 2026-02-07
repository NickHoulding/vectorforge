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
"""

import os
import tempfile

import psutil
import pytest

from benchmarks.conftest import (
    SCALES,
    generate_document,
    generate_documents,
    generate_file_chunk,
)
from vectorforge.vector_engine import VectorEngine

# ============================================================================
# Query Latency Scaling
# ============================================================================


@pytest.mark.parametrize("scale", ["tiny", "small", "medium", "large"])
def test_query_latency_scaling(benchmark, scale: str, simple_queries: list[str]):
    """Measure query latency across different index sizes."""
    # Create engine with appropriate scale
    engine = VectorEngine()
    docs = generate_documents(SCALES[scale])
    for doc in docs:
        engine.add_doc(doc["content"], doc["metadata"])

    query = simple_queries[0]
    benchmark(engine.search, query=query, top_k=10)


# ============================================================================
# Indexing Speed Scaling
# ============================================================================


@pytest.mark.parametrize("scale", ["tiny", "small", "medium"])
def test_indexing_speed_scaling(benchmark, scale: str):
    """Measure indexing speed at different index sizes."""
    # Pre-populate engine
    engine = VectorEngine()
    docs = generate_documents(SCALES[scale])
    for doc in docs:
        engine.add_doc(doc["content"], doc["metadata"])

    # Now benchmark adding one more document
    new_doc = generate_document(99999)

    def add_doc():
        engine.add_doc(new_doc["content"], new_doc["metadata"])

    benchmark(add_doc)


# ============================================================================
# Memory Scaling
# ============================================================================


@pytest.mark.parametrize("doc_count", [100, 500, 1000, 5000, 10000])
def test_memory_scaling(benchmark, doc_count: int):
    """Measure memory usage at different scales."""

    def build_index_and_measure():
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB

        engine = VectorEngine()
        docs = generate_documents(doc_count)
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_used = mem_after - mem_before

        return mem_used

    result = benchmark.pedantic(build_index_and_measure, iterations=1, rounds=3)


# ============================================================================
# Batch Size Scaling
# ============================================================================


@pytest.mark.parametrize("batch_size", [10, 50, 100, 500, 1000])
def test_batch_insertion_scaling(benchmark, batch_size: int):
    """Measure batch insertion performance at different batch sizes."""
    docs = generate_documents(batch_size)

    def batch_insert():
        engine = VectorEngine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    if batch_size >= 500:
        benchmark.pedantic(batch_insert, iterations=1, rounds=3)
    else:
        benchmark.pedantic(batch_insert, iterations=3, rounds=3)


# ============================================================================
# Top-K Scaling
# ============================================================================


@pytest.mark.parametrize("top_k", [1, 5, 10, 50, 100, 500])
def test_top_k_scaling(
    benchmark, engine_medium: VectorEngine, top_k: int, simple_queries: list[str]
):
    """Measure search performance with different top_k values."""
    query = simple_queries[0]

    # Ensure we have enough documents
    if top_k > len(engine_medium.documents):
        pytest.skip(f"Not enough documents for top_k={top_k}")

    benchmark(engine_medium.search, query=query, top_k=top_k)


# ============================================================================
# End-to-End Workflow Scaling
# ============================================================================


def test_e2e_workflow_small_scale(benchmark):
    """End-to-end workflow: build index, search, save, load - small scale."""
    docs = generate_documents(100)

    def workflow():
        # Build index
        engine = VectorEngine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

        # Perform searches
        for _ in range(10):
            engine.search("test query", top_k=10)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save(tmpdir)

            # Load
            new_engine = VectorEngine()
            new_engine.load(tmpdir)

            # Search again
            new_engine.search("another query", top_k=10)

    benchmark.pedantic(workflow, iterations=1, rounds=3)


@pytest.mark.slow
def test_e2e_workflow_medium_scale(benchmark):
    """End-to-end workflow: build index, search, save, load - medium scale."""
    docs = generate_documents(1000)

    def workflow():
        # Build index
        engine = VectorEngine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

        # Perform searches
        for _ in range(10):
            engine.search("test query", top_k=10)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save(tmpdir)

            # Load
            new_engine = VectorEngine()
            new_engine.load(tmpdir)

            # Search again
            new_engine.search("another query", top_k=10)

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


def test_realistic_file_upload_workflow(benchmark, empty_engine: VectorEngine):
    """Simulate realistic file upload and processing workflow."""

    # Simulate uploading 3 files with 20 chunks each
    def upload_workflow():
        for file_idx in range(3):
            filename = f"document_{file_idx}.pdf"
            for chunk_idx in range(20):
                chunk = generate_file_chunk(
                    chunk_id=file_idx * 20 + chunk_idx,
                    source_file=filename,
                    chunk_index=chunk_idx,
                )
                empty_engine.add_doc(chunk["content"], chunk["metadata"])

            # Search after each file
            empty_engine.search("test", top_k=5)

    benchmark.pedantic(upload_workflow, iterations=1, rounds=3)


def test_realistic_crud_workflow(benchmark):
    """Simulate realistic CRUD operations workflow."""

    def crud_workflow():
        engine = VectorEngine()

        # Create: Add 100 documents
        docs = generate_documents(100)
        doc_ids = []
        for doc in docs:
            doc_id = engine.add_doc(doc["content"], doc["metadata"])
            doc_ids.append(doc_id)

        # Read: Search
        for _ in range(10):
            engine.search("test query", top_k=5)

        # Update: Delete and re-add 10 documents
        for i in range(10):
            engine.delete_doc(doc_ids[i])
            new_doc = generate_document(1000 + i)
            engine.add_doc(new_doc["content"], new_doc["metadata"])

        # Delete: Remove 20 documents
        for i in range(10, 30):
            engine.delete_doc(doc_ids[i])

        # Search again
        engine.search("final query", top_k=10)

    benchmark.pedantic(crud_workflow, iterations=1, rounds=3)


# ============================================================================
# Compaction Impact on Performance
# ============================================================================


def test_performance_before_compaction(
    benchmark, engine_medium: VectorEngine, simple_queries: list[str]
):
    """Measure search performance before compaction (with deleted docs)."""
    # Delete 20% of documents (below compaction threshold)
    doc_ids = list(engine_medium.documents.keys())
    for doc_id in doc_ids[:200]:
        engine_medium.deleted_docs.add(
            doc_id
        )  # Add to deleted set without triggering compaction

    query = simple_queries[0]
    benchmark(engine_medium.search, query=query, top_k=10)


def test_performance_after_compaction(benchmark, simple_queries: list[str]):
    """Measure search performance after compaction."""
    # Build index
    engine = VectorEngine()
    docs = generate_documents(SCALES["medium"])
    for doc in docs:
        engine.add_doc(doc["content"], doc["metadata"])

    # Delete docs and trigger compaction
    doc_ids = list(engine.documents.keys())
    for doc_id in doc_ids[:300]:
        engine.delete_doc(doc_id)

    # Compaction should have happened
    assert len(engine.deleted_docs) == 0

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
def test_long_running_session(benchmark):
    """Simulate long-running session with mixed operations."""

    def long_session():
        engine = VectorEngine()

        # Initial index build (1000 docs)
        docs = generate_documents(1000)
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

        doc_ids = list(engine.documents.keys())

        # Perform 50 mixed operations
        for i in range(50):
            # Search
            engine.search(f"query {i}", top_k=5)

            # Add a document every 5 operations
            if i % 5 == 0:
                new_doc = generate_document(2000 + i)
                new_id = engine.add_doc(new_doc["content"], new_doc["metadata"])
                doc_ids.append(new_id)

            # Delete a document every 10 operations
            if i % 10 == 0 and len(doc_ids) > 100:
                engine.delete_doc(doc_ids[i])

    benchmark.pedantic(long_session, iterations=1, rounds=2)
