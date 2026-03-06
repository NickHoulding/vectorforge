"""Indexing performance benchmarks for VectorForge.

Tests document addition throughput across different scenarios:
- Single document insertion at various index sizes
- Batch insertion (varying batch sizes)
- Document size impact on embedding time
- File-based chunked insertion
- Interleaved add + search operations
- Metadata size impact

Metrics tracked:
- Documents per second
- Time per document
- Batch insertion throughput
- Memory growth during insertion

Notes on removed tests vs. the pre-ChromaDB suite:
- ``test_add_doc_triggering_compaction``: removed — ChromaDB manages compaction
  automatically and there is no explicit trigger or ``documents`` dict on
  VectorEngine.  The concept does not map to the ChromaDB-backed architecture.
- ``test_batch_insert_1000_docs`` / ``test_batch_insert_10000_docs``: previously
  called ``VectorEngine()`` directly; now use ``make_ephemeral_engine`` inside the
  timed function so each iteration starts from a clean, empty engine.
"""

from typing import Callable

import pytest
from faker import Faker

from benchmarks.conftest import (
    SCALES,
    _bulk_populate,
    generate_document,
    generate_documents,
    generate_file_chunk,
)
from vectorforge.vector_engine import VectorEngine

# ============================================================================
# Single Document Insertion
# ============================================================================


def test_add_single_doc_empty_index(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark adding a single document to empty index."""
    doc = generate_document(0)

    def add_doc():
        engine = make_ephemeral_engine()
        engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(add_doc, iterations=1, rounds=5)


def test_add_single_doc_small_index(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark adding a single document to small index (100 docs)."""
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["small"]))
    doc = generate_document(1000)

    def add_doc():
        engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(add_doc, iterations=1, rounds=5)


def test_add_single_doc_medium_index(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark adding a single document to medium index (1,000 docs)."""
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["medium"]))
    doc = generate_document(10000)

    def add_doc():
        engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(add_doc, iterations=1, rounds=5)


@pytest.mark.slow
def test_add_single_doc_large_index(benchmark, engine_large: VectorEngine):
    """Benchmark adding a single document to large index (10,000 docs)."""
    doc = generate_document(100000)

    def add_doc():
        engine_large.add_doc(doc["content"], doc["metadata"])

    benchmark(add_doc)


# ============================================================================
# Batch Insertion
# ============================================================================


def test_batch_insert_10_docs(benchmark, empty_engine: VectorEngine):
    """Benchmark batch insertion of 10 documents."""
    docs = generate_documents(10)

    def batch_insert():
        for doc in docs:
            empty_engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=1, rounds=5)


def test_batch_insert_100_docs(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark batch insertion of 100 documents into a fresh engine."""
    docs = generate_documents(100)

    def batch_insert():
        engine = make_ephemeral_engine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=1, rounds=3)


def test_batch_insert_1000_docs(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark batch insertion of 1,000 documents into a fresh engine."""
    docs = generate_documents(1000)

    def batch_insert():
        engine = make_ephemeral_engine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=1, rounds=2)


@pytest.mark.slow
def test_batch_insert_10000_docs(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark batch insertion of 10,000 documents into a fresh engine."""
    docs = generate_documents(10000)

    def batch_insert():
        engine = make_ephemeral_engine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=1, rounds=2)


# ============================================================================
# Document Size Impact
# ============================================================================


def test_add_small_doc(benchmark, make_ephemeral_engine: Callable[[], VectorEngine]):
    """Benchmark adding small document (50-100 chars)."""
    fake = Faker()
    doc = {"content": fake.sentence(), "metadata": {"size": "small"}}

    def add_doc():
        engine = make_ephemeral_engine()
        engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(add_doc, iterations=1, rounds=5)


def test_add_medium_doc(benchmark, make_ephemeral_engine: Callable[[], VectorEngine]):
    """Benchmark adding medium document (500-1000 chars)."""
    fake = Faker()
    content = " ".join([fake.sentence() for _ in range(20)])
    doc = {"content": content, "metadata": {"size": "medium"}}

    def add_doc():
        engine = make_ephemeral_engine()
        engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(add_doc, iterations=1, rounds=5)


def test_add_large_doc(benchmark, make_ephemeral_engine: Callable[[], VectorEngine]):
    """Benchmark adding large document (2000-5000 chars)."""
    fake = Faker()
    content = " ".join([fake.sentence() for _ in range(100)])
    doc = {"content": content, "metadata": {"size": "large"}}

    def add_doc():
        engine = make_ephemeral_engine()
        engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(add_doc, iterations=1, rounds=5)


# ============================================================================
# File Chunking Simulation
# ============================================================================


def test_add_file_chunks_10(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark adding 10 chunks from a file into a fresh engine."""
    chunks = [generate_file_chunk(i, "test.pdf", i) for i in range(10)]

    def add_chunks():
        engine = make_ephemeral_engine()
        for chunk in chunks:
            engine.add_doc(chunk["content"], chunk["metadata"])

    benchmark.pedantic(add_chunks, iterations=1, rounds=5)


def test_add_file_chunks_50(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark adding 50 chunks from a file into a fresh engine."""
    chunks = [generate_file_chunk(i, "large.pdf", i) for i in range(50)]

    def add_chunks():
        engine = make_ephemeral_engine()
        for chunk in chunks:
            engine.add_doc(chunk["content"], chunk["metadata"])

    benchmark.pedantic(add_chunks, iterations=1, rounds=3)


def test_add_file_chunks_100(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark adding 100 chunks from a very large file into a fresh engine."""
    chunks = [generate_file_chunk(i, "huge.pdf", i) for i in range(100)]

    def add_chunks():
        engine = make_ephemeral_engine()
        for chunk in chunks:
            engine.add_doc(chunk["content"], chunk["metadata"])

    benchmark.pedantic(add_chunks, iterations=3, rounds=2)


# ============================================================================
# Throughput Measurement
# ============================================================================


def test_indexing_throughput_docs_per_second(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Measure indexing throughput in documents per second."""
    docs = generate_documents(100)

    def index_batch():
        engine = make_ephemeral_engine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(index_batch, iterations=1, rounds=3)


# ============================================================================
# Concurrent-style Insertion (Sequential)
# ============================================================================


def test_interleaved_add_search(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark interleaved document addition and searching."""
    docs = generate_documents(20)

    def interleaved_ops():
        engine = make_ephemeral_engine()
        for i, doc in enumerate(docs):
            engine.add_doc(doc["content"], doc["metadata"])

            if i % 5 == 0 and i > 0:
                engine.search("test query", top_k=5)

    benchmark.pedantic(interleaved_ops, iterations=1, rounds=3)


# ============================================================================
# Metadata Variations
# ============================================================================


def test_add_doc_minimal_metadata(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark adding document with minimal metadata."""
    doc = {"content": "This is a test document with minimal metadata.", "metadata": {}}

    def add_doc():
        engine = make_ephemeral_engine()
        engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(add_doc, iterations=1, rounds=5)


def test_add_doc_rich_metadata(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark adding document with rich metadata."""
    fake = Faker()

    doc = {
        "content": "This is a test document with rich metadata.",
        "metadata": {
            "author": fake.name(),
            "title": fake.sentence(),
            "category": fake.word(),
            "tags": " ".join([fake.word() for _ in range(10)]),
            "created_at": fake.date_time_this_year().isoformat(),
            "modified_at": fake.date_time_this_year().isoformat(),
            "version": fake.random_int(min=1, max=10),
            "priority": fake.random_element(["low", "medium", "high"]),
        },
    }

    def add_doc():
        engine = make_ephemeral_engine()
        engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(add_doc, iterations=1, rounds=5)
