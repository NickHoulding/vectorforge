"""Indexing performance benchmarks for VectorForge.

Tests document addition throughput across different scenarios:
- Single document insertion
- Batch insertion (varying batch sizes)
- Document size impact
- Insertion with compaction
- File-based chunked insertion

Metrics tracked:
- Documents per second
- Time per document
- Batch insertion throughput
- Memory growth during insertion
"""

import pytest

from benchmarks.conftest import (
    generate_document,
    generate_documents,
    generate_file_chunk,
)
from vectorforge.vector_engine import VectorEngine


# ============================================================================
# Single Document Insertion
# ============================================================================


def test_add_single_doc_empty_index(benchmark, empty_engine: VectorEngine):
    """Benchmark adding a single document to empty index."""
    doc = generate_document(0)

    def add_doc():
        # Note: This will keep adding docs, so engine grows
        # For pure single-add benchmark, we accept this
        empty_engine.add_doc(doc["content"], doc["metadata"])

    benchmark(add_doc)


def test_add_single_doc_small_index(benchmark, engine_small: VectorEngine):
    """Benchmark adding a single document to small index (100 docs)."""
    doc = generate_document(1000)

    def add_doc():
        engine_small.add_doc(doc["content"], doc["metadata"])

    benchmark(add_doc)


def test_add_single_doc_medium_index(benchmark, engine_medium: VectorEngine):
    """Benchmark adding a single document to medium index (1,000 docs)."""
    doc = generate_document(10000)

    def add_doc():
        engine_medium.add_doc(doc["content"], doc["metadata"])

    benchmark(add_doc)


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

    # Use pedantic mode to reset engine between rounds
    benchmark.pedantic(batch_insert, iterations=5, rounds=5)


def test_batch_insert_100_docs(benchmark, empty_engine: VectorEngine):
    """Benchmark batch insertion of 100 documents."""
    docs = generate_documents(100)

    def batch_insert():
        for doc in docs:
            empty_engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=3, rounds=3)


def test_batch_insert_1000_docs(benchmark):
    """Benchmark batch insertion of 1,000 documents."""
    docs = generate_documents(1000)

    def batch_insert():
        engine = VectorEngine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=1, rounds=3)


@pytest.mark.slow
def test_batch_insert_10000_docs(benchmark):
    """Benchmark batch insertion of 10,000 documents."""
    docs = generate_documents(10000)

    def batch_insert():
        engine = VectorEngine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=1, rounds=2)


# ============================================================================
# Document Size Impact
# ============================================================================


def test_add_small_doc(benchmark, empty_engine: VectorEngine):
    """Benchmark adding small document (50-100 chars)."""
    from faker import Faker

    fake = Faker()

    doc = {"content": fake.sentence(), "metadata": {"size": "small"}}

    def add_doc():
        empty_engine.add_doc(doc["content"], doc["metadata"])

    benchmark(add_doc)


def test_add_medium_doc(benchmark, empty_engine: VectorEngine):
    """Benchmark adding medium document (500-1000 chars)."""
    from faker import Faker

    fake = Faker()

    content = " ".join([fake.sentence() for _ in range(20)])
    doc = {"content": content, "metadata": {"size": "medium"}}

    def add_doc():
        empty_engine.add_doc(doc["content"], doc["metadata"])

    benchmark(add_doc)


def test_add_large_doc(benchmark, empty_engine: VectorEngine):
    """Benchmark adding large document (2000-5000 chars)."""
    from faker import Faker

    fake = Faker()

    content = " ".join([fake.sentence() for _ in range(100)])
    doc = {"content": content, "metadata": {"size": "large"}}

    def add_doc():
        empty_engine.add_doc(doc["content"], doc["metadata"])

    benchmark(add_doc)


# ============================================================================
# Insertion with Compaction
# ============================================================================


def test_add_doc_triggering_compaction(benchmark):
    """Benchmark document addition that triggers compaction."""

    def add_with_compaction():
        engine = VectorEngine()

        # Add 100 docs
        docs = generate_documents(100)
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

        # Delete 30% to exceed compaction threshold (25%)
        doc_ids = list(engine.documents.keys())
        for doc_id in doc_ids[:30]:
            engine.delete_doc(doc_id)

        # This addition should trigger compaction
        new_doc = generate_document(1000)
        engine.add_doc(new_doc["content"], new_doc["metadata"])

    benchmark.pedantic(add_with_compaction, iterations=3, rounds=3)


# ============================================================================
# File Chunking Simulation
# ============================================================================


def test_add_file_chunks_10(benchmark, empty_engine: VectorEngine):
    """Benchmark adding 10 chunks from a file."""
    chunks = [generate_file_chunk(i, "test.pdf", i) for i in range(10)]

    def add_chunks():
        for chunk in chunks:
            empty_engine.add_doc(chunk["content"], chunk["metadata"])

    benchmark(add_chunks)


def test_add_file_chunks_50(benchmark, empty_engine: VectorEngine):
    """Benchmark adding 50 chunks from a file."""
    chunks = [generate_file_chunk(i, "large.pdf", i) for i in range(50)]

    def add_chunks():
        for chunk in chunks:
            empty_engine.add_doc(chunk["content"], chunk["metadata"])

    benchmark.pedantic(add_chunks, iterations=5, rounds=3)


def test_add_file_chunks_100(benchmark):
    """Benchmark adding 100 chunks from a very large file."""
    chunks = [generate_file_chunk(i, "huge.pdf", i) for i in range(100)]

    def add_chunks():
        engine = VectorEngine()
        for chunk in chunks:
            engine.add_doc(chunk["content"], chunk["metadata"])

    benchmark.pedantic(add_chunks, iterations=3, rounds=2)


# ============================================================================
# Throughput Measurement
# ============================================================================


def test_indexing_throughput_docs_per_second(benchmark):
    """Measure indexing throughput in documents per second."""
    docs = generate_documents(100)

    def index_batch():
        engine = VectorEngine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    result = benchmark.pedantic(index_batch, iterations=5, rounds=5)

    # Docs per second = 100 docs / time_per_iteration
    # This will be visible in benchmark stats


# ============================================================================
# Concurrent-style Insertion (Sequential)
# ============================================================================


def test_interleaved_add_search(benchmark, empty_engine: VectorEngine):
    """Benchmark interleaved document addition and searching."""
    docs = generate_documents(50)

    def interleaved_ops():
        for i, doc in enumerate(docs):
            empty_engine.add_doc(doc["content"], doc["metadata"])
            # Every 5 docs, perform a search
            if i % 5 == 0 and i > 0:
                empty_engine.search("test query", top_k=5)

    benchmark.pedantic(interleaved_ops, iterations=3, rounds=3)


# ============================================================================
# Metadata Variations
# ============================================================================


def test_add_doc_minimal_metadata(benchmark, empty_engine: VectorEngine):
    """Benchmark adding document with minimal metadata."""
    doc = {"content": "This is a test document with minimal metadata.", "metadata": {}}

    def add_doc():
        empty_engine.add_doc(doc["content"], doc["metadata"])

    benchmark(add_doc)


def test_add_doc_rich_metadata(benchmark, empty_engine: VectorEngine):
    """Benchmark adding document with rich metadata."""
    from faker import Faker

    fake = Faker()

    doc = {
        "content": "This is a test document with rich metadata.",
        "metadata": {
            "author": fake.name(),
            "title": fake.sentence(),
            "category": fake.word(),
            "tags": [fake.word() for _ in range(10)],
            "created_at": fake.date_time_this_year().isoformat(),
            "modified_at": fake.date_time_this_year().isoformat(),
            "version": fake.random_int(min=1, max=10),
            "priority": fake.random_element(["low", "medium", "high"]),
            "nested": {
                "field1": fake.word(),
                "field2": fake.word(),
                "field3": [fake.word() for _ in range(5)],
            },
        },
    }

    def add_doc():
        empty_engine.add_doc(doc["content"], doc["metadata"])

    benchmark(add_doc)
