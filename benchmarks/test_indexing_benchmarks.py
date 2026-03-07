"""Indexing throughput benchmarks for VectorForge.

Two tests covering the core insertion performance:
- Single document into an empty index
- Batch of 100 documents into a fresh index

Both tests start from an empty engine, so index size is not a variable here. We are measuring raw insertion latency, not the effect of index growth.
"""

from typing import Callable

from benchmarks.conftest import generate_document, generate_documents
from vectorforge.vector_engine import VectorEngine


def test_add_single_doc(benchmark, make_ephemeral_engine: Callable[[], VectorEngine]):
    """Benchmark adding a single document to an empty index."""
    doc = generate_document(0)

    def add_doc():
        engine = make_ephemeral_engine()
        engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(add_doc, iterations=1, rounds=5)


def test_batch_insert_100_docs(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark batch insertion of 100 documents into a fresh index."""
    docs = generate_documents(100)

    def batch_insert():
        engine = make_ephemeral_engine()
        for doc in docs:
            engine.add_doc(doc["content"], doc["metadata"])

    benchmark.pedantic(batch_insert, iterations=1, rounds=3)
