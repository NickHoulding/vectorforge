"""Persistence benchmarks for the ChromaDB-integrated VectorForge.

ChromaDB writes all data to disk automatically on every ``add_doc`` — there is no
separate ``save()`` call.  This file replaces the old save/load-centric suite with
three groups that are meaningful for the current architecture:

1. **Disk size growth** — how does on-disk storage grow as we add documents at
   different scales?  Uses ``_get_chromadb_disk_size()`` after indexing each batch.
   (These are measurement/assertion tests, not timed benchmarks.)

2. **Cold-start load time** — how long does it take to open a PersistentClient
   against a pre-populated on-disk collection and run the first query?  This models
   the latency users would experience after a server restart.

3. **Checkpoint write simulation** — repeated add_doc + query cycles on a
   PersistentClient to understand how disk-write pressure from ChromaDB's WAL and
   sync_threshold affects throughput compared with EphemeralClient.

Metrics tracked:
- Cold-start latency (PersistentClient open + first query)
- Throughput delta: persistent vs. in-memory indexing
- Disk bytes per document at different scales
"""

import os
import tempfile
from typing import Callable

import chromadb
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from benchmarks.conftest import (
    SCALES,
    _bulk_populate,
    generate_document,
    generate_documents,
)
from vectorforge.config import VFGConfig
from vectorforge.vector_engine import VectorEngine

# ============================================================================
# Helper: build a pre-populated PersistentClient engine in a given tmpdir
# ============================================================================


def _build_persistent_engine(
    chroma_path: str,
    shared_model,
    doc_count: int,
) -> VectorEngine:
    """Create and populate a PersistentClient-backed VectorEngine.

    Uses bulk insertion (batch encode + single collection.add) to avoid the
    per-document SQLite write overhead of ``engine.add_doc``.  This keeps
    fixture setup fast so the timed sections only measure what they intend.

    Args:
        chroma_path: Directory for ChromaDB to persist data.
        shared_model: Pre-loaded SentenceTransformer model.
        doc_count: Number of synthetic documents to index.

    Returns:
        Populated VectorEngine instance.
    """
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=VFGConfig.DEFAULT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    engine = VectorEngine(
        collection=collection,
        model=shared_model,
        chroma_client=client,
    )
    _bulk_populate(engine, generate_documents(doc_count))
    return engine


# ============================================================================
# Disk Size Growth (measurement tests — no benchmark timer)
# ============================================================================


@pytest.mark.parametrize(
    "scale",
    ["tiny", "small", "medium"],
)
def test_disk_size_growth(scale: str, shared_model, tmp_path):
    """Measure on-disk ChromaDB storage after indexing at different scales.

    This is a measurement/assertion test, not a timed benchmark.  It verifies
    that ``_get_chromadb_disk_size()`` returns a positive value and prints the
    bytes/document ratio so it shows up in verbose test output.

    Args:
        scale: One of 'tiny', 'small', 'medium'.
        shared_model: Session-scoped SentenceTransformer.
        tmp_path: pytest-provided temporary directory.
    """
    chroma_path = str(tmp_path / "chroma")
    doc_count = SCALES[scale]
    engine = _build_persistent_engine(chroma_path, shared_model, doc_count)

    disk_bytes, disk_mb = engine._get_chromadb_disk_size()

    assert disk_bytes > 0, "Expected non-zero disk usage after indexing"
    assert disk_mb > 0.0

    bytes_per_doc = disk_bytes / doc_count
    print(
        f"\n[{scale}] {doc_count} docs → {disk_mb:.2f} MB "
        f"({bytes_per_doc:.0f} bytes/doc)"
    )

    SharedSystemClient.clear_system_cache()


@pytest.mark.slow
def test_disk_size_growth_large(shared_model, tmp_path):
    """Measure on-disk storage for large index (10,000 docs).

    Args:
        shared_model: Session-scoped SentenceTransformer.
        tmp_path: pytest-provided temporary directory.
    """
    chroma_path = str(tmp_path / "chroma")
    doc_count = SCALES["large"]
    engine = _build_persistent_engine(chroma_path, shared_model, doc_count)

    disk_bytes, disk_mb = engine._get_chromadb_disk_size()

    assert disk_bytes > 0
    bytes_per_doc = disk_bytes / doc_count
    print(
        f"\n[large] {doc_count} docs → {disk_mb:.2f} MB ({bytes_per_doc:.0f} bytes/doc)"
    )

    SharedSystemClient.clear_system_cache()


# ============================================================================
# Cold-Start Load Time
# ============================================================================


def test_cold_start_small(benchmark, shared_model, tmp_path):
    """Benchmark: open a pre-populated PersistentClient + run first query.

    Simulates a server restart against a small (100-doc) collection.

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
        tmp_path: pytest-provided temporary directory.
    """
    chroma_path = str(tmp_path / "chroma")
    _build_persistent_engine(chroma_path, shared_model, SCALES["small"])
    SharedSystemClient.clear_system_cache()

    def cold_start_and_query():
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name=VFGConfig.DEFAULT_COLLECTION_NAME)
        engine = VectorEngine(
            collection=collection,
            model=shared_model,
            chroma_client=client,
        )
        results = engine.search("test query", top_k=10)
        SharedSystemClient.clear_system_cache()
        return results

    benchmark.pedantic(cold_start_and_query, iterations=1, rounds=5)


def test_cold_start_medium(benchmark, shared_model, tmp_path):
    """Benchmark: open a pre-populated PersistentClient + run first query.

    Simulates a server restart against a medium (1,000-doc) collection.

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
        tmp_path: pytest-provided temporary directory.
    """
    chroma_path = str(tmp_path / "chroma")
    _build_persistent_engine(chroma_path, shared_model, SCALES["medium"])
    SharedSystemClient.clear_system_cache()

    def cold_start_and_query():
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name=VFGConfig.DEFAULT_COLLECTION_NAME)
        engine = VectorEngine(
            collection=collection,
            model=shared_model,
            chroma_client=client,
        )
        results = engine.search("test query", top_k=10)
        SharedSystemClient.clear_system_cache()
        return results

    benchmark.pedantic(cold_start_and_query, iterations=1, rounds=5)


@pytest.mark.slow
def test_cold_start_large(benchmark, shared_model, tmp_path):
    """Benchmark: open a pre-populated PersistentClient + run first query.

    Simulates a server restart against a large (10,000-doc) collection.

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
        tmp_path: pytest-provided temporary directory.
    """
    chroma_path = str(tmp_path / "chroma")
    _build_persistent_engine(chroma_path, shared_model, SCALES["large"])
    SharedSystemClient.clear_system_cache()

    def cold_start_and_query():
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(name=VFGConfig.DEFAULT_COLLECTION_NAME)
        engine = VectorEngine(
            collection=collection,
            model=shared_model,
            chroma_client=client,
        )
        results = engine.search("test query", top_k=10)
        SharedSystemClient.clear_system_cache()
        return results

    benchmark.pedantic(cold_start_and_query, iterations=1, rounds=3)


# ============================================================================
# Persistent vs. Ephemeral Indexing Throughput
# ============================================================================


def test_persistent_indexing_throughput(benchmark, shared_model):
    """Measure indexing throughput with a PersistentClient (includes disk writes).

    Compares against ``test_indexing_throughput_docs_per_second`` in
    ``test_indexing_benchmarks.py`` which uses EphemeralClient.

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
    """
    docs = generate_documents(100)

    def index_batch():
        with tempfile.TemporaryDirectory() as tmpdir:
            client = chromadb.PersistentClient(path=tmpdir)
            collection = client.get_or_create_collection(
                name=VFGConfig.DEFAULT_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            engine = VectorEngine(
                collection=collection,
                model=shared_model,
                chroma_client=client,
            )
            for doc in docs:
                engine.add_doc(doc["content"], doc["metadata"])
            SharedSystemClient.clear_system_cache()

    benchmark.pedantic(index_batch, iterations=1, rounds=3)


# ============================================================================
# Checkpoint Simulation (repeated writes on persistent store)
# ============================================================================


def test_checkpoint_simulation(benchmark, shared_model, tmp_path):
    """Simulate periodic add_doc + query cycles on a persistent index.

    Models a write-heavy session where documents are added incrementally and
    queries are interleaved, exercising ChromaDB's WAL / sync_threshold path.

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
        tmp_path: pytest-provided temporary directory.
    """
    chroma_path = str(tmp_path / "chroma")
    engine = _build_persistent_engine(chroma_path, shared_model, SCALES["small"])

    def checkpoint_cycle():
        for i in range(10):
            doc = generate_document(9000 + i)
            engine.add_doc(doc["content"], doc["metadata"])
        engine.search("checkpoint query", top_k=5)

    benchmark.pedantic(checkpoint_cycle, iterations=5, rounds=3)
