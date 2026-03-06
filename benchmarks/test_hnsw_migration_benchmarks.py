"""HNSW parameter migration benchmarks for VectorForge.

The ``update_hnsw_config()`` method performs a blue-green collection migration:
it creates a temporary collection with new HNSW settings, batch-copies all
documents from the old collection, atomically swaps the live collection
reference, then deletes the old collection.

This operation is new in the ChromaDB-integrated version — there is no
equivalent in the pre-ChromaDB suite.  It is a user-facing operation invoked via
``PUT /collections/{name}/hnsw-config`` and its latency scales with document
count.

Benchmarks in this file cover:

1. **Migration time at small / medium / large scale** — how long does the full
   blue-green swap take as the corpus grows?

2. **Parameter variation** — does migrating to a higher ``hnsw:M`` (more
   neighbours, better recall, slower build) meaningfully change migration time?

3. **Post-migration search correctness** — not a timing test but asserts that
   doc count and search results are preserved after migration.

Notes:
- All tests use PersistentClient because EphemeralClient's collection UUIDs do
  not persist across the blue-green temp-collection create/delete cycle in the
  same way; PersistentClient is the realistic deployment context anyway.
- Large and xlarge tests are marked ``@slow``.
- Iterations are set to 1 and rounds to 3 because each migration is destructive
  (irreversible in-place); we cannot re-run it on the same engine without
  re-populating.
"""

import tempfile
import timeit
from typing import Callable

import chromadb
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from benchmarks.conftest import SCALES, _bulk_populate, generate_documents
from vectorforge.config import VFGConfig
from vectorforge.vector_engine import VectorEngine

# Default HNSW config used as the migration target in most tests.
_DEFAULT_MIGRATION_TARGET = {
    "space": "cosine",
    "ef_construction": 100,
    "ef_search": 100,
    "max_neighbors": 16,
    "resize_factor": 1.2,
    "sync_threshold": 1000,
}


# ============================================================================
# Helper: build a fresh persistent engine for each benchmark round
# ============================================================================


def _make_persistent_engine_factory(
    shared_model, doc_count: int
) -> Callable[[], tuple[VectorEngine, tempfile.TemporaryDirectory]]:
    """Return a factory that creates a freshly populated PersistentClient engine.

    Each call creates a new TemporaryDirectory, a new PersistentClient, and
    indexes ``doc_count`` documents.  The caller owns the TemporaryDirectory and
    is responsible for cleanup.

    Args:
        shared_model: Pre-loaded SentenceTransformer model.
        doc_count: Number of documents to pre-index.

    Returns:
        Zero-argument callable that returns (VectorEngine, TemporaryDirectory).
    """
    docs = generate_documents(doc_count)

    def factory():
        tmpdir = tempfile.TemporaryDirectory()
        client = chromadb.PersistentClient(path=tmpdir.name)
        collection = client.get_or_create_collection(
            name=VFGConfig.DEFAULT_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        engine = VectorEngine(
            collection=collection,
            model=shared_model,
            chroma_client=client,
        )
        _bulk_populate(engine, docs)
        return engine, tmpdir

    return factory


# ============================================================================
# Migration Time - Varying Scale
# ============================================================================


def test_hnsw_migration_small(benchmark, shared_model):
    """Benchmark HNSW migration on small index (100 docs).

    Each benchmark round creates a fresh populated engine so the migration
    always starts from the same baseline state.

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
    """
    factory = _make_persistent_engine_factory(shared_model, SCALES["small"])
    tmpdirs = []

    def migrate():
        engine, tmpdir = factory()
        tmpdirs.append(tmpdir)
        result = engine.update_hnsw_config(_DEFAULT_MIGRATION_TARGET)
        return result

    benchmark.pedantic(migrate, iterations=1, rounds=5)

    SharedSystemClient.clear_system_cache()
    for d in tmpdirs:
        d.cleanup()


def test_hnsw_migration_medium(benchmark, shared_model):
    """Benchmark HNSW migration on medium index (1,000 docs).

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
    """
    factory = _make_persistent_engine_factory(shared_model, SCALES["medium"])
    tmpdirs = []

    def migrate():
        engine, tmpdir = factory()
        tmpdirs.append(tmpdir)
        result = engine.update_hnsw_config(_DEFAULT_MIGRATION_TARGET)
        return result

    benchmark.pedantic(migrate, iterations=1, rounds=3)

    SharedSystemClient.clear_system_cache()
    for d in tmpdirs:
        d.cleanup()


@pytest.mark.slow
def test_hnsw_migration_large(benchmark, shared_model):
    """Benchmark HNSW migration on large index (10,000 docs).

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
    """
    factory = _make_persistent_engine_factory(shared_model, SCALES["large"])
    tmpdirs = []

    def migrate():
        engine, tmpdir = factory()
        tmpdirs.append(tmpdir)
        result = engine.update_hnsw_config(_DEFAULT_MIGRATION_TARGET)
        return result

    benchmark.pedantic(migrate, iterations=1, rounds=2)

    SharedSystemClient.clear_system_cache()
    for d in tmpdirs:
        d.cleanup()


# ============================================================================
# Migration with Parameter Variation
# ============================================================================


@pytest.mark.parametrize(
    "max_neighbors",
    [8, 16, 32, 64],
    ids=["M8", "M16", "M32", "M64"],
)
def test_hnsw_migration_varying_m(benchmark, shared_model, max_neighbors: int):
    """Benchmark migration time when changing hnsw:M (max_neighbors).

    Higher M means a denser graph, better recall, but longer build time.
    This shows whether M has a significant effect on migration duration.

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
        max_neighbors: HNSW M parameter to migrate to.
    """
    factory = _make_persistent_engine_factory(shared_model, SCALES["small"])
    config = {**_DEFAULT_MIGRATION_TARGET, "max_neighbors": max_neighbors}
    tmpdirs = []

    def migrate():
        engine, tmpdir = factory()
        tmpdirs.append(tmpdir)
        return engine.update_hnsw_config(config)

    benchmark.pedantic(migrate, iterations=1, rounds=3)

    SharedSystemClient.clear_system_cache()
    for d in tmpdirs:
        d.cleanup()


@pytest.mark.parametrize(
    "ef_construction",
    [50, 100, 200],
    ids=["ef50", "ef100", "ef200"],
)
def test_hnsw_migration_varying_ef_construction(
    benchmark, shared_model, ef_construction: int
):
    """Benchmark migration time when changing ef_construction.

    Higher ef_construction improves index quality at build time but increases
    migration cost.

    Args:
        benchmark: pytest-benchmark fixture.
        shared_model: Session-scoped SentenceTransformer.
        ef_construction: HNSW ef_construction value to migrate to.
    """
    factory = _make_persistent_engine_factory(shared_model, SCALES["small"])
    config = {**_DEFAULT_MIGRATION_TARGET, "ef_construction": ef_construction}
    tmpdirs = []

    def migrate():
        engine, tmpdir = factory()
        tmpdirs.append(tmpdir)
        return engine.update_hnsw_config(config)

    benchmark.pedantic(migrate, iterations=1, rounds=3)

    SharedSystemClient.clear_system_cache()
    for d in tmpdirs:
        d.cleanup()


# ============================================================================
# Post-Migration Correctness (assertion test — no benchmark timer)
# ============================================================================


def test_migration_preserves_doc_count(shared_model, tmp_path):
    """Assert that migration preserves all documents.

    Not a timed benchmark — verifies correctness of the blue-green migration
    by checking that doc count and search results are intact afterwards.

    Args:
        shared_model: Session-scoped SentenceTransformer.
        tmp_path: pytest-provided temporary directory.
    """
    chroma_path = str(tmp_path / "chroma")
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

    doc_count = SCALES["small"]
    _bulk_populate(engine, generate_documents(doc_count))

    assert engine.collection.count() == doc_count

    result = engine.update_hnsw_config({"ef_search": 150, "max_neighbors": 32})

    assert result["status"] == "success"
    assert result["migration"]["documents_migrated"] == doc_count
    assert engine.collection.count() == doc_count

    search_results = engine.search("test query", top_k=5)
    assert len(search_results) > 0

    SharedSystemClient.clear_system_cache()


def test_migration_search_after_migration(shared_model, tmp_path):
    """Benchmark search latency immediately after HNSW migration.

    Useful for comparing pre- vs. post-migration query latency to verify that
    the new HNSW parameters take effect (e.g. higher ef_search → slower but
    more accurate queries).

    Args:
        shared_model: Session-scoped SentenceTransformer.
        tmp_path: pytest-provided temporary directory.
    """
    chroma_path = str(tmp_path / "chroma")
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

    _bulk_populate(engine, generate_documents(SCALES["small"]))

    pre_time = (
        timeit.timeit(
            lambda: engine.search("test query", top_k=10),
            number=10,
        )
        / 10
    )

    engine.update_hnsw_config({"ef_search": 200, "max_neighbors": 32})

    post_time = (
        timeit.timeit(
            lambda: engine.search("test query", top_k=10),
            number=10,
        )
        / 10
    )

    print(
        f"\n[migration search] pre={pre_time * 1000:.1f} ms, "
        f"post={post_time * 1000:.1f} ms"
    )

    assert engine.collection.count() == SCALES["small"]
    SharedSystemClient.clear_system_cache()
