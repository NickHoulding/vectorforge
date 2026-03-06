"""Metrics system benchmarks for the ChromaDB-integrated VectorForge.

The metrics subsystem is a new addition vs. the pre-ChromaDB version.  It adds
synchronous SQLite writes on *every* ``add_doc`` and every ``search`` call.  At
low throughput these writes are negligible, but they can become a bottleneck at
high ingestion or query rates.

This file benchmarks:

1. **``_update_query_metrics`` overhead** — the SQLite write that happens after
   every search.  Isolated by calling it directly rather than through ``search``.

2. **``_metrics_store.save`` overhead** — the underlying MetricsStore.save() call
   that both ``add_doc`` and ``search`` funnel into.

3. **``get_metrics()`` latency** — computes ``np.percentile`` over the rolling
   ``query_times`` deque; interesting to see how it scales with deque fill level.

4. **``_get_chromadb_disk_size()`` overhead** — every ``/metrics`` HTTP request
   triggers an ``os.walk`` over the ChromaDB data directory.  Measured against
   indexes of different sizes to understand whether it's a bottleneck at scale.

5. **End-to-end search-with-metrics** — full ``engine.search()`` timing compared
   against a hypothetical search without the SQLite flush (measured by timing the
   flush separately and comparing).
"""

import timeit
from datetime import datetime, timezone
from typing import Callable

import chromadb
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from benchmarks.conftest import SCALES, _bulk_populate, generate_documents
from vectorforge.config import VFGConfig
from vectorforge.metrics_store import MetricsStore
from vectorforge.vector_engine import VectorEngine

# ============================================================================
# MetricsStore.save() — raw SQLite write latency
# ============================================================================


def test_metrics_store_save_latency(benchmark, tmp_path):
    """Benchmark the raw SQLite write performed by MetricsStore.save().

    This is the lowest-level measurement of the persistence overhead that is
    incurred on every ``add_doc`` and every ``search`` call.

    Args:
        benchmark: pytest-benchmark fixture.
        tmp_path: pytest-provided temporary directory.
    """
    db_path = str(tmp_path / "metrics.db")
    store = MetricsStore(db_path)
    store.insert("test_collection", datetime.now(timezone.utc).isoformat())

    payload = {
        "total_queries": 1,
        "total_query_time_ms": 12.5,
        "last_query_at": datetime.now(timezone.utc).isoformat(),
    }

    benchmark(store.save, collection_name="test_collection", data=payload)


def test_metrics_store_increment_latency(benchmark, tmp_path):
    """Benchmark MetricsStore.increment() — atomic counter increment in SQLite.

    Args:
        benchmark: pytest-benchmark fixture.
        tmp_path: pytest-provided temporary directory.
    """
    db_path = str(tmp_path / "metrics.db")
    store = MetricsStore(db_path)
    store.insert("test_collection", datetime.now(timezone.utc).isoformat())

    benchmark(
        store.increment,
        collection_name="test_collection",
        field="total_queries",
        delta=1,
    )


def test_metrics_store_load_latency(benchmark, tmp_path):
    """Benchmark MetricsStore.load() — SQLite read latency.

    Args:
        benchmark: pytest-benchmark fixture.
        tmp_path: pytest-provided temporary directory.
    """
    db_path = str(tmp_path / "metrics.db")
    store = MetricsStore(db_path)
    store.insert("test_collection", datetime.now(timezone.utc).isoformat())

    benchmark(store.load, collection_name="test_collection")


# ============================================================================
# _update_query_metrics — overhead per search call
# ============================================================================


def test_update_query_metrics_overhead(
    benchmark, make_ephemeral_engine: Callable[[], VectorEngine]
):
    """Benchmark the metrics flush that occurs at the end of every search.

    Calls ``_update_query_metrics()`` directly to isolate it from embedding
    and HNSW query time.

    Args:
        benchmark: pytest-benchmark fixture.
        make_ephemeral_engine: Factory for in-memory engines.
    """
    engine = make_ephemeral_engine()
    elapsed_ms = 42.0

    benchmark(engine._update_query_metrics, elapsed_ms=elapsed_ms)


# ============================================================================
# get_metrics() — np.percentile over rolling query_times deque
# ============================================================================


@pytest.mark.parametrize("history_size", [10, 100, 500, 1000])
def test_get_metrics_latency(
    benchmark,
    history_size: int,
    make_ephemeral_engine: Callable[[], VectorEngine],
):
    """Benchmark get_metrics() at different query_times deque fill levels.

    The function calls ``np.percentile`` on the rolling deque, which involves
    converting it to a list and sorting.  This test shows whether the overhead
    grows meaningfully with the deque size.

    Args:
        benchmark: pytest-benchmark fixture.
        history_size: Number of entries pre-loaded into query_times deque.
        make_ephemeral_engine: Factory for in-memory engines.
    """
    engine = make_ephemeral_engine()

    # Pre-fill the rolling window with synthetic query times.
    for i in range(history_size):
        engine.metrics.query_times.append(float(i % 200))

    benchmark(engine.get_metrics)


# ============================================================================
# _get_chromadb_disk_size() — os.walk overhead at different index sizes
# ============================================================================


@pytest.mark.parametrize("scale", ["tiny", "small", "medium"])
def test_disk_size_scan_overhead(benchmark, scale: str, shared_model, tmp_path):
    """Benchmark the os.walk scan triggered by every /metrics request.

    ``_get_chromadb_disk_size()`` is called on every GET /metrics request.  This
    test measures whether it becomes a bottleneck as the number of files in the
    ChromaDB data directory grows with index size.

    Args:
        benchmark: pytest-benchmark fixture.
        scale: Index size key ('tiny', 'small', 'medium').
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
    _bulk_populate(engine, generate_documents(SCALES[scale]))

    benchmark(engine._get_chromadb_disk_size)

    SharedSystemClient.clear_system_cache()


# ============================================================================
# End-to-end: search with metrics vs. isolated search overhead
# ============================================================================


def test_search_metrics_flush_proportion(
    benchmark,
    engine_medium: VectorEngine,
    make_ephemeral_engine: Callable[[], VectorEngine],
):
    """Compare full search timing against the isolated metrics flush overhead.

    Runs the full ``search()`` under the benchmark timer and also separately
    times ``_update_query_metrics()`` alone, printing both so we can see what
    fraction of total search time is consumed by the SQLite flush.

    Args:
        benchmark: pytest-benchmark fixture.
        engine_medium: Pre-populated 1,000-doc VectorEngine (EphemeralClient).
        make_ephemeral_engine: Factory for in-memory engines (for flush isolation).
    """
    benchmark(engine_medium.search, query="performance test", top_k=10)
    flush_engine = make_ephemeral_engine()

    flush_time_us = (
        timeit.timeit(
            lambda: flush_engine._update_query_metrics(10.0),
            number=100,
        )
        / 100
        * 1e6
    )
    print(f"\n[info] _update_query_metrics isolated: {flush_time_us:.1f} µs/call")
