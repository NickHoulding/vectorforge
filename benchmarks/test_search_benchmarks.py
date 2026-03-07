"""Search latency benchmarks for VectorForge.

Three tests covering how query latency scales with index size:
- Small  (100 docs)
- Medium (1,000 docs)
- Large  (10,000 docs) — @pytest.mark.slow
"""

import pytest

from vectorforge.vector_engine import VectorEngine

_QUERY = "machine learning vector search"


def test_search_latency_small(benchmark, engine_small: VectorEngine):
    """Benchmark search latency on a small index (100 docs)."""
    benchmark(engine_small.search, query=_QUERY, top_k=10)


def test_search_latency_medium(benchmark, engine_medium: VectorEngine):
    """Benchmark search latency on a medium index (1,000 docs)."""
    benchmark(engine_medium.search, query=_QUERY, top_k=10)


@pytest.mark.slow
def test_search_latency_large(benchmark, engine_large: VectorEngine):
    """Benchmark search latency on a large index (10,000 docs)."""
    benchmark(engine_large.search, query=_QUERY, top_k=10)
