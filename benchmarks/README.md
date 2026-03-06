# VectorForge Benchmarking Suite

Comprehensive performance benchmarks for VectorForge's ChromaDB-integrated Python implementation.

## Overview

The suite tests seven distinct performance dimensions across 112 non-slow tests and 15 slow tests:

| File | Tests | What it measures |
|------|-------|-----------------|
| `test_search_benchmarks.py` | 22 | Query latency, top-k impact, query complexity, throughput, filtered search |
| `test_indexing_benchmarks.py` | 18 | Single-doc insertion cost, batch throughput, document size impact |
| `test_file_processing_benchmarks.py` | 23 | PDF extraction, text chunking parameters, end-to-end file pipeline |
| `test_scaling_benchmarks.py` | 17 | Latency curves, memory growth, realistic multi-step workflows |
| `test_persistence_benchmarks.py` | 8 | Disk size growth, cold-start load time, persistent write overhead |
| `test_metrics_benchmarks.py` | 9 | SQLite write latency, `get_metrics()` overhead, disk scan cost |
| `test_hnsw_migration_benchmarks.py` | 9 | Blue-green HNSW migration time, parameter variation |

## Architecture

### ChromaDB Backend

All benchmarks construct `VectorEngine(collection, model, chroma_client)` directly, bypassing
`CollectionManager` (which reads from `VFGConfig` and is not injectable). Two client types are
used deliberately:

- **EphemeralClient** — in-memory, no disk I/O, used for all unit/algorithm benchmarks where
  isolation and speed matter
- **PersistentClient** — on-disk (temp directories), used for persistence, cold-start, metrics
  disk-scan, and HNSW migration benchmarks where disk behavior is the thing being measured

### Fast Fixture Population: `_bulk_populate`

The helper `_bulk_populate(engine, docs, batch_size=500)` is used in all pre-populated fixtures
instead of calling `engine.add_doc()` in a loop. It batch-encodes all documents in a single
`model.encode()` call, normalizes vectors manually, then calls `engine.collection.add()` in
batches of 500.

**Rationale:** `add_doc` costs ~46ms/call (embedding + ChromaDB insert + SQLite write). At 1,000
docs that is 46 seconds of setup time. `_bulk_populate` reduces this to ~0.7 seconds (67× faster)
by eliminating per-document SQLite writes and batching the embedding pass.

### Shared Model

`shared_model` is a session-scoped fixture that loads `SentenceTransformer` exactly once per
pytest run (~5 seconds). All engine fixtures receive it, so model-load time is never included in
benchmark measurements.

### Avoiding Accumulation Bugs

Benchmarks that call `add_doc` on a shared engine across rounds would grow the index unboundedly,
distorting results. Every test that measures insertion cost uses either:

- `make_ephemeral_engine()` — a function-scoped factory called inside the timed function to
  produce a fresh engine per round
- `benchmark.pedantic(iterations=1, rounds=N)` — so the setup callable runs before each round

## Data Scales

```
SCALES = {
    "tiny":   10 documents,
    "small":  100 documents,
    "medium": 1,000 documents,
    "large":  10,000 documents,   # @pytest.mark.slow
    "xlarge": 50,000 documents,   # @pytest.mark.slow
}
```

The `large` and `xlarge` fixtures exist in `conftest.py` but are only used by `@pytest.mark.slow`
tests. The default run (`-m "not slow"`) never touches them.

## What Each Suite Benchmarks

### Search (`test_search_benchmarks.py`)

- **Latency vs. index size** — five scales (tiny → xlarge) with `top_k=10`
- **Top-k sweep** — `top_k` in `[1, 5, 10, 50, 100]` on a medium index
- **Query complexity** — single word vs. 3-word phrase vs. full sentence
- **Batch queries** — 10 and 100 sequential queries in a single round
- **Warm vs. cold** — pre-executes 5 queries to warm HNSW structures before timing
- **Throughput (QPS)** — `pedantic(iterations=100, rounds=5)` for a stable ops/sec figure
- **Post-deletion search** — latency after removing 25% of documents
- **Filtered search** — `where` clause on metadata fields at small/medium/large scales

The filtered search tests build their own collections inline (so metadata field values are
controlled) rather than relying on the shared scale fixtures.

### Indexing (`test_indexing_benchmarks.py`)

- **Marginal insertion cost** — adds one document to indexes of 0, 100, 1,000, and 10,000 docs;
  each round uses a fresh engine to prevent accumulation
- **Batch throughput** — inserts 10, 100, 1,000, and 10,000 documents into a fresh engine per
  round (slow at 10k)
- **Document size** — short (~75 chars), medium (~750 chars), and long (~3,500 chars) documents
- **File-simulated chunks** — 10, 50, and 100 chunks as if coming from a real file upload
- **Interleaved add+search** — 20 inserts with a search every 5th document
- **Metadata overhead** — minimal (`{}`) vs. rich (8-field) metadata dicts

All tests use `benchmark.pedantic(iterations=1, rounds=N)` to avoid round-to-round state
accumulation. Round counts decrease with cost: 5 for single-doc tests, 2–3 for batches.

### File Processing (`test_file_processing_benchmarks.py`)

Pure-function tests (no VectorEngine dependency):
- **Chunk size sweep** — 200, 500, 1000, 2000-char chunk sizes on medium text
- **Overlap sweep** — 0, 25, 50, 100, 200-char overlaps
- **Text size sweep** — small (20–50 sentences), medium (100–300), large (400–800)
- **Edge cases** — text equal to chunk size, text shorter than chunk size
- **Memory efficiency** — 100,000-sentence text to probe allocations

End-to-end tests (with VectorEngine):
- **Text file pipeline** — `chunk_text` → `engine.add_doc` on medium text
- **PDF pipeline** — `extract_pdf` (PyMuPDF synthetic PDF) → `chunk_text` → `engine.add_doc`

PDF tests generate synthetic in-memory PDFs using `fitz` — no real PDF files are required.

### Scaling (`test_scaling_benchmarks.py`)

- **Latency curve** — query latency at tiny/small/medium (and large with `@slow`)
- **Indexing speed curve** — marginal `add_doc` cost at increasing index sizes
- **Memory growth** — RSS delta (via `psutil`) at 100 and 500 docs (1k–10k with `@slow`)
- **Batch insertion curve** — `_bulk_populate` throughput at 10, 50, 100, 500 docs
- **Top-k scaling** — search at `top_k` in `[1, 5, 10, 50, 100, 500]`
- **Realistic workflows** — file upload simulation, CRUD cycle, concurrent query simulation,
  long-running session (add/search/delete loops)

### Persistence (`test_persistence_benchmarks.py`)

Replaces the old save/load-centric suite (removed: `save()`, `load()`, compression tests).
ChromaDB writes to disk automatically; there is no explicit `save()` call.

- **Disk size growth** — asserts `_get_chromadb_disk_size()` is positive and prints bytes/doc at
  tiny/small/medium/large scales
- **Cold-start latency** — opens a PersistentClient against a pre-populated on-disk index and
  runs the first query; `SharedSystemClient.clear_system_cache()` is called inside each round to
  simulate a true cold open
- **Persistent indexing throughput** — 100-doc `add_doc` loop with a PersistentClient, for direct
  comparison with the EphemeralClient baseline in `test_indexing_benchmarks.py`
- **Checkpoint simulation** — cycles of 10 `add_doc` + 1 `search` on a 100-doc persistent base
  index, measuring incremental write overhead

### Metrics (`test_metrics_benchmarks.py`)

Suite covering the synchronous SQLite subsystem introduced in the ChromaDB refactor. Every
`add_doc` and `search` call triggers a SQLite write synchronously on the hot path.

- **`MetricsStore.save()` latency** — raw SQLite write cost
- **`MetricsStore.increment()` latency** — atomic counter update cost
- **`MetricsStore.load()` latency** — SQLite read cost
- **`_update_query_metrics()` overhead** — isolated from embedding/HNSW time
- **`get_metrics()` scaling** — latency as `query_times` deque grows (10 → 100 → 500 → 1,000
  entries); tests `np.percentile` scaling behavior
- **`_get_chromadb_disk_size()` scan** — `os.walk` cost at tiny/small/medium scales using a
  PersistentClient; called on every `/metrics` HTTP request

### HNSW Migration (`test_hnsw_migration_benchmarks.py`)

Suite covering `update_hnsw_config()`, which performs a blue-green migration: create a temp
collection with new HNSW parameters → batch-copy all documents → atomic name swap → delete old
collection. This is the most disruptive operation in the system.

All tests use PersistentClient (EphemeralClient UUIDs do not survive the temp-collection cycle).

- **Migration time vs. scale** — 100, 1,000, and 10,000 docs
- **M parameter sweep** — `max_neighbors` in `[8, 16, 32, 64]` on 100 docs
- **ef_construction sweep** — `[50, 100, 200]` on 100 docs
- **Correctness assertion** — verifies `result["status"] == "success"`, doc count preserved, and
  search still works post-migration
- **Pre/post search latency** — `timeit`-based comparison printed to stdout (informational)

Each round gets a fresh engine (factory pattern with manual `TemporaryDirectory` cleanup)
because each migration is destructive — it renames and deletes collections.

## Running Benchmarks

### Prerequisites

```bash
uv sync --group dev
```

### Fastest Feedback (smoke test — no benchmark timing)

```bash
pytest benchmarks/ -m "not slow" --benchmark-disable -q
```

This runs 112 tests in ~8 minutes. Use this to verify nothing is broken.

### Standard Benchmark Run

```bash
# All non-slow benchmarks with timing
pytest benchmarks/ -m "not slow" --benchmark-only

# Single suite
pytest benchmarks/test_search_benchmarks.py --benchmark-only
```

### Slow Tests (large-scale)

```bash
# All tests including slow
pytest benchmarks/ --benchmark-only

# Just slow tests
pytest benchmarks/ -m "slow" --benchmark-only
```

### Using `run_benchmarks.py`

```bash
# All non-slow benchmarks
python benchmarks/run_benchmarks.py

# Specific suite(s)
python benchmarks/run_benchmarks.py --search
python benchmarks/run_benchmarks.py --metrics --hnsw

# Available flags:
#   --search, --indexing, --file-processing, --persistence,
#   --scaling, --metrics, --hnsw
# 
#   --all           Include slow tests
#   --save NAME     Save results as baseline
#   --compare NAME  Compare against saved baseline
#   --json FILE     Write results to JSON
#   --histogram     Generate histogram (requires pygal)
#   --rounds N      Override minimum rounds
#   -k EXPR         Filter tests by expression
#   -v / --verbose  Verbose output
```

### Saving and Comparing Results

```bash
# Save a baseline
pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-save=baseline

# Compare after changes
pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-compare=baseline

# Fail CI if mean regresses more than 20%
pytest benchmarks/ -m "not slow" --benchmark-only \
  --benchmark-compare=baseline \
  --benchmark-compare-fail=mean:20%
```

## Markers

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.slow` | Skipped by `-m "not slow"`; large/xlarge scale tests |
| `@pytest.mark.scale(size="...")` | Parametrize by scale name |

## Interpreting Results

```
Name (time in ms)              Min      Max     Mean  StdDev  Median     IQR  Outliers  OPS
test_search_latency_small    10.23   15.67    11.45    1.23   11.34    0.89      5;2   87.3
```

- **Min/Max** — fastest and slowest rounds
- **Mean/Median** — focus on median for skewed distributions
- **StdDev/IQR** — consistency; lower is better
- **OPS** — operations per second (inverse of mean)
- **Outliers** — rounds outside 1.5×IQR; high counts suggest system noise

## Tips

1. **Model load time is not measured.** The `shared_model` session fixture loads
   `SentenceTransformer` once before any benchmark runs (~5 seconds).

2. **Run on a quiet system.** Close background processes for consistent results, especially for
   memory tests.

3. **Slow tests are gated intentionally.** `large` (10k docs) and `xlarge` (50k docs) fixtures
   take minutes to populate even with `_bulk_populate`. Never add them to the default suite.

4. **PersistentClient teardowns call `SharedSystemClient.clear_system_cache()`.** Without this,
   ChromaDB leaks file descriptors across tests when many persistent engines are created.

5. **Insertion benchmarks never reuse engines across rounds.** Any test that measures `add_doc`
   uses `make_ephemeral_engine()` inside the timed function or `pedantic(iterations=1)` to get a
   fresh index per round.

## Structure

```
benchmarks/
├── __init__.py
├── conftest.py                          # Fixtures, _bulk_populate, SCALES constant
├── run_benchmarks.py                    # CLI runner script
├── test_search_benchmarks.py
├── test_indexing_benchmarks.py
├── test_file_processing_benchmarks.py
├── test_scaling_benchmarks.py
├── test_persistence_benchmarks.py       # Rewritten: disk/cold-start/throughput
├── test_metrics_benchmarks.py           # SQLite and metrics subsystem
├── test_hnsw_migration_benchmarks.py    # Blue-green HNSW migration
├── datasets/                            # Sample data files
├── results/                             # Benchmark results (gitignored)
├── QUICKSTART.md
└── README.md
```

## Resources

- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
- [ChromaDB docs](https://docs.trychroma.com/)
- [VectorForge main docs](../README.md)
