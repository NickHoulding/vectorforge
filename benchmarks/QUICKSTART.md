# VectorForge Benchmarking Quick Start

## Prerequisites

```bash
uv sync --group dev
```

## The Fastest Sanity Check

Run all 112 non-slow tests without benchmark timing (just verify nothing is broken):

```bash
pytest benchmarks/ -m "not slow" --benchmark-disable -q
# ~8 minutes, 112 passed
```

## Standard Benchmark Run

```bash
# All non-slow benchmarks with timing output
pytest benchmarks/ -m "not slow" --benchmark-only

# Single suite
pytest benchmarks/test_search_benchmarks.py --benchmark-only

# With verbose test names
pytest benchmarks/ -m "not slow" --benchmark-only -v
```

## Run Specific Suites

```bash
# Search performance
pytest benchmarks/test_search_benchmarks.py --benchmark-only

# Indexing throughput
pytest benchmarks/test_indexing_benchmarks.py --benchmark-only

# File chunking and PDF extraction
pytest benchmarks/test_file_processing_benchmarks.py --benchmark-only

# Disk size, cold-start, persistent write overhead
pytest benchmarks/test_persistence_benchmarks.py --benchmark-only

# Latency curves, memory growth, realistic workflows
pytest benchmarks/test_scaling_benchmarks.py --benchmark-only

# SQLite metrics subsystem overhead
pytest benchmarks/test_metrics_benchmarks.py --benchmark-only

# HNSW blue-green migration timing
pytest benchmarks/test_hnsw_migration_benchmarks.py --benchmark-only
```

Or use the runner script with flags:

```bash
python benchmarks/run_benchmarks.py --search
python benchmarks/run_benchmarks.py --metrics --hnsw
python benchmarks/run_benchmarks.py --all   # includes slow tests
```

## Filter Tests

```bash
# By expression
pytest benchmarks/ -m "not slow" --benchmark-only -k "latency"
pytest benchmarks/ -m "not slow" --benchmark-only -k "small or medium"

# One specific test
pytest benchmarks/test_search_benchmarks.py::test_search_latency_small --benchmark-only
```

## Baseline and Regression Workflow

```bash
# 1. Save a baseline before your changes
pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-save=before_my_change

# 2. Make your changes...

# 3. Compare
pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-compare=before_my_change

# 4. Enforce a regression gate (CI)
pytest benchmarks/ -m "not slow" --benchmark-only \
  --benchmark-compare=before_my_change \
  --benchmark-compare-fail=mean:20%
```

## Reading the Output

```
Name (time in ms)              Min     Max    Mean  StdDev  Median     IQR  Outliers  OPS
test_search_latency_small    10.23   15.67   11.45    1.23   11.34    0.89      5;2   87.3
```

| Column | Meaning |
|--------|---------|
| **Mean** | Average time — use this for comparisons |
| **Median** | 50th percentile — more robust to outliers |
| **StdDev/IQR** | Consistency — lower is better |
| **OPS** | Operations per second (1 / mean) |
| **Outliers** | Rounds outside 1.5×IQR; high counts mean system noise |

## Key Facts About This Suite

- **Model load is not timed.** `SentenceTransformer` loads once at session start (~5s), not inside
  any benchmark.
- **`_bulk_populate` is used for all fixture setup** — not `add_doc`. Bypasses per-document SQLite
  writes and batch-encodes everything in one pass. 67× faster than `add_doc` loops.
- **`large` (10k) and `xlarge` (50k) tests are `@pytest.mark.slow`** and never run by default.
- **Insertion benchmarks always use a fresh engine per round** via `make_ephemeral_engine()` or
  `pedantic(iterations=1)` — index state does not accumulate across rounds.
- **PersistentClient tests clean up file descriptors** by calling
  `SharedSystemClient.clear_system_cache()` on teardown.

## Slow Tests (Large Scale)

```bash
# Only slow tests
pytest benchmarks/ -m "slow" --benchmark-only

# Everything
pytest benchmarks/ --benchmark-only
```

Slow tests include 10k-doc search latency, 10k-doc HNSW migration, memory growth at 1k–10k docs,
and batch insertion of 1,000 documents.

## Troubleshooting

**Tests fail to import**
```bash
uv sync --group dev
```

**Out of memory on large tests**
```
Use -m "not slow" to skip xlarge/large fixtures
```

**Inconsistent results**
```bash
pytest benchmarks/ --benchmark-only --benchmark-min-rounds=10
# Also: close background applications
```

**Want timing data as JSON**
```bash
pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-json=results/run.json
```

---

See `benchmarks/README.md` for full documentation including rationale, architecture decisions, and
per-suite descriptions.
