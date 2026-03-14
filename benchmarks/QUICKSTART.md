# VectorForge Benchmarking Quick Start

## Prerequisites

```bash
uv sync --group dev
```

## The Fastest Sanity Check

Run all 7 non-slow tests without benchmark timing (just verify nothing is broken):

```bash
uv run pytest benchmarks/ -m "not slow" --benchmark-disable -q
# ~30 seconds, 7 passed
```

## Standard Benchmark Run

```bash
# All non-slow benchmarks with timing output
uv run pytest benchmarks/ -m "not slow" --benchmark-only

# Single suite
uv run pytest benchmarks/test_search_benchmarks.py --benchmark-only

# With verbose test names
uv run pytest benchmarks/ -m "not slow" --benchmark-only -v
```

## Run Specific Suites

```bash
# Search latency (small / medium / large index)
uv run pytest benchmarks/test_search_benchmarks.py --benchmark-only

# Indexing throughput (single doc, 100-doc sequential)
uv run pytest benchmarks/test_indexing_benchmarks.py --benchmark-only

# File chunking and PDF extraction
uv run pytest benchmarks/test_file_processing_benchmarks.py --benchmark-only
```

## Filter Tests

```bash
# By expression
uv run pytest benchmarks/ -m "not slow" --benchmark-only -k "latency"
uv run pytest benchmarks/ -m "not slow" --benchmark-only -k "small or medium"

# One specific test
uv run pytest benchmarks/test_search_benchmarks.py::test_search_latency_small --benchmark-only
```

## Baseline and Regression Workflow

```bash
# 1. Save a baseline before your changes
uv run pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-save=before_my_change

# 2. Make your changes...

# 3. Compare
uv run pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-compare=before_my_change

# 4. Enforce a regression gate (CI)
uv run pytest benchmarks/ -m "not slow" --benchmark-only \
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
| **Mean** | Average time; use this for comparisons |
| **Median** | 50th percentile; more robust to outliers |
| **StdDev/IQR** | Consistency; lower is better |
| **OPS** | Operations per second (1 / mean) |
| **Outliers** | Rounds outside 1.5×IQR; high counts mean system noise |

## Key Facts About This Suite

- **Model load is not timed.** `SentenceTransformer` loads once at session start (~5s).
- **`_bulk_populate` is used for all fixture setup** (not `add_docs`). Batch-encodes everything
  in one pass. 67× faster than `add_docs` loops.
- **The large (10k-doc) test is `@pytest.mark.slow`** and never runs by default.
- **Insertion benchmarks always use a fresh engine per round** via `make_ephemeral_engine()`;
  index state does not accumulate across rounds.

## Slow Tests (Large Scale)

```bash
# Only slow tests
uv run pytest benchmarks/ -m "slow" --benchmark-only

# Everything
uv run pytest benchmarks/ --benchmark-only
```

The slow test is `test_search_latency_large` (10,000-doc index). It takes ~10 seconds to
populate the fixture even with `_bulk_populate`.

## Troubleshooting

**Tests fail to import**
```bash
uv sync --group dev
```

**Inconsistent results**
```bash
uv run pytest benchmarks/ --benchmark-only --benchmark-min-rounds=10
# Also: close background applications
```

**Want timing data as JSON**
```bash
uv run pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-json=results/run.json
```

---

See [benchmarks/README](./README.md) for full documentation including architecture decisions and
per-suite descriptions.
