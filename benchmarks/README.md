# VectorForge Benchmarking Suite

Performance benchmarks for VectorForge covering the three most important dimensions: search latency, indexing throughput, and file processing. This suite is designed to be minimal, yet cover the most important benchmarking metrics to reveal how the main features of the system performs.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Scales](#data-scales)
- [What Each Suite Benchmarks](#what-each-suite-benchmarks)
- [Running Benchmarks](#running-benchmarks)
- [Markers](#markers)
- [Interpreting Results](#interpreting-results)
- [Tips](#tips)
- [Structure](#structure)
- [Resources](#resources)

---

## Overview

| File | Tests | What it measures |
|------|-------|-----------------|
| `test_search_benchmarks.py` | 3 | Query latency at small (100), medium (1,000), and large (10,000) index sizes |
| `test_indexing_benchmarks.py` | 2 | Single-doc insertion cost; 100-doc sequential insertion throughput |
| `test_file_processing_benchmarks.py` | 2 | Text chunking throughput; PDF extraction speed |

---

## Architecture

### ChromaDB Backend

All benchmarks construct `VectorEngine(collection, model, chroma_client)` directly, bypassing
`CollectionManager`. An **EphemeralClient** (in-memory, no disk I/O) is used throughout, keeping
setup fast and results free of disk-I/O noise.

### Fast Fixture Population: `_bulk_populate`

The helper `_bulk_populate(engine, docs, batch_size=500)` is used in all pre-populated fixtures
instead of calling `engine.add_docs()` in a loop. It batch-encodes all documents in a single
`model.encode()` call, normalizes vectors manually, then calls `engine.collection.add()` in
batches of 500.

**Rationale:** `add_docs` costs ~46ms/call (embedding + ChromaDB insert). At 1,000 docs that is
46 seconds of setup time. `_bulk_populate` reduces this to ~0.7 seconds (67× faster).

### Shared Model

`shared_model` is a session-scoped fixture that loads `SentenceTransformer` exactly once per
pytest run (~5 seconds). All engine fixtures receive it, so model-load time is never included in
benchmark measurements.

### Avoiding Accumulation Bugs

Every test that measures insertion cost uses `make_ephemeral_engine()`, a function-scoped factory
called inside the timed function, so each benchmark round starts from a clean, empty engine.

---

## Data Scales

```
SCALES = {
    "small":  100 documents,
    "medium": 1,000 documents,
    "large":  10,000 documents,   # @pytest.mark.slow
}
```

The `large` fixture is only used by `@pytest.mark.slow` tests. The default run
(`-m "not slow"`) never touches it.

---

## What Each Suite Benchmarks

### Search (`test_search_benchmarks.py`)

Query latency at three index sizes using `top_k=10` and a fixed query string:

- `test_search_latency_small`: 100 docs
- `test_search_latency_medium`: 1,000 docs
- `test_search_latency_large`: 10,000 docs (`@pytest.mark.slow`)

### Indexing (`test_indexing_benchmarks.py`)

- `test_add_single_doc`: cost of embedding + inserting one document into an empty index
  (5 rounds, fresh engine per round via `make_ephemeral_engine`)
- `test_batch_insert_100_docs`: total time to insert 100 documents sequentially into a fresh engine
  (3 rounds; uses `add_docs` in a loop, measuring the full per-call cost at scale)

### File Processing (`test_file_processing_benchmarks.py`)

Pure-function tests (no VectorEngine dependency):

- `test_chunk_medium_text`: `chunk_text` throughput on ~1,000-word text at default settings
  (chunk_size=500, overlap=50)
- `test_extract_pdf_synthetic_medium`: `extract_pdf` speed on a 5-page synthetic PDF generated
  in-memory with PyMuPDF (`fitz`)

---

## Running Benchmarks

### Prerequisites

```bash
uv sync --group dev
```

### Smoke Test (no timing; just verify nothing is broken)

```bash
uv run pytest benchmarks/ -m "not slow" --benchmark-disable -q
```

### Standard Benchmark Run

```bash
# All non-slow benchmarks with timing
uv run pytest benchmarks/ -m "not slow" --benchmark-only

# Single suite
uv run pytest benchmarks/test_search_benchmarks.py --benchmark-only
```

### Slow Tests (large-scale)

```bash
uv run pytest benchmarks/ --benchmark-only
```

### Saving and Comparing Results

```bash
# Save a baseline
uv run pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-save=baseline

# Compare after changes
uv run pytest benchmarks/ -m "not slow" --benchmark-only --benchmark-compare=baseline

# Fail CI if mean regresses more than 20%
uv run pytest benchmarks/ -m "not slow" --benchmark-only \
  --benchmark-compare=baseline \
  --benchmark-compare-fail=mean:20%
```

---

## Markers

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.slow` | Skipped by `-m "not slow"`; large-scale tests only |

---

## Interpreting Results

```
Name (time in ms)              Min      Max     Mean  StdDev  Median     IQR  Outliers  OPS
test_search_latency_small    10.23   15.67    11.45    1.23   11.34    0.89      5;2   87.3
```

- **Min/Max**: fastest and slowest rounds
- **Mean/Median**: focus on median for skewed distributions
- **StdDev/IQR**: consistency; lower is better
- **OPS**: operations per second (inverse of mean)
- **Outliers**: rounds outside 1.5×IQR; high counts suggest system noise

---

## Tips

1. **Model load time is not measured.** `SentenceTransformer` loads once before any benchmark (~5s).
2. **Run on a quiet system.** Close background processes for consistent results.
3. **The large (10k-doc) fixture is gated behind `@pytest.mark.slow`.** It takes ~10s to populate
   even with `_bulk_populate`; never add it to the default suite.
4. **Insertion benchmarks never reuse engines across rounds.** `make_ephemeral_engine()` is called
   inside the timed function to get a fresh index each round.

---

## Structure

```
benchmarks/
├── __init__.py
├── conftest.py                          # Fixtures, _bulk_populate, SCALES constant
├── test_search_benchmarks.py            # 3 tests: latency at small/medium/large
├── test_indexing_benchmarks.py          # 2 tests: single-doc and 100-doc sequential
├── test_file_processing_benchmarks.py   # 2 tests: chunking and PDF extraction
├── QUICKSTART.md
└── README.md
```

---

## Resources

- [VectorForge main docs](../README.md)

---

<div align="center">
  <strong>VectorForge Benchmarking Suite</strong>
</div>
