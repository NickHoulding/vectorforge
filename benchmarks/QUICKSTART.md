# VectorForge Benchmarking - Quick Start

## Installation

Ensure dev dependencies are installed:
```bash
cd python
uv sync --group dev
```

## Quick Commands

### Run Fast Benchmarks (Recommended for Development)
```bash
uv run pytest benchmarks/ --benchmark-only -m "not slow"
```

### Run All Benchmarks
```bash
uv run pytest benchmarks/ --benchmark-only
```

### Run Specific Benchmark Suite
```bash
# Search benchmarks
uv run pytest benchmarks/test_search_benchmarks.py --benchmark-only

# Indexing benchmarks
uv run pytest benchmarks/test_indexing_benchmarks.py --benchmark-only

# File processing benchmarks
uv run pytest benchmarks/test_file_processing_benchmarks.py --benchmark-only

# Persistence benchmarks
uv run pytest benchmarks/test_persistence_benchmarks.py --benchmark-only

# Scaling benchmarks
uv run pytest benchmarks/test_scaling_benchmarks.py --benchmark-only
```

### Save and Compare Results
```bash
# Save baseline
uv run pytest benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare against baseline (after making changes)
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=baseline

# Compare and fail if performance regresses >20%
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=mean:20%
```

### Run Specific Tests
```bash
# Run tests matching pattern
uv run pytest benchmarks/ --benchmark-only -k "search_latency"

# Run only small scale tests
uv run pytest benchmarks/ --benchmark-only -k "small"

# Run specific test
uv run pytest benchmarks/test_search_benchmarks.py::test_search_latency_small --benchmark-only
```

## Understanding Results

Example output:
```
Name (time in ms)           Min     Max    Mean  StdDev  Median     IQR  Outliers  OPS
test_search_small         5.13   6.56   5.33    0.24    5.25    0.17     20;16  187.75
```

- **Mean**: Average time (focus on this for general performance)
- **Median**: 50th percentile (less affected by outliers)
- **Min/Max**: Best and worst case
- **StdDev**: Consistency (lower is better)
- **OPS**: Operations per second
- **Outliers**: Unusual measurements

## Key Metrics by Test Type

### Search Performance
- Query latency: How fast can we search?
- Scales: tiny (10 docs) → xlarge (50k docs)
- Top-k impact: Does result count matter?

### Indexing Performance
- Documents/second: How fast can we add documents?
- Batch efficiency: Single vs batch insertion
- Compaction overhead: Impact of deletions

### File Processing
- Extraction speed: PDF/text parsing
- Chunking throughput: Text splitting performance

### Persistence
- Save/load time: Disk I/O performance
- File sizes: Compression effectiveness
- Round-trip: Complete save→load cycle

### Scaling
- Performance curves: How does it scale?
- Memory usage: Resource consumption
- Real-world workflows: End-to-end scenarios

## Typical Development Workflow

1. **Baseline**: Before making changes
   ```bash
   uv run pytest benchmarks/ --benchmark-only -m "not slow" --benchmark-save=before_changes
   ```

2. **Make your changes**: Optimize code, refactor, etc.

3. **Compare**: After changes
   ```bash
   uv run pytest benchmarks/ --benchmark-only -m "not slow" --benchmark-compare=before_changes
   ```

4. **Analyze**: Look for improvements or regressions
   - Green = faster (good!)
   - Red = slower (investigate)
   - Focus on mean/median values

5. **Iterate**: Repeat as needed

## Tips

- **First run slow**: Model loading happens on first run. Subsequent runs are faster.
- **Close apps**: For consistent results, close heavy applications
- **Warm up**: Benchmarks automatically warm up, but first overall run may vary
- **Use `-m "not slow"`**: Skip large-scale tests during development
- **Save results**: Track performance over time with `--benchmark-save`

## Next Steps

See `benchmarks/README.md` for comprehensive documentation.

## Example Session

```bash
# Quick check - how's search performance?
cd python
uv run pytest benchmarks/test_search_benchmarks.py -k "small or medium" --benchmark-only

# Full benchmark suite (takes ~5-10 minutes)
uv run pytest benchmarks/ --benchmark-only --benchmark-save=v0.9.0

# Compare after optimization
# ... make changes ...
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=v0.9.0
```

## Troubleshooting

**Issue**: Tests failing to import
- **Fix**: `uv sync --group dev`

**Issue**: Out of memory on large tests
- **Fix**: Use `-m "not slow"` or increase system RAM

**Issue**: Inconsistent results
- **Fix**: Increase rounds with `--benchmark-min-rounds=10`

**Issue**: Want faster feedback
- **Fix**: Run specific tests with `-k pattern`
