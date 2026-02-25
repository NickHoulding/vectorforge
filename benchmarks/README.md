# VectorForge Benchmarking Suite

Comprehensive performance benchmarks for VectorForge's Python implementation.

## Overview

This benchmarking suite tests VectorForge across multiple dimensions:

- **Search Performance** (`test_search_benchmarks.py`) - Query latency, throughput, scaling
- **Indexing Performance** (`test_indexing_benchmarks.py`) - Document addition speed, batch insertion
- **File Processing** (`test_file_processing_benchmarks.py`) - PDF extraction, text chunking
- **Persistence** (`test_persistence_benchmarks.py`) - Save/load operations, disk I/O
- **Scaling Behavior** (`test_scaling_benchmarks.py`) - End-to-end workflows, realistic scenarios

## Quick Start

### Run all benchmarks (fast tests only):
```bash
cd python
pytest benchmarks/ --benchmark-only
```

### Run including slow tests:
```bash
pytest benchmarks/ --benchmark-only -m "not slow"  # Fast only
pytest benchmarks/ --benchmark-only                # All tests
```

### Run specific benchmark file:
```bash
pytest benchmarks/test_search_benchmarks.py --benchmark-only
```

### Run with specific scale:
```bash
pytest benchmarks/ --benchmark-only -m "scale(size='small')"
```

## Benchmark Commands

### Basic Usage

```bash
# Run benchmarks and display results
pytest benchmarks/ --benchmark-only

# Run with verbose output
pytest benchmarks/ --benchmark-only -v

# Run and skip slow tests
pytest benchmarks/ --benchmark-only -m "not slow"
```

### Saving and Comparing Results

```bash
# Save baseline results
pytest benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare against baseline
pytest benchmarks/ --benchmark-only --benchmark-compare=baseline

# Compare and show full results
pytest benchmarks/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
```

### Advanced Options

```bash
# Generate histogram (requires pygal and pygaljs)
pytest benchmarks/ --benchmark-only --benchmark-histogram

# Save results as JSON
pytest benchmarks/ --benchmark-only --benchmark-json=results/benchmark_results.json

# Run only specific benchmarks
pytest benchmarks/test_search_benchmarks.py::test_search_latency_small --benchmark-only

# Run with custom rounds and iterations
pytest benchmarks/ --benchmark-only --benchmark-min-rounds=10
```

## Test Organization

### Scales

Benchmarks use different data scales:
- **tiny**: 10 documents
- **small**: 100 documents
- **medium**: 1,000 documents
- **large**: 10,000 documents (marked `@pytest.mark.slow`)
- **xlarge**: 50,000 documents (marked `@pytest.mark.slow`)

### Markers

- `@pytest.mark.slow` - Long-running tests (skip with `-m "not slow"`)
- `@pytest.mark.scale(size="...")` - Tests at specific scale levels

## Key Metrics

### Search Performance
- Query latency (mean, min, max, p50, p95, p99)
- Queries per second (QPS)
- Impact of index size on search speed
- Top-k performance

### Indexing Performance
- Documents per second
- Batch insertion throughput
- Document size impact
- Compaction overhead

### File Processing
- Extraction speed (KB/s, MB/s)
- Chunking throughput
- End-to-end processing time

### Persistence
- Save/load time vs index size
- Compression ratios
- Disk space per document
- Round-trip performance

### Scaling
- Performance degradation curves
- Memory growth patterns
- Realistic workflow scenarios

## Example Workflows

### Daily Performance Check
```bash
# Quick check on key metrics
pytest benchmarks/ --benchmark-only -m "not slow" -k "small or medium"
```

### Full Benchmark Suite
```bash
# Complete benchmarking (takes longer)
pytest benchmarks/ --benchmark-only --benchmark-save=nightly_$(date +%Y%m%d)
```

### Regression Testing
```bash
# Save baseline before changes
pytest benchmarks/ --benchmark-only --benchmark-save=before_optimization

# Make your changes...

# Compare after changes
pytest benchmarks/ --benchmark-only --benchmark-compare=before_optimization
```

### Performance Profiling
```bash
# Focus on specific area
pytest benchmarks/test_search_benchmarks.py --benchmark-only -v

# With memory profiling
pytest benchmarks/test_scaling_benchmarks.py::test_memory_scaling --benchmark-only
```

## Interpreting Results

pytest-benchmark provides rich statistics for each test:

```
Name (time in ms)                Min      Max    Mean  StdDev  Median     IQR  Outliers
test_search_latency_small     10.23   15.67   11.45    1.23   11.34    0.89     5;2
```

- **Min/Max**: Fastest and slowest execution times
- **Mean**: Average execution time
- **StdDev**: Standard deviation (lower is more consistent)
- **Median**: 50th percentile
- **IQR**: Interquartile range (spread of middle 50%)
- **Outliers**: Number of outliers detected

## Tips

1. **Warm Up**: First run may be slower due to model loading. Results stabilize after warmup.

2. **Consistency**: Run benchmarks on a quiet system for best accuracy.

3. **Baselines**: Save baselines regularly to track performance over time.

4. **Focus**: Use `-k` to run specific tests during development:
   ```bash
   pytest benchmarks/ -k "search_latency" --benchmark-only
   ```

5. **Memory**: Some tests track memory usage. Ensure enough RAM for large-scale tests.

## Continuous Integration

For CI pipelines, use:
```bash
# Run fast benchmarks only, fail on regression
pytest benchmarks/ --benchmark-only -m "not slow" \
  --benchmark-compare=baseline \
  --benchmark-compare-fail=mean:20%
```

## Troubleshooting

**Issue**: Tests are too slow
- **Solution**: Use `-m "not slow"` to skip large-scale tests

**Issue**: Inconsistent results
- **Solution**: Increase `--benchmark-min-rounds` or close other applications

**Issue**: Out of memory
- **Solution**: Skip xlarge tests or reduce test scale in conftest.py

## Structure

```
benchmarks/
├── __init__.py                          # Package init
├── conftest.py                          # Fixtures and test data generation
├── test_search_benchmarks.py            # Search performance tests
├── test_indexing_benchmarks.py          # Indexing performance tests
├── test_file_processing_benchmarks.py   # File processing tests
├── test_persistence_benchmarks.py       # Save/load tests
├── test_scaling_benchmarks.py           # Scaling and end-to-end tests
├── datasets/                            # Sample data files
├── results/                             # Benchmark results (gitignored)
└── README.md                            # This file
```

## Next Steps

After benchmarking, you may want to:
1. Optimize hot paths identified by benchmarks
2. Compare Python implementation against future C++/HNSW implementation
3. Track performance metrics over time
4. Set up automated regression testing in CI/CD

## Resources

- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
- [VectorForge main docs](../README.md)
