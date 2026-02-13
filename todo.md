# VectorForge Optimization Roadmap

## Python Baseline (v0.9.0) COMPLETE

**Status:** Feature-complete Python implementation with comprehensive testing
**Git tag:** `v0.9.0-python-baseline`
**Metrics:**
- 468 unit tests, 87% coverage (97% on core engine)
- 117 benchmarks with saved baseline
- Search latency: 5-19ms (10-10K docs)
- Performance documented in README

**Ready for C++/pybind11 optimization!**

---

## C++/pybind11 Optimization Plan

**Goal:** Optimize performance-critical paths with C++ while maintaining Python API compatibility

**Profiling insights:**
-  **Hotspot #1:** `_cosine_similarity()` - 1.2s for 1M calls → **PRIMARY TARGET**
-  **Hotspot #2:** Result sorting - 0.08s for 1M calls → Secondary target
-   **Not targeting:** Embedding generation (85% of time, requires ONNX/GPU)

---

### Phase 1: C++ Build Infrastructure & Hello World (Week 1)

**Objective:** Set up C++ build system and create minimal working pybind11 integration

#### Tasks:
- [x] **Learn pybind11 basics**
  - [x] Complete official tutorial: https://pybind11.readthedocs.io/en/stable/basics.html
  - [x] Focus on: NumPy array interfacing, exception handling, memory management
  - [x] Practice with "hello world" C++ function callable from Python

- [x] **Set up build system**
  - [x] Create `CMakeLists.txt` at repository root
  - [x] Configure pybind11 as dependency (via uv package manager)
  - [x] Set compiler flags: `-O3 -march=native` for performance
  - [x] Add build target for `vectorforge_cpp` extension module
  - [x] Configure CMake to use uv's Python 3.11 (not system Python)

- [x] **Create first C++ module**
  - [x] File: `cpp/src/hello.cpp`
  - [x] Implement simple function: `helloWorld()` returning string
  - [x] Create pybind11 binding
  - [x] Build shared library (vectorforge_cpp.cpython-311-x86_64-linux-gnu.so)

- [x] **Integrate with Python**
  - [x] Import C++ module in Python: `from vectorforge.vectorforge_cpp import helloWorld`
  - [x] Write tests: `tests/test_cpp_integration.py` (2 tests)
  - [x] Expose C++ module in `vectorforge/__init__.py` for convenient access
  - [x] Create `build.sh` script for automated rebuilds

- [x] **Verify toolchain**
  - [x] Confirm NumPy environment ready (basic test added)
  - [x] Run under valgrind to detect memory leaks (no leaks found)
  - [x] Update .gitignore with C++ build artifacts

#### Success Criteria:
- [x] C++ code builds successfully with CMake
- [x] Python can import and call C++ functions
- [x] NumPy arrays can be passed to/from C++ (environment verified)
- [x] All existing Python tests still pass (470 tests passing)
- [x] No memory leaks detected (valgrind clean)

#### Files Created:
- [x] `CMakeLists.txt` (configured for Python 3.11)
- [x] `cpp/src/hello.cpp` (hello world implementation)
- [x] `cpp/include/vectorforge/hello.h` (header structure)
- [x] `tests/test_cpp_integration.py` (2 integration tests)
- [x] `build.sh` (automated build script)
- [x] `.gitignore` (updated with C++ artifacts)

**Phase 1 Status: COMPLETE** ✓
**Date Completed:** February 10, 2026
**Test Results:** 470/470 tests passing (468 Python + 2 C++ integration)
**Valgrind:** No memory leaks detected

---

### Phase 2: Optimized Cosine Similarity Batch Function (Week 2)

**Objective:** Replace Python loop with vectorized C++ batch similarity calculation

#### Tasks:
- [-] **Implement C++ batch cosine similarity**
  - File: `cpp/src/similarity.cpp`
  - Function signature:
    ```cpp
    // Compute cosine similarity between query and all document embeddings
    py::array_t<float> cosine_similarity_batch(
        py::array_t<float> query_embedding,    // Shape: (384,)
        py::array_t<float> doc_embeddings      // Shape: (n_docs, 384)
    )
    // Returns: np.ndarray of shape (n_docs,) with similarity scores
    ```
  - Use Eigen library for vectorized operations (or manual SIMD)
  - Optimize with loop unrolling, SIMD intrinsics (AVX2/AVX512)
  - Handle edge cases: empty arrays, dimension mismatch

- [x] **Create pybind11 bindings**
  - Expose `cosine_similarity_batch` to Python
  - Add docstring with usage example
  - Handle memory layout (C-contiguous vs Fortran-contiguous)

- [x] **Update vector_engine.py to use C++ function**
  - Replace Python loop in `search()` (lines 379-388):
    ```python
    # Old: Python loop
    for pos, embedding in enumerate(self.embeddings):
        score = self._cosine_similarity(query_embedding, embedding)

    # New: Single C++ call
    import vectorforge_cpp
    all_embeddings = np.array(self.embeddings)  # Stack into (n, 384)
    scores = vectorforge_cpp.cosine_similarity_batch(query_embedding, all_embeddings)
    ```
  - Add fallback to Python if C++ not available (graceful degradation)

- [x] **Testing**
  - Add C++ unit tests (Google Test framework)
  - Verify numerical equivalence: C++ results == Python results
  - Test edge cases: single doc, 10K docs, empty index
  - Ensure all 468 Python tests still pass

- [ ] **Benchmarking**
  - Run `benchmarks/test_search_benchmarks.py` with C++ enabled
  - Compare against baseline: `--benchmark-compare=0001_baseline`
  - Target: >5x speedup on 10K documents

#### Success Criteria:
- C++ batch function produces identical results to Python loop
- All 468 Python tests pass
- Search latency improves by >5x on 10K documents (18ms → <3.5ms)
- No memory leaks (valgrind check)
- Graceful fallback if C++ module not available

#### Files Modified:
- `python/vectorforge/vector_engine.py` (add C++ integration)
- `python/benchmarks/results/0002_cpp_similarity.json` (new baseline)

#### Files Created:
- `cpp/src/similarity.cpp`
- `cpp/include/vectorforge/similarity.h`
- `cpp/tests/test_similarity.cpp`

---

### Phase 3: HNSW Index Integration (Week 3-4)

**Objective:** Replace linear scan with Hierarchical Navigable Small World graph index

#### Tasks:
- [ ] **Integrate hnswlib**
  - Add hnswlib as dependency (header-only library)
  - Create C++ wrapper class: `HNSWIndex`
  - Implement methods:
    - `build_index(embeddings, M=16, ef_construction=200)`
    - `search(query, k=10, ef_search=50)`
    - `add_items(embeddings, ids)`
    - `remove_items(ids)` (mark deleted)

- [ ] **Create pybind11 bindings**
  - Expose `HNSWIndex` as Python class
  - Handle NumPy array ownership (reference vs copy)
  - Implement serialization: save/load index to disk

- [ ] **Update vector_engine.py**
  - Add `use_hnsw: bool` flag to `VectorEngine.__init__()`
  - Implement dual-mode search:
    - Linear scan for small indexes (<1K docs)
    - HNSW for large indexes (>1K docs)
  - Migrate existing embeddings to HNSW on first search

- [ ] **Testing**
  - Test HNSW accuracy: recall@10 should be >95%
  - Test with metadata filtering (post-HNSW filtering)
  - Ensure all tests pass with `use_hnsw=True`
  - Test index persistence (save/load)

- [ ] **Benchmarking**
  - Add new benchmarks: 50K, 100K, 500K documents
  - Compare HNSW vs linear scan
  - Target: <5ms search on 100K documents (vs ~1800ms linear)

#### Success Criteria:
- HNSW search produces >95% recall compared to linear scan
- Search latency <5ms on 100K documents
- All metadata filtering still works
- Index can be saved/loaded from disk
- All tests pass in both linear and HNSW modes

#### Files Modified:
- `python/vectorforge/vector_engine.py` (add HNSW mode)
- `python/vectorforge/__init__.py` (expose HNSW option)

#### Files Created:
- `cpp/src/hnsw_wrapper.cpp`
- `cpp/include/vectorforge/hnsw_wrapper.h`
- `cpp/tests/test_hnsw.cpp`
- `python/benchmarks/test_hnsw_benchmarks.py`

---

### Phase 4: Polish & Production Readiness (Week 5)

**Objective:** Production-grade error handling, documentation, and CI/CD

#### Tasks:
- [ ] **Error handling**
  - Comprehensive C++ exception handling
  - Clear error messages propagated to Python
  - Validate input dimensions/types at C++ boundary

- [ ] **Documentation**
  - Update README with C++ build instructions
  - Document HNSW configuration parameters
  - Add performance comparison charts (before/after C++)
  - Update CHANGELOG.md with v1.0.0 release notes

- [ ] **CI/CD**
  - GitHub Actions: build C++ extension on push/PR
  - Test on multiple platforms (Linux, macOS)
  - Run benchmarks in CI (regression detection)
  - Publish wheels to PyPI (optional)

- [ ] **Packaging**
  - Create source distribution with C++ code
  - Binary wheels for common platforms (manylinux, macOS)
  - Update `pyproject.toml` with build requirements

- [ ] **Performance tuning**
  - Profile C++ code with perf/gprof
  - Optimize memory allocations
  - Tune HNSW parameters (M, ef_construction, ef_search)
  - Add benchmarking guide to docs

#### Success Criteria:
- Clean builds on Linux and macOS
- CI runs tests and benchmarks automatically
- Comprehensive error messages for all failure modes
- README has clear C++ build/install instructions
- Performance metrics documented and reproducible

---

## Future Enhancements (Post-v1.0)

### GPU Acceleration
- Investigate CUDA/ROCm for batch similarity on GPU
- Requires: ONNX Runtime for embedding generation on GPU
- Expected: 50-100x speedup for >1M documents

### Alternative Indices
- FAISS (Facebook AI Similarity Search) - more features than hnswlib
- Annoy (Spotify) - simpler, good for static indices
- ScaNN (Google) - state-of-the-art ANN

### Advanced Features
- Multi-index support (separate indices per collection)
- Incremental HNSW updates (avoid full rebuild)
- Quantization (PQ/SQ) for memory reduction
- Hybrid search (sparse + dense embeddings)

---

## Success Metrics

### Performance Targets
- **Phase 1:** Build system works, tests pass
- **Phase 2:** Search latency <3.5ms on 10K docs (5-10x speedup)
- **Phase 3:** Search latency <5ms on 100K docs (100x+ speedup)
- **Phase 4:** Production-ready CI/CD, docs, packaging

### Quality Targets
- All 468 Python tests pass in all modes
- No memory leaks (valgrind clean)
- >95% HNSW recall vs exact search
- Comprehensive error handling
- Clear documentation

---

## Resources

### Learning Materials
- **pybind11 tutorial:** https://pybind11.readthedocs.io/en/stable/
- **NumPy C API:** https://numpy.org/doc/stable/reference/c-api/
- **Eigen library:** https://eigen.tuxfamily.org/
- **hnswlib:** https://github.com/nmslib/hnswlib

### Reference Implementations
- **Sentence Transformers:** https://github.com/UKPLab/sentence-transformers
- **FAISS:** https://github.com/facebookresearch/faiss
- **Annoy:** https://github.com/spotify/annoy

### Profiling Tools
- **cProfile:** Built-in Python profiler
- **perf:** Linux performance counter tool
- **valgrind:** Memory leak detection
- **Google Benchmark:** C++ microbenchmarking framework
