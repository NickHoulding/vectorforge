# VectorForge
VectorForge is a high-performance document retrieval system that demonstrates the power of C++/Python integration for production ML workloads. Built from the ground up with a pure Python baseline and custom C++ vector search engine, this project showcases measurable performance optimization through profiling-driven development and low-level systems programming.

## Project Structure
```
vectorforge/
├── README.md
├── benchmarks/
│   ├── benchmark_results.md
│   └── profiling_scripts/
├── python/
│   ├── vectorforge/
│   │   ├── api.py
│   │   ├── document_processor.py
│   │   ├── embeddings.py
│   │   └── search.py (pure Python version)
│   └── tests/
├── cpp/
│   ├── src/
│   │   ├── vector_index.cpp
│   │   ├── similarity.cpp
│   │   └── bindings.cpp
│   ├── include/
│   └── CMakeLists.txt
├── datasets/
│   └── sample_documents/
└── setup.py
```
