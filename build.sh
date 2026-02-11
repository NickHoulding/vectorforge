#!/bin/bash
set -e

echo "Building VectorForge C++ extensions..."
echo ""

# Clean old builds
echo "Cleaning old build artifacts..."
rm -rf build
rm -f python/vectorforge/vectorforge_cpp*.so

# Configure CMake
echo "Configuring CMake..."
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=.

# Build
echo "Building C++ extension..."
cmake --build build

# Install
echo "Installing to python/vectorforge/..."
cmake --build build --target install

# Verify
echo ""
echo "Build complete!"
ls -lh python/vectorforge/*.so

# Test
echo ""
echo "Testing C++ module..."
cd python
uv run python -c "from vectorforge.vectorforge_cpp import helloWorld; print('Module test:', helloWorld())"

echo ""
echo "C++ module is working!"
echo ""
echo "Run tests with: cd python && uv run pytest tests/test_cpp_integration.py -v"
