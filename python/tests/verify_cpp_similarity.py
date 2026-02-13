"""Verification script for C++ cosine_similarity_batch numerical correctness.

This script compares the C++ implementation against the Python implementation
to ensure numerical equivalence before integration into VectorEngine.
"""

import sys

import numpy as np

from vectorforge.vectorforge_cpp import cosine_similarity_batch

# =============================================================================
# Helper Methods
# =============================================================================


def python_cosine_similarity(query: np.ndarray, embedding: np.ndarray) -> float:
    """Python implementation matching VectorEngine._cosine_similarity."""
    return float(np.dot(query, embedding))


def python_batch_similarity(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Python batch implementation for comparison."""
    results = []

    for i in range(docs.shape[0]):
        score = python_cosine_similarity(query, docs[i])
        results.append(score)

    return np.array(results, dtype=np.float32)


# =============================================================================
# Verification Tests
# =============================================================================


def verify_identical_vectors() -> bool:
    """Verify identical vectors produce similarity of 1.0."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    cpp_result = cosine_similarity_batch(query, docs)
    py_result = python_batch_similarity(query, docs)

    return np.allclose(cpp_result, py_result, rtol=1e-5)


def verify_orthogonal_vectors() -> bool:
    """Verify orthogonal vectors produce similarity of 0.0."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    cpp_result = cosine_similarity_batch(query, docs)
    py_result = python_batch_similarity(query, docs)

    return np.allclose(cpp_result, py_result, rtol=1e-5)


def verify_multiple_documents() -> bool:
    """Verify correctness with multiple documents."""
    np.random.seed(42)
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)
    docs = np.random.randn(10, 384).astype(np.float32)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    cpp_result = cosine_similarity_batch(query, docs)
    py_result = python_batch_similarity(query, docs)

    return np.allclose(cpp_result, py_result, rtol=1e-5)


def verify_large_batch() -> bool:
    """Verify correctness with realistic large batch."""
    np.random.seed(123)
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)
    docs = np.random.randn(1000, 384).astype(np.float32)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    cpp_result = cosine_similarity_batch(query, docs)
    py_result = python_batch_similarity(query, docs)

    return np.allclose(cpp_result, py_result, rtol=1e-5, atol=1e-7)


def verify_edge_cases() -> bool:
    """Verify edge case handling."""
    query = np.array([0.5, 0.5], dtype=np.float32)
    query = query / np.linalg.norm(query)
    docs = np.array([[1.0, 0.0]], dtype=np.float32)

    cpp_result = cosine_similarity_batch(query, docs)
    py_result = python_batch_similarity(query, docs)

    return np.allclose(cpp_result, py_result, rtol=1e-5)


def verify_error_handling() -> bool:
    """Verify error handling for invalid inputs."""
    query = np.array([1.0, 0.0], dtype=np.float32)

    try:
        docs_1d = np.array([1.0, 0.0], dtype=np.float32)
        cosine_similarity_batch(query, docs_1d)
        return False
    except RuntimeError:
        pass

    try:
        docs_mismatch = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        cosine_similarity_batch(query, docs_mismatch)
        return False
    except RuntimeError:
        pass

    return True


# =============================================================================
# Main Routine and Entry Point
# =============================================================================


def main() -> int:
    """Run all verification tests."""
    tests = [
        ("Identical vectors", verify_identical_vectors),
        ("Orthogonal vectors", verify_orthogonal_vectors),
        ("Multiple documents", verify_multiple_documents),
        ("Large batch (1000 docs)", verify_large_batch),
        ("Edge cases", verify_edge_cases),
        ("Error handling", verify_error_handling),
    ]

    failed = []
    for name, test_func in tests:
        try:
            if test_func():
                status = "PASS"
            else:
                status = "FAIL"
                failed.append(name)
        except Exception as e:
            status = f"ERROR: {e}"
            failed.append(name)

        print(f"{name:.<40} {status}")

    print()
    if failed:
        print(f"FAILED: {len(failed)} test(s) failed")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print("SUCCESS: All verification tests passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
    print("Testing C++ cosine_similarity_batch vs Python implementation")
    print("=" * 70)
    print()

    test_identical_vectors()
    test_orthogonal_vectors()
    test_multiple_documents()
    test_large_batch()
    test_edge_cases()
    test_error_handling()

    print()
    print("=" * 70)
    print("All tests passed! C++ implementation is numerically correct.")
    print("=" * 70)
