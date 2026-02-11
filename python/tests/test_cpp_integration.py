"""Tests for Python/C++ integration"""

import numpy as np

from vectorforge.vectorforge_cpp import helloWorld

# =============================================================================
# CPP Integration tests
# =============================================================================


def test_cpp_integration_hello_world(client):
    """Test that Python/C++ integration is functioning properly."""
    assert isinstance(helloWorld(), str)
    assert helloWorld() == "Hello World!"


def test_cpp_numpy_environment(client):
    """Verify NumPy environment is ready for C++ array passing (Phase 2)."""
    # Basic NumPy check - actual array passing will be tested in Phase 2
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert arr.dtype == np.float32
    assert len(arr) == 3

    # Verify C++ module still works
    assert helloWorld() == "Hello World!"
