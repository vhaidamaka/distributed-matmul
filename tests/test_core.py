"""
Unit tests for core modules: matrix I/O, metrics, local computation.
Run with: pytest tests/test_core.py -v
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.matrix_io import (
    generate_random_matrices,
    load_matrices,
    save_matrices,
    matrix_size_bytes,
    format_bytes,
)
from core.metrics import ComputeMetrics
from core.local_compute import local_multiply


# ─────────────────────────────────────────────
# matrix_io
# ─────────────────────────────────────────────

class TestGenerateRandomMatrices:
    def test_square(self):
        A, B = generate_random_matrices(100, 100, 100)
        assert A.shape == (100, 100)
        assert B.shape == (100, 100)

    def test_rectangular(self):
        A, B = generate_random_matrices(50, 80, 60)
        assert A.shape == (50, 80)
        assert B.shape == (80, 60)

    def test_dtype(self):
        A, B = generate_random_matrices(10, 10, 10)
        assert A.dtype == np.float64
        assert B.dtype == np.float64

    def test_values_in_range(self):
        A, B = generate_random_matrices(50, 50, 50)
        assert A.min() >= 0.0 and A.max() <= 1.0
        assert B.min() >= 0.0 and B.max() <= 1.0

    def test_reproducible(self):
        A1, B1 = generate_random_matrices(20, 20, 20)
        A2, B2 = generate_random_matrices(20, 20, 20)
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(B1, B2)


class TestSaveLoadMatrices:
    def test_npz_roundtrip(self):
        A, B = generate_random_matrices(30, 40, 20)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_matrices(path, A=A, B=B)
            A2, B2 = load_matrices(path)
            np.testing.assert_array_almost_equal(A, A2)
            np.testing.assert_array_almost_equal(B, B2)
        finally:
            os.unlink(path)

    def test_txt_roundtrip(self):
        A, B = generate_random_matrices(5, 6, 4)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            path = f.name
        try:
            save_matrices(path, A=A, B=B)
            A2, B2 = load_matrices(path)
            np.testing.assert_array_almost_equal(A, A2, decimal=5)
            np.testing.assert_array_almost_equal(B, B2, decimal=5)
        finally:
            os.unlink(path)

    def test_save_result(self):
        result = np.random.rand(10, 10)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_matrices(path, result=result)
            data = np.load(path)
            np.testing.assert_array_almost_equal(result, data["result"])
        finally:
            os.unlink(path)

    def test_incompatible_shapes_raise(self):
        A = np.random.rand(10, 5)
        B = np.random.rand(6, 10)  # 5 ≠ 6 → should raise
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_matrices(path, A=A, B=B)
            with pytest.raises(ValueError):
                load_matrices(path)
        finally:
            os.unlink(path)

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError):
            load_matrices("some_file.xyz")


class TestFormatBytes:
    def test_bytes(self):
        assert format_bytes(512) == "512.0 B"

    def test_kilobytes(self):
        assert "KB" in format_bytes(2048)

    def test_megabytes(self):
        assert "MB" in format_bytes(2 * 1024 * 1024)

    def test_gigabytes(self):
        assert "GB" in format_bytes(2 * 1024 ** 3)


class TestMatrixSizeBytes:
    def test_size(self):
        A = np.zeros((100, 100), dtype=np.float64)
        assert matrix_size_bytes(A) == 100 * 100 * 8  # float64 = 8 bytes


# ─────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────

class TestComputeMetrics:
    def test_total_time_increases(self):
        import time
        m = ComputeMetrics()
        time.sleep(0.05)
        assert m.total_time >= 0.05

    def test_finish_freezes_time(self):
        import time
        m = ComputeMetrics()
        time.sleep(0.02)
        m.finish()
        t1 = m.total_time
        time.sleep(0.05)
        t2 = m.total_time
        assert abs(t1 - t2) < 1e-6

    def test_communication_overhead(self):
        m = ComputeMetrics()
        m.serialize_time = 0.1
        m.transfer_time = 0.2
        m.deserialize_time = 0.05
        assert abs(m.communication_overhead - 0.35) < 1e-9

    def test_speedup_ratio(self):
        m = ComputeMetrics()
        m.finish()
        m.set_baseline(2.0)
        # speedup = baseline / total; total is very small so speedup >> 1
        assert m.speedup_ratio > 1.0

    def test_to_dict_keys(self):
        m = ComputeMetrics(mode="local", backend="numpy")
        m.matrix_shape_a = (100, 100)
        m.matrix_shape_b = (100, 100)
        m.finish()
        d = m.to_dict()
        for key in ["mode", "backend", "total_time_s", "compute_time_s",
                    "communication_overhead_s", "bytes_sent", "bytes_received"]:
            assert key in d


# ─────────────────────────────────────────────
# local_compute
# ─────────────────────────────────────────────

class TestLocalMultiply:
    """Integration tests — uses real process pool."""

    @pytest.mark.asyncio
    async def test_small_square(self):
        A, B = generate_random_matrices(20, 20, 20)
        expected = A @ B
        C, metrics = await local_multiply(A, B)
        np.testing.assert_array_almost_equal(C, expected, decimal=10)

    @pytest.mark.asyncio
    async def test_rectangular(self):
        A, B = generate_random_matrices(15, 25, 10)
        expected = A @ B
        C, metrics = await local_multiply(A, B)
        np.testing.assert_array_almost_equal(C, expected, decimal=10)

    @pytest.mark.asyncio
    async def test_result_shape(self):
        A, B = generate_random_matrices(30, 40, 20)
        C, _ = await local_multiply(A, B)
        assert C.shape == (30, 20)

    @pytest.mark.asyncio
    async def test_metrics_populated(self):
        A, B = generate_random_matrices(20, 20, 20)
        _, metrics = await local_multiply(A, B)
        assert metrics.total_time > 0
        assert metrics.compute_time > 0
        assert metrics.num_workers >= 1
        assert len(metrics.worker_times) > 0
        assert metrics.mode == "local"

    @pytest.mark.asyncio
    async def test_single_row(self):
        """Edge case: A has only 1 row."""
        A = np.random.rand(1, 50)
        B = np.random.rand(50, 30)
        C, _ = await local_multiply(A, B)
        np.testing.assert_array_almost_equal(C, A @ B, decimal=10)

    @pytest.mark.asyncio
    async def test_identity(self):
        """Multiplying by identity should return A."""
        n = 25
        A = np.random.rand(n, n)
        I = np.eye(n)
        C, _ = await local_multiply(A, I)
        np.testing.assert_array_almost_equal(C, A, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
