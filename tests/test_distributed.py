"""
Integration tests for the custom TCP distributed protocol.
Spins up a real worker in-process and tests head→worker communication.

Run with: pytest tests/test_distributed.py -v
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.matrix_io import generate_random_matrices
from distributed.worker_node import WorkerNode
from distributed.custom_head_node import CustomHeadNode, WorkerConnection


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

FREE_PORT = 19500  # use a high port to avoid conflicts


async def start_worker(port: int) -> asyncio.AbstractServer:
    """Start a worker node, return the server object for teardown."""
    worker = WorkerNode(host="127.0.0.1", port=port)
    server = await asyncio.start_server(
        worker._handle_connection, "127.0.0.1", port
    )
    return server


# ─────────────────────────────────────────────
# Serialisation round-trip
# ─────────────────────────────────────────────

class TestSerialisation:
    def test_numpy_roundtrip(self):
        from distributed.worker_node import serialize_arrays, deserialize_arrays
        A = np.random.rand(20, 30)
        B = np.random.rand(30, 15)
        payload = serialize_arrays(A_block=A, B=B, task_id=7)
        data = deserialize_arrays(payload)
        np.testing.assert_array_almost_equal(data["A_block"], A)
        np.testing.assert_array_almost_equal(data["B"], B)
        assert data["task_id"] == 7

    def test_empty_array(self):
        from distributed.worker_node import serialize_arrays, deserialize_arrays
        A = np.zeros((0, 5))
        payload = serialize_arrays(A_block=A, task_id=0)
        data = deserialize_arrays(payload)
        assert data["A_block"].shape == (0, 5)

    def test_large_array(self):
        from distributed.worker_node import serialize_arrays, deserialize_arrays
        A = np.random.rand(200, 200)
        payload = serialize_arrays(arr=A)
        data = deserialize_arrays(payload)
        np.testing.assert_array_almost_equal(data["arr"], A)


# ─────────────────────────────────────────────
# Protocol message encoding
# ─────────────────────────────────────────────

class TestProtocol:
    def test_encode_decode_header(self):
        import struct
        from distributed.worker_node import encode_message, MSG_TASK
        payload = b"hello world"
        msg = encode_message(MSG_TASK, payload)
        # header = 4 + 8 = 12 bytes
        assert len(msg) == 12 + len(payload)
        msg_type, length = struct.unpack(">IQ", msg[:12])
        assert msg_type == MSG_TASK
        assert length == len(payload)
        assert msg[12:] == payload

    def test_empty_payload(self):
        from distributed.worker_node import encode_message, MSG_PING
        msg = encode_message(MSG_PING, b"")
        assert len(msg) == 12

    def test_large_payload_length_field(self):
        import struct
        from distributed.worker_node import encode_message, MSG_RESULT
        payload = b"x" * 100_000
        msg = encode_message(MSG_RESULT, payload)
        _, length = struct.unpack(">IQ", msg[:12])
        assert length == 100_000


# ─────────────────────────────────────────────
# Live worker ↔ head node
# ─────────────────────────────────────────────

@pytest.mark.asyncio
class TestWorkerHeadIntegration:
    """
    Spins up a real WorkerNode and connects to it with a CustomHeadNode.
    Uses localhost so no actual network needed.
    """

    async def _run_with_worker(self, port: int, A, B):
        """Helper: start worker, run multiply, stop worker."""
        worker = WorkerNode(host="127.0.0.1", port=port)
        server = await asyncio.start_server(
            worker._handle_connection, "127.0.0.1", port
        )
        try:
            head = CustomHeadNode(servers=[f"127.0.0.1:{port}"])
            C, metrics = await head.multiply(A, B)
            return C, metrics
        finally:
            server.close()
            await server.wait_closed()

    async def test_small_multiply(self):
        A, B = generate_random_matrices(20, 20, 20)
        C, _ = await self._run_with_worker(FREE_PORT, A, B)
        np.testing.assert_array_almost_equal(C, A @ B, decimal=10)

    async def test_result_shape(self):
        A, B = generate_random_matrices(30, 40, 15)
        C, _ = await self._run_with_worker(FREE_PORT + 1, A, B)
        assert C.shape == (30, 15)

    async def test_metrics_mode(self):
        A, B = generate_random_matrices(10, 10, 10)
        _, metrics = await self._run_with_worker(FREE_PORT + 2, A, B)
        assert metrics.mode == "distributed"
        assert metrics.backend == "custom"
        assert metrics.num_workers == 1

    async def test_ping_pong(self):
        """Direct connection test — ping should get pong."""
        port = FREE_PORT + 3
        worker = WorkerNode(host="127.0.0.1", port=port)
        server = await asyncio.start_server(
            worker._handle_connection, "127.0.0.1", port
        )
        try:
            conn = WorkerConnection("127.0.0.1", port)
            await conn.connect()
            ok = await conn.ping()
            assert ok is True
            await conn.close()
        finally:
            server.close()
            await server.wait_closed()

    async def test_transfer_metrics_nonzero(self):
        A, B = generate_random_matrices(20, 20, 20)
        _, metrics = await self._run_with_worker(FREE_PORT + 4, A, B)
        assert metrics.bytes_sent > 0
        assert metrics.bytes_received > 0
        assert metrics.total_time > 0

    async def test_rectangular_matrices(self):
        A, B = generate_random_matrices(12, 18, 8)
        C, _ = await self._run_with_worker(FREE_PORT + 5, A, B)
        np.testing.assert_array_almost_equal(C, A @ B, decimal=10)


# ─────────────────────────────────────────────
# WorkerConnection error handling
# ─────────────────────────────────────────────

@pytest.mark.asyncio
class TestWorkerConnectionErrors:
    async def test_connect_to_closed_port_raises(self):
        conn = WorkerConnection("127.0.0.1", 19999)
        with pytest.raises(Exception):
            await conn.connect(timeout=1.0)

    async def test_ping_on_disconnected_returns_false(self):
        conn = WorkerConnection("127.0.0.1", 19998)
        conn.reader = None
        conn.writer = None
        result = await conn.ping()
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
