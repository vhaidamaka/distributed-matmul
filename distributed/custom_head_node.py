"""
Custom head node — own TCP-based distributed implementation (no Ray).
Connects to worker nodes, distributes matrix blocks, collects results.
"""

import asyncio
import time
import numpy as np
import logging
from typing import Tuple, List, Optional

from core.metrics import ComputeMetrics
from core.matrix_io import matrix_size_bytes
from distributed.worker_node import (
    encode_message, read_message, serialize_arrays, deserialize_arrays,
    MSG_TASK, MSG_RESULT, MSG_PING, MSG_PONG, MSG_ERROR
)

logger = logging.getLogger(__name__)


class WorkerConnection:
    """Manages an async TCP connection to one worker."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False

    async def connect(self, timeout: float = 10.0):
        self.reader, self.writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=timeout
        )
        self.connected = True
        logger.info(f"[Head] Connected to worker {self.host}:{self.port}")

    async def ping(self) -> bool:
        try:
            self.writer.write(encode_message(MSG_PING, b""))
            await self.writer.drain()
            msg_type, _ = await asyncio.wait_for(read_message(self.reader), timeout=5.0)
            return msg_type == MSG_PONG
        except Exception:
            return False

    async def send_task(self, task_id: int, A_block: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Send a multiplication task and receive the result."""
        payload = serialize_arrays(A_block=A_block, B=B, task_id=task_id)
        msg = encode_message(MSG_TASK, payload)

        t_send = time.perf_counter()
        self.writer.write(msg)
        await self.writer.drain()
        t_sent = time.perf_counter()

        msg_type, resp_payload = await read_message(self.reader)
        t_recv = time.perf_counter()

        if msg_type != MSG_RESULT:
            raise RuntimeError(f"Expected MSG_RESULT ({MSG_RESULT}), got {msg_type}")

        data = deserialize_arrays(resp_payload)
        result: np.ndarray = data["result"]
        worker_elapsed: float = float(data.get("elapsed", 0.0))
        recv_task_id: int = int(data.get("task_id", task_id))

        net_time = (t_recv - t_send) - worker_elapsed
        return result, worker_elapsed, net_time

    async def close(self):
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
        self.connected = False


class CustomHeadNode:
    """
    Head node using own TCP protocol (no Ray).
    """

    def __init__(self, servers: List[str], ui=None):
        self.servers = servers
        self.ui = ui
        self.num_workers = len(servers)
        self.connections: List[WorkerConnection] = []

    def _log(self, msg: str):
        if self.ui:
            self.ui.log(msg)

    def _parse_server(self, addr: str) -> Tuple[str, int]:
        parts = addr.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid server address: {addr}. Expected IP:PORT")
        return parts[0], int(parts[1])

    async def _connect_all(self):
        """Establish connections to all workers."""
        self.connections = []
        for addr in self.servers:
            host, port = self._parse_server(addr)
            conn = WorkerConnection(host, port)
            try:
                await conn.connect(timeout=10.0)
                ok = await conn.ping()
                if ok:
                    self._log(f"[Custom] Worker {addr} is alive")
                    self.connections.append(conn)
                else:
                    self._log(f"[Custom] Worker {addr} ping failed — skipping")
            except Exception as e:
                self._log(f"[Custom] Cannot connect to {addr}: {e} — skipping")

        if not self.connections:
            raise RuntimeError("No workers available! Start worker nodes first.")

        self._log(f"[Custom] Connected to {len(self.connections)} worker(s)")

    async def _close_all(self):
        for conn in self.connections:
            await conn.close()
        self.connections = []

    async def multiply(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, ComputeMetrics]:
        """Distribute A @ B across custom TCP workers."""
        metrics = ComputeMetrics(mode="distributed", backend="custom")
        metrics.matrix_shape_a = A.shape
        metrics.matrix_shape_b = B.shape

        await self._connect_all()
        metrics.num_workers = len(self.connections)

        # Split A into row blocks — one per available worker
        row_splits = np.array_split(A, metrics.num_workers, axis=0)
        row_splits = [block for block in row_splits if block.shape[0] > 0]
        num_blocks = len(row_splits)

        self._log(f"[Custom] Split A{A.shape} into {num_blocks} blocks for {metrics.num_workers} worker(s)")

        # Measure serialization overhead
        serialize_start = time.perf_counter()
        _ = serialize_arrays(A_block=row_splits[0], B=B, task_id=0)  # probe
        metrics.serialize_time = (time.perf_counter() - serialize_start) * num_blocks  # estimate
        metrics.bytes_sent = matrix_size_bytes(A) + matrix_size_bytes(B) * num_blocks

        if self.ui:
            self.ui.update_progress(0, num_blocks, "Dispatching tasks to workers...")

        # Dispatch tasks concurrently to all workers
        async def do_task(i: int, conn: WorkerConnection, block: np.ndarray):
            result, worker_time, net_time = await conn.send_task(i, block, B)
            return i, result, worker_time, net_time

        compute_start = time.perf_counter()

        # Assign tasks: if more blocks than workers, wrap around
        tasks = []
        for i, block in enumerate(row_splits):
            conn = self.connections[i % len(self.connections)]
            tasks.append(asyncio.create_task(do_task(i, conn, block)))

        # Collect results in order
        ordered_results = [None] * num_blocks
        net_times = []
        worker_times_list = []
        completed = 0

        for coro in asyncio.as_completed(tasks):
            i, result_block, worker_time, net_time = await coro
            ordered_results[i] = result_block
            worker_times_list.append(worker_time)
            net_times.append(net_time)
            completed += 1
            if self.ui:
                self.ui.update_progress(completed, num_blocks, f"Task {completed}/{num_blocks} complete")

        metrics.compute_time = time.perf_counter() - compute_start
        metrics.worker_times = worker_times_list
        metrics.transfer_time = sum(net_times)

        # Assemble result
        gather_start = time.perf_counter()
        C = np.vstack([r for r in ordered_results if r is not None])
        metrics.gather_time = time.perf_counter() - gather_start
        metrics.bytes_received = matrix_size_bytes(C)

        await self._close_all()

        metrics.finish()
        self._log(f"[Custom] Done. C{C.shape}, total={metrics.total_time:.3f}s, "
                  f"net={metrics.transfer_time:.3f}s, compute={metrics.compute_time:.3f}s")

        return C, metrics
