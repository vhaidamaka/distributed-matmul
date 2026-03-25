"""
Head node implementation using Ray for distributed matrix multiplication.
"""

import time
import numpy as np
from typing import Tuple, List

from core.metrics import ComputeMetrics
from core.matrix_io import matrix_size_bytes


class HeadNode:
    """
    Head node that distributes matrix multiplication across Ray workers.
    Each remote server runs a Ray worker actor.
    """

    def __init__(self, servers: List[str], ui=None):
        self.servers = servers
        self.ui = ui
        self.num_workers = len(servers)

    def _log(self, msg: str):
        if self.ui:
            self.ui.log(msg)

    async def multiply(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, ComputeMetrics]:
        """Distribute A @ B across Ray workers."""
        try:
            import ray
        except ImportError:
            raise ImportError(
                "Ray is not installed. Install it with: pip install ray\n"
                "Or use --backend custom for the built-in implementation."
            )

        metrics = ComputeMetrics(mode="distributed", backend="ray")
        metrics.matrix_shape_a = A.shape
        metrics.matrix_shape_b = B.shape
        metrics.num_workers = self.num_workers

        self._log(f"[Ray] Initializing Ray cluster with {self.num_workers} worker(s)...")

        # Connect to Ray cluster (head node must be running)
        # For single-machine simulation, we init locally
        if not ray.is_initialized():
            # Parse first server as head address
            head_addr = self.servers[0]
            try:
                ray.init(address=f"ray://{head_addr}", ignore_reinit_error=True)
                self._log(f"[Ray] Connected to cluster at {head_addr}")
            except Exception as e:
                self._log(f"[Ray] Could not connect to {head_addr}: {e}. Falling back to local Ray.")
                ray.init(ignore_reinit_error=True, num_cpus=self.num_workers)

        # Define the remote worker function
        @ray.remote
        def multiply_block(A_block: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
            t0 = time.perf_counter()
            result = A_block @ B
            return result, time.perf_counter() - t0

        # Split A into row blocks
        row_splits = np.array_split(A, self.num_workers, axis=0)
        row_splits = [block for block in row_splits if block.shape[0] > 0]
        actual_workers = len(row_splits)
        self._log(f"[Ray] Split A{A.shape} into {actual_workers} blocks")

        # Serialize B once (Ray handles distribution)
        serialize_start = time.perf_counter()
        B_ref = ray.put(B)  # Store B in Ray object store once
        metrics.serialize_time = time.perf_counter() - serialize_start
        metrics.bytes_sent = matrix_size_bytes(A) + matrix_size_bytes(B)

        if self.ui:
            self.ui.update_progress(0, actual_workers, "Dispatching to Ray workers...")

        # Dispatch tasks
        transfer_start = time.perf_counter()
        futures = [multiply_block.remote(block, B_ref) for block in row_splits]
        metrics.transfer_time = time.perf_counter() - transfer_start

        # Gather results
        compute_start = time.perf_counter()
        raw_results = []
        for i, future in enumerate(futures):
            result_block, worker_time = ray.get(future)
            raw_results.append(result_block)
            metrics.worker_times.append(worker_time)
            if self.ui:
                self.ui.update_progress(i + 1, actual_workers, f"Worker {i+1}/{actual_workers} done")
        metrics.compute_time = time.perf_counter() - compute_start

        # Assemble
        gather_start = time.perf_counter()
        C = np.vstack(raw_results)
        metrics.gather_time = time.perf_counter() - gather_start
        metrics.bytes_received = matrix_size_bytes(C)

        metrics.finish()
        self._log(f"[Ray] Done. Result shape: {C.shape}, total time: {metrics.total_time:.3f}s")
        return C, metrics
