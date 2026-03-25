"""
Local (single-node) matrix multiplication using NumPy + multiprocessing.
Uses all available CPU cores via row-block decomposition of A.
"""

import asyncio
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Tuple

from core.metrics import ComputeMetrics


def _multiply_block(args: tuple) -> tuple:
    """
    Worker process function.
    Receives serialised bytes to avoid pickling overhead of ndarray.
    Returns (result_bytes, shape, dtype_str, elapsed_seconds).
    """
    A_bytes, A_shape, A_dtype, B_bytes, B_shape, B_dtype = args
    A_block = np.frombuffer(A_bytes, dtype=np.dtype(A_dtype)).reshape(A_shape).copy()
    B = np.frombuffer(B_bytes, dtype=np.dtype(B_dtype)).reshape(B_shape)
    t0 = time.perf_counter()
    result = A_block @ B
    elapsed = time.perf_counter() - t0
    return result.tobytes(), result.shape, result.dtype.str, elapsed


async def local_multiply(
    A: np.ndarray,
    B: np.ndarray,
    ui=None,
) -> Tuple[np.ndarray, ComputeMetrics]:
    """
    Multiply A @ B locally using all available CPU cores.
    Splits A into row-blocks — one block per core — and assembles the result.
    """
    metrics = ComputeMetrics(mode="local", backend="numpy")
    metrics.matrix_shape_a = A.shape
    metrics.matrix_shape_b = B.shape

    num_workers = min(multiprocessing.cpu_count(), A.shape[0])
    metrics.num_workers = num_workers

    if ui:
        ui.log(f"Local multiply: A{A.shape} × B{B.shape}  |  {num_workers} CPU cores")

    # ---- Split A into row blocks ----
    row_splits = np.array_split(np.arange(A.shape[0]), num_workers)
    row_splits = [idx for idx in row_splits if len(idx) > 0]
    num_blocks = len(row_splits)

    # ---- Serialise once ----
    B_bytes = B.tobytes()
    B_shape = B.shape
    B_dtype = B.dtype.str

    args_list = []
    for idx in row_splits:
        block = A[idx]
        args_list.append((
            block.tobytes(), block.shape, block.dtype.str,
            B_bytes, B_shape, B_dtype,
        ))

    if ui:
        ui.update_progress(0, num_blocks, "Dispatching blocks to CPU cores…")

    loop = asyncio.get_event_loop()
    compute_start = time.perf_counter()

    # ---- Run in process pool, preserve order via gather ----
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            loop.run_in_executor(executor, _multiply_block, arg)
            for arg in args_list
        ]
        raw_results = await asyncio.gather(*futures)

    metrics.compute_time = time.perf_counter() - compute_start

    # ---- Reassemble ----
    gather_start = time.perf_counter()
    result_blocks = []
    worker_times = []
    for i, (res_bytes, shape, dtype_str, elapsed) in enumerate(raw_results):
        block = np.frombuffer(res_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
        result_blocks.append(block)
        worker_times.append(elapsed)
        if ui:
            ui.update_progress(i + 1, num_blocks, f"Block {i + 1}/{num_blocks} done")

    C = np.vstack(result_blocks)
    metrics.gather_time = time.perf_counter() - gather_start
    metrics.worker_times = worker_times

    metrics.finish()

    if ui:
        ui.log(f"Local done in {metrics.total_time:.4f}s  "
               f"(compute={metrics.compute_time:.4f}s, gather={metrics.gather_time:.4f}s)")
    return C, metrics
