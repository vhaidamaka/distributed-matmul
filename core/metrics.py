"""
Metrics collection for matrix multiplication benchmarking.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class ComputeMetrics:
    """All timing and transfer metrics for one multiplication run."""

    # Overall timing
    total_start: float = field(default_factory=time.perf_counter)
    total_end: Optional[float] = None

    # Sub-timings
    serialize_time: float = 0.0       # Time to serialize matrices for sending
    transfer_time: float = 0.0        # Network transfer time (send + receive)
    compute_time: float = 0.0         # Pure computation time across all workers
    gather_time: float = 0.0          # Time to collect and merge partial results
    deserialize_time: float = 0.0     # Time to deserialize received data

    # Data transfer volumes
    bytes_sent: int = 0
    bytes_received: int = 0

    # Mode info
    mode: str = "local"               # 'local' | 'distributed'
    backend: str = "numpy"            # 'numpy' | 'ray' | 'custom'
    num_workers: int = 1
    matrix_shape_a: tuple = ()
    matrix_shape_b: tuple = ()

    # Per-worker breakdown
    worker_times: list = field(default_factory=list)   # list of float (seconds per worker)

    def finish(self):
        """Record end time."""
        self.total_end = time.perf_counter()

    @property
    def total_time(self) -> float:
        if self.total_end is None:
            return time.perf_counter() - self.total_start
        return self.total_end - self.total_start

    @property
    def communication_overhead(self) -> float:
        """Time spent on serialization + network transfer."""
        return self.serialize_time + self.transfer_time + self.deserialize_time

    @property
    def speedup_ratio(self) -> float:
        """
        Speedup = T_local / T_distributed.
        Requires baseline to be set externally; otherwise returns 0.
        """
        return getattr(self, "_baseline_time", 0.0) / self.total_time if self.total_time > 0 else 0.0

    def set_baseline(self, baseline_seconds: float):
        self._baseline_time = baseline_seconds

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "num_workers": self.num_workers,
            "matrix_a": f"{self.matrix_shape_a[0]}x{self.matrix_shape_a[1]}" if len(self.matrix_shape_a) == 2 else str(self.matrix_shape_a),
            "matrix_b": f"{self.matrix_shape_b[0]}x{self.matrix_shape_b[1]}" if len(self.matrix_shape_b) == 2 else str(self.matrix_shape_b),
            "total_time_s": round(self.total_time, 4),
            "compute_time_s": round(self.compute_time, 4),
            "communication_overhead_s": round(self.communication_overhead, 4),
            "serialize_time_s": round(self.serialize_time, 4),
            "transfer_time_s": round(self.transfer_time, 4),
            "deserialize_time_s": round(self.deserialize_time, 4),
            "gather_time_s": round(self.gather_time, 4),
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "speedup_ratio": round(self.speedup_ratio, 4),
            "worker_times_s": [round(t, 4) for t in self.worker_times],
        }
