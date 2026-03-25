"""
Terminal UI — real-time metrics display with self-updating lines (\r).
"""

import sys
import time
import shutil
from typing import Optional, Tuple
from core.metrics import ComputeMetrics
from core.matrix_io import format_bytes


class TerminalUI:
    """
    Displays progress and metrics in the terminal.
    Uses \\r to update in-place for progress, prints log lines normally.
    """

    def __init__(self):
        self._start_time = time.perf_counter()
        self._matrix_a: Optional[Tuple] = None
        self._matrix_b: Optional[Tuple] = None
        self._progress_active = False
        self._width = shutil.get_terminal_size((80, 24)).columns

    def set_matrix_info(self, shape_a: tuple, shape_b: tuple):
        self._matrix_a = shape_a
        self._matrix_b = shape_b

    def log(self, message: str):
        """Print a log line (new line)."""
        if self._progress_active:
            print()  # end the current progress line
            self._progress_active = False
        elapsed = time.perf_counter() - self._start_time
        print(f"  [{elapsed:7.2f}s] {message}", flush=True)

    def update_progress(self, done: int, total: int, status: str = ""):
        """Update progress in-place using \\r."""
        if total == 0:
            return

        pct = done / total
        bar_width = 30
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        elapsed = time.perf_counter() - self._start_time

        line = f"  [{elapsed:7.2f}s] [{bar}] {done}/{total}  {status}"
        # Truncate to terminal width
        if len(line) > self._width - 1:
            line = line[:self._width - 4] + "..."

        print(f"\r{line}", end="", flush=True)
        self._progress_active = True

        if done >= total:
            print()  # newline when complete
            self._progress_active = False

    def show_final_metrics(self, metrics: ComputeMetrics):
        """Print a formatted summary table."""
        if self._progress_active:
            print()
            self._progress_active = False

        m = metrics.to_dict()
        width = min(self._width, 70)
        sep = "─" * width

        print(f"\n  {'═' * width}")
        print(f"  {'RESULTS':^{width}}")
        print(f"  {'═' * width}")

        rows = [
            ("Mode",              m["mode"].upper()),
            ("Backend",          m["backend"].upper()),
            ("Workers",          str(m["num_workers"])),
            ("Matrix A",         m["matrix_a"]),
            ("Matrix B",         m["matrix_b"]),
            (sep, ""),
            ("Total Time",       f"{m['total_time_s']:.4f} s"),
            ("Compute Time",     f"{m['compute_time_s']:.4f} s"),
            ("Gather Time",      f"{m['gather_time_s']:.4f} s"),
            (sep, ""),
            ("Comm. Overhead",   f"{m['communication_overhead_s']:.4f} s"),
            ("  Serialize",      f"{m['serialize_time_s']:.4f} s"),
            ("  Transfer",       f"{m['transfer_time_s']:.4f} s"),
            ("  Deserialize",    f"{m['deserialize_time_s']:.4f} s"),
            (sep, ""),
            ("Data Sent",        format_bytes(m["bytes_sent"])),
            ("Data Received",    format_bytes(m["bytes_received"])),
            (sep, ""),
            ("Speedup Ratio",    f"{m['speedup_ratio']:.4f}×" if m["speedup_ratio"] > 0 else "N/A"),
        ]

        if m.get("worker_times_s"):
            rows.append((sep, ""))
            for i, wt in enumerate(m["worker_times_s"]):
                rows.append((f"  Worker {i+1} Time", f"{wt:.4f} s"))

        for label, value in rows:
            if label == sep:
                print(f"  {sep}")
            else:
                print(f"  {label:<25} {value}")

        print(f"  {'═' * width}\n")

    def print_header(self):
        """Print application header."""
        print("\n" + "=" * 60)
        print("   Distributed Matrix Multiplication System")
        print("=" * 60 + "\n")
