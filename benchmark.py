#!/usr/bin/env python3
"""
Benchmark script — runs all scenarios from the spec and prints a comparison table.

Scenarios:
  1. Local (baseline)
  2. Distributed High Bandwidth (simulated — skipped if no workers)
  3. Distributed Low Bandwidth
  4. Distributed Unstable Network
  5. Distributed Worst Case

Usage:
  python benchmark.py --sizes 200 500 1000 2000
  python benchmark.py --sizes 500 1000 --remote_servers 192.168.1.2:9000 --interface eth0
"""

import argparse
import asyncio
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from core.matrix_io import generate_random_matrices, format_bytes
from core.local_compute import local_multiply
from core.metrics import ComputeMetrics
from ui.terminal_ui import TerminalUI
from network.simulator import NetworkSimulator, PROFILES


SCENARIOS = [
    {"name": "Local (baseline)",            "mode": "local",   "net_profile": None},
    {"name": "Distributed — High BW",       "mode": "dist",    "net_profile": "high_bandwidth"},
    {"name": "Distributed — Low BW (1Mbps)","mode": "dist",    "net_profile": "low_bandwidth"},
    {"name": "Distributed — Unstable",      "mode": "dist",    "net_profile": "unstable"},
    {"name": "Distributed — Worst Case",    "mode": "dist",    "net_profile": "worst_case"},
]


def parse_args():
    p = argparse.ArgumentParser(description="Matrix Multiplication Benchmark")
    p.add_argument("--sizes", nargs="+", type=int, default=[200, 500, 1000, 2000],
                   help="Matrix sizes to test (NxN)")
    p.add_argument("--remote_servers", nargs="*", default=None,
                   help="Worker addresses for distributed tests")
    p.add_argument("--backend", choices=["ray", "custom"], default="ray")
    p.add_argument("--interface", default="eth0",
                   help="Network interface for tc simulation")
    p.add_argument("--output", default=None,
                   help="Save results to JSON/CSV file")
    p.add_argument("--no_sim", action="store_true",
                   help="Skip network simulation (tc)")
    return p.parse_args()


async def run_local(A: np.ndarray, B: np.ndarray, ui: TerminalUI) -> ComputeMetrics:
    _, metrics = await local_multiply(A, B, ui)
    return metrics


async def run_distributed(
    A: np.ndarray,
    B: np.ndarray,
    servers: List[str],
    backend: str,
    ui: TerminalUI
) -> Optional[ComputeMetrics]:
    if not servers:
        return None
    if backend == "ray":
        from distributed.head_node import HeadNode
        head = HeadNode(servers=servers, ui=ui)
    else:
        from distributed.custom_head_node import CustomHeadNode
        head = CustomHeadNode(servers=servers, ui=ui)
    try:
        _, metrics = await head.multiply(A, B)
        return metrics
    except Exception as e:
        ui.log(f"[Benchmark] Distributed run failed: {e}")
        return None


def print_table(results: List[Dict[str, Any]]):
    """Print results as a comparison table."""
    print("\n" + "═" * 100)
    print(f"{'BENCHMARK RESULTS':^100}")
    print("═" * 100)

    header = f"{'Scenario':<35} {'Size':>8} {'Total(s)':>10} {'Compute(s)':>12} {'Overhead(s)':>13} {'Sent':>10} {'Speedup':>8}"
    print(header)
    print("─" * 100)

    baseline: Dict[int, float] = {}  # size → local time

    for r in results:
        if r["mode"] == "local":
            baseline[r["size"]] = r["total_s"]

    for r in results:
        sp = ""
        if r["mode"] != "local" and r["size"] in baseline and r["total_s"] > 0:
            ratio = baseline[r["size"]] / r["total_s"]
            sp = f"{ratio:.2f}×"
        elif r["mode"] == "local":
            sp = "baseline"

        status = r.get("status", "ok")
        line = (
            f"{r['scenario']:<35}"
            f" {r['size']:>5}×{r['size']:<2}"
            f" {r['total_s']:>10.4f}"
            f" {r['compute_s']:>12.4f}"
            f" {r['overhead_s']:>13.4f}"
            f" {r['sent']:>10}"
            f" {sp:>8}"
        )
        if status != "ok":
            line += f"  ⚠ {status}"
        print(line)

    print("═" * 100 + "\n")


def save_results(results: List[Dict[str, Any]], path: str):
    p = Path(path)
    if p.suffix == ".json":
        with open(p, "w") as f:
            json.dump(results, f, indent=2)
    elif p.suffix == ".csv":
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"[Benchmark] Results saved to {p}")


async def main():
    args = parse_args()
    ui = TerminalUI()
    sim = NetworkSimulator(interface=args.interface)
    all_results: List[Dict[str, Any]] = []

    print("\n" + "═" * 60)
    print("   Matrix Multiplication Benchmark")
    print(f"   Sizes: {args.sizes}")
    print(f"   Workers: {args.remote_servers or 'none (local only)'}")
    print(f"   Network sim: {'disabled' if args.no_sim else 'enabled'}")
    print("═" * 60 + "\n")

    for size in args.sizes:
        A, B = generate_random_matrices(size, size, size)

        # ---- Local ----
        print(f"[{size}×{size}] Running local...")
        t0 = time.perf_counter()
        metrics = await run_local(A, B, ui)
        all_results.append({
            "scenario": "Local (baseline)",
            "mode": "local",
            "size": size,
            "total_s": round(metrics.total_time, 4),
            "compute_s": round(metrics.compute_time, 4),
            "overhead_s": round(metrics.communication_overhead, 4),
            "sent": format_bytes(metrics.bytes_sent),
            "status": "ok",
        })
        ui.log(f"[{size}×{size}] Local: {metrics.total_time:.4f}s")

        if not args.remote_servers:
            ui.log(f"[{size}×{size}] No remote servers — skipping distributed scenarios")
            continue

        # ---- Distributed scenarios ----
        for scenario in SCENARIOS[1:]:  # skip local
            profile_name = scenario["net_profile"]
            sc_name = scenario["name"]

            if not args.no_sim and sim.available and profile_name:
                profile = PROFILES[profile_name]
                sim.apply(profile.bandwidth_mbps, profile.loss_percent, profile.delay_ms, profile.delay_jitter_ms)
                ui.log(f"[{size}×{size}] {sc_name} | Network: {profile.describe()}")
            else:
                ui.log(f"[{size}×{size}] {sc_name} | (no network sim)")

            dist_metrics = await run_distributed(A, B, args.remote_servers, args.backend, ui)

            if not args.no_sim and sim.available:
                sim.clear(quiet=True)

            if dist_metrics is None:
                all_results.append({
                    "scenario": sc_name,
                    "mode": "dist",
                    "size": size,
                    "total_s": 0.0,
                    "compute_s": 0.0,
                    "overhead_s": 0.0,
                    "sent": "—",
                    "status": "failed",
                })
            else:
                all_results.append({
                    "scenario": sc_name,
                    "mode": "dist",
                    "size": size,
                    "total_s": round(dist_metrics.total_time, 4),
                    "compute_s": round(dist_metrics.compute_time, 4),
                    "overhead_s": round(dist_metrics.communication_overhead, 4),
                    "sent": format_bytes(dist_metrics.bytes_sent),
                    "status": "ok",
                })

    print_table(all_results)

    if args.output:
        save_results(all_results, args.output)

    return all_results


if __name__ == "__main__":
    asyncio.run(main())
