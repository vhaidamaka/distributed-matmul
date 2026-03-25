#!/usr/bin/env python3
"""
Distributed Matrix Multiplication System
Entry point for the application.
"""

import argparse
import sys
import asyncio
from pathlib import Path

from core.matrix_io import load_matrices, generate_random_matrices, save_matrices
from core.local_compute import local_multiply
from distributed.head_node import HeadNode
from distributed.custom_head_node import CustomHeadNode
from ui.terminal_ui import TerminalUI
from ui.web_ui import WebUI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed Matrix Multiplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local mode with input file:
  python main.py --input_file data/matrices.npz

  # Distributed mode (Ray):
  python main.py --remote_servers 192.168.1.2:9000 192.168.1.3:9000 --input_file data/matrices.npz

  # Distributed mode (custom implementation):
  python main.py --remote_servers 192.168.1.2:9000 --input_file data/matrices.npz --backend custom

  # Generate random matrices:
  python main.py --gen_random_matrixes yes --matrix_size 1000

  # Web UI:
  python main.py --input_file data/matrices.npz --ui web
        """
    )

    parser.add_argument(
        "--remote_servers",
        nargs="*",
        metavar="IP:PORT",
        default=None,
        help="Remote server addresses (IP:port). If not set, runs in local mode."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input file with matrices (.npz or .txt)"
    )
    parser.add_argument(
        "--gen_random_matrixes",
        choices=["yes", "no"],
        default="no",
        help="Generate random matrices (default: no)"
    )
    parser.add_argument(
        "--matrix_size",
        type=int,
        default=500,
        help="Size for random matrix generation (NxN, default: 500)"
    )
    parser.add_argument(
        "--matrix_size_a",
        type=str,
        default=None,
        help="Size for matrix A as ROWSxCOLS (e.g. 500x800)"
    )
    parser.add_argument(
        "--matrix_size_b",
        type=str,
        default=None,
        help="Size for matrix B as ROWSxCOLS (e.g. 800x600)"
    )
    parser.add_argument(
        "--backend",
        choices=["ray", "custom"],
        default="ray",
        help="Distributed backend: 'ray' (default) or 'custom' (own implementation)"
    )
    parser.add_argument(
        "--ui",
        choices=["terminal", "web"],
        default="terminal",
        help="UI mode: 'terminal' (default) or 'web'"
    )
    parser.add_argument(
        "--web_port",
        type=int,
        default=8080,
        help="Port for web UI (default: 8080)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save result matrix"
    )
    parser.add_argument(
        "--mode",
        choices=["worker"],
        default=None,
        help="Run as worker node (used internally)"
    )
    parser.add_argument(
        "--worker_port",
        type=int,
        default=9000,
        help="Port to listen on when running as worker (default: 9000)"
    )

    return parser.parse_args()


async def run_worker(port: int):
    """Run this process as a worker node."""
    from distributed.worker_node import WorkerNode
    worker = WorkerNode(port=port)
    print(f"[Worker] Starting on port {port}")
    await worker.start()


async def run_computation(args, ui):
    """Main computation logic."""
    import numpy as np

    # === Load or generate matrices ===
    if args.gen_random_matrixes == "yes":
        if args.matrix_size_a and args.matrix_size_b:
            rows_a, cols_a = map(int, args.matrix_size_a.split("x"))
            rows_b, cols_b = map(int, args.matrix_size_b.split("x"))
            if cols_a != rows_b:
                print(f"[Error] Matrix dimensions incompatible: A({rows_a}x{cols_a}) x B({rows_b}x{cols_b})")
                sys.exit(1)
            A, B = generate_random_matrices(rows_a, cols_a, cols_b)
        else:
            n = args.matrix_size
            A, B = generate_random_matrices(n, n, n)
        ui.log(f"Generated random matrices: A={A.shape}, B={B.shape}")
    elif args.input_file:
        path = Path(args.input_file)
        if not path.exists():
            print(f"[Error] Input file not found: {args.input_file}")
            sys.exit(1)
        A, B = load_matrices(args.input_file)
        ui.log(f"Loaded matrices from {args.input_file}: A={A.shape}, B={B.shape}")
    else:
        print("[Error] Either --input_file or --gen_random_matrixes yes must be specified.")
        sys.exit(1)

    ui.set_matrix_info(A.shape, B.shape)

    # === Choose mode ===
    if not args.remote_servers:
        # Local mode
        ui.log("Mode: LOCAL")
        result, metrics = await local_multiply(A, B, ui)
    else:
        # Distributed mode
        servers = args.remote_servers
        ui.log(f"Mode: DISTRIBUTED ({args.backend.upper()}) — {len(servers)} worker(s): {servers}")
        if args.backend == "ray":
            head = HeadNode(servers=servers, ui=ui)
        else:
            head = CustomHeadNode(servers=servers, ui=ui)
        result, metrics = await head.multiply(A, B)

    # === Output ===
    ui.show_final_metrics(metrics)

    if args.output_file:
        save_matrices(args.output_file, result=result)
        ui.log(f"Result saved to {args.output_file}")

    return result, metrics


def main():
    args = parse_args()

    # Worker mode — just run the worker and exit
    if args.mode == "worker":
        asyncio.run(run_worker(args.worker_port))
        return

    # Choose UI
    if args.ui == "web":
        ui = WebUI(port=args.web_port)
    else:
        ui = TerminalUI()

    # Run
    if args.ui == "web":
        # Web UI starts its own event loop with background task
        ui.run_with_task(run_computation(args, ui))
    else:
        asyncio.run(run_computation(args, ui))


if __name__ == "__main__":
    main()
