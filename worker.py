#!/usr/bin/env python3
"""
Convenience script to start a worker node.
Used on remote machines.

Usage:
  python worker.py                    # Listen on 0.0.0.0:9000
  python worker.py --port 9001        # Custom port
  python worker.py --host 0.0.0.0 --port 9000
"""

import asyncio
import argparse
import logging


def parse_args():
    p = argparse.ArgumentParser(description="Start a matrix multiplication worker node")
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=9000, help="Bind port (default: 9000)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


async def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S"
    )

    from distributed.worker_node import WorkerNode
    worker = WorkerNode(host=args.host, port=args.port)
    print(f"\n  Worker node starting on {args.host}:{args.port}")
    print("  Press Ctrl+C to stop.\n")
    await worker.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Worker stopped.")
