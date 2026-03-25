#!/usr/bin/env python3
"""
Helper to generate test matrix files.

Usage:
  python gen_data.py --size 500 --out data/matrices_500.npz
  python gen_data.py --rows_a 800 --cols_a 600 --cols_b 400 --out data/custom.npz
  python gen_data.py --size 200 --out data/matrices_200.txt --format txt
"""

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Generate test matrix files")
    p.add_argument("--size", type=int, default=500, help="Square matrix size (NxN)")
    p.add_argument("--rows_a", type=int, default=None)
    p.add_argument("--cols_a", type=int, default=None)
    p.add_argument("--cols_b", type=int, default=None)
    p.add_argument("--out", type=str, default="data/matrices.npz")
    p.add_argument("--format", choices=["npz", "txt"], default="npz")
    args = p.parse_args()

    from core.matrix_io import generate_random_matrices, save_matrices

    rows_a = args.rows_a or args.size
    cols_a = args.cols_a or args.size
    cols_b = args.cols_b or args.size

    A, B = generate_random_matrices(rows_a, cols_a, cols_b)

    out = Path(args.out)
    if args.format == "txt" and not out.suffix:
        out = out.with_suffix(".txt")
    elif args.format == "npz" and not out.suffix:
        out = out.with_suffix(".npz")

    out.parent.mkdir(parents=True, exist_ok=True)
    save_matrices(str(out), A=A, B=B)

    size_mb = (A.nbytes + B.nbytes) / 1024 / 1024
    print(f"Generated: A{A.shape} × B{B.shape}")
    print(f"Saved to:  {out}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
