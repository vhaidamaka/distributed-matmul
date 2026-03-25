"""
Matrix I/O utilities — load, save, and generate matrices.
"""

import numpy as np
from pathlib import Path
from typing import Tuple


def generate_random_matrices(rows_a: int, cols_a: int, cols_b: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random float64 matrices A (rows_a x cols_a) and B (cols_a x cols_b)."""
    rng = np.random.default_rng(42)
    A = rng.random((rows_a, cols_a), dtype=np.float64)
    B = rng.random((cols_a, cols_b), dtype=np.float64)
    return A, B


def load_matrices(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load matrices A and B from file.
    Supported formats:
      .npz  — NumPy archive with keys 'A' and 'B'
      .txt  — two matrices separated by blank line (space-delimited rows)
      .npy  — single file with shape (2, N, M) stacked
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".npz":
        data = np.load(p)
        A = data["A"]
        B = data["B"]
    elif ext == ".npy":
        data = np.load(p)
        if data.ndim == 3 and data.shape[0] == 2:
            A, B = data[0], data[1]
        else:
            raise ValueError("For .npy format, file must have shape (2, rows, cols)")
    elif ext == ".txt":
        A, B = _load_txt(p)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .npz, .npy, or .txt")

    # Validate shapes
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Matrices must be 2D. Got A.ndim={A.ndim}, B.ndim={B.ndim}")
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Incompatible shapes for multiplication: A{A.shape} x B{B.shape}"
        )

    return A.astype(np.float64), B.astype(np.float64)


def _load_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a text file with two matrices separated by a blank line."""
    text = path.read_text()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if len(blocks) < 2:
        raise ValueError("Text file must contain two matrices separated by a blank line")

    def parse_block(block: str) -> np.ndarray:
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split()])
        return np.array(rows, dtype=np.float64)

    return parse_block(blocks[0]), parse_block(blocks[1])


def save_matrices(path: str, *, result: np.ndarray = None, A: np.ndarray = None, B: np.ndarray = None):
    """
    Save matrices to file.
    If 'result' is provided, saves the result matrix.
    If 'A' and 'B' are provided, saves them as input pair.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    if result is not None:
        if ext == ".npz":
            np.savez(p, result=result)
        elif ext == ".npy":
            np.save(p, result)
        elif ext == ".txt":
            np.savetxt(p, result, fmt="%.6f")
        else:
            np.savez(p, result=result)

    elif A is not None and B is not None:
        if ext == ".npz":
            np.savez(p, A=A, B=B)
        elif ext == ".txt":
            with open(p, "w") as f:
                for row in A:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
                f.write("\n")
                for row in B:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
        else:
            np.savez(p, A=A, B=B)
    else:
        raise ValueError("Provide either 'result' or both 'A' and 'B'")


def matrix_size_bytes(m: np.ndarray) -> int:
    """Return size of matrix in bytes."""
    return m.nbytes


def format_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
