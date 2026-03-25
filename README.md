# Distributed Matrix Multiplication

Python system for computing `C = A × B` in **local mode** (all CPU cores on one machine)
and **distributed mode** (custom TCP protocol or Ray across multiple nodes),
with real-time terminal and web UI, benchmarking, and network simulation.

---

## Project Structure

```
matmul_distributed/
├── main.py                  # Entry point (head node / client)
├── worker.py                # Worker node launcher
├── benchmark.py             # Full benchmark suite (all scenarios)
├── gen_data.py              # Generate & save test matrix files
│
├── core/
│   ├── local_compute.py     # Local parallel multiply (ProcessPoolExecutor)
│   ├── matrix_io.py         # Load / save / generate matrices (.npz, .txt)
│   └── metrics.py           # ComputeMetrics dataclass
│
├── distributed/
│   ├── head_node.py         # Ray-based head node
│   ├── custom_head_node.py  # Own TCP head node (no Ray)
│   └── worker_node.py       # Async TCP worker node + binary protocol
│
├── network/
│   └── simulator.py         # tc / netem wrapper (Linux only)
│
├── ui/
│   ├── terminal_ui.py       # Live terminal output with \r progress
│   └── web_ui.py            # FastAPI + WebSocket dashboard
│
├── tests/
│   ├── test_core.py         # 26 unit tests — matrix I/O, metrics, local compute
│   └── test_distributed.py  # 14 tests — serialisation, protocol, live TCP worker
│
├── data/
│   └── sample_3x3_x_3x2.txt
│
├── requirements.txt
└── pytest.ini
```

---

## Installation

```bash
# Python 3.12+ required
pip install -r requirements.txt

# For Ray backend (optional):
pip install ray

# For web UI (optional):
pip install fastapi uvicorn
```

---

## Quick Start

### Local mode

```bash
# Generate a 500×500 random matrix pair and multiply locally
python main.py --gen_random_matrixes yes --matrix_size 500

# With a custom rectangular pair
python main.py --gen_random_matrixes yes --matrix_size_a 800x600 --matrix_size_b 600x400

# From a saved file
python main.py --input_file data/matrices.npz

# Save result to file
python main.py --gen_random_matrixes yes --matrix_size 1000 --output_file data/result.npz
```

### Distributed mode — custom TCP backend

**Step 1 — start workers on each remote machine:**
```bash
# On worker machine 1 (e.g. 192.168.1.2):
python worker.py --port 9000

# On worker machine 2 (e.g. 192.168.1.3):
python worker.py --port 9000
```

**Step 2 — run from head node:**
```bash
python main.py \
  --remote_servers 192.168.1.2:9000 192.168.1.3:9000 \
  --gen_random_matrixes yes \
  --matrix_size 2000 \
  --backend custom
```

### Distributed mode — Ray backend

```bash
# Start Ray on each machine first:
#   Head:   ray start --head --port=6379
#   Worker: ray start --address=<HEAD_IP>:6379

python main.py \
  --remote_servers 192.168.1.2:6379 \
  --gen_random_matrixes yes \
  --matrix_size 2000 \
  --backend ray
```

### Web UI

```bash
python main.py --gen_random_matrixes yes --matrix_size 1000 --ui web --web_port 8080
# Browser opens automatically at http://localhost:8080
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--remote_servers IP:PORT …` | *(none)* | Worker addresses. Omit for local mode. |
| `--input_file PATH` | *(none)* | `.npz` or `.txt` file with matrices A and B |
| `--gen_random_matrixes yes\|no` | `no` | Generate random matrices |
| `--matrix_size N` | `500` | Square matrix size for random generation |
| `--matrix_size_a ROWSxCOLS` | *(none)* | Matrix A dimensions, e.g. `800x600` |
| `--matrix_size_b ROWSxCOLS` | *(none)* | Matrix B dimensions, e.g. `600x400` |
| `--backend ray\|custom` | `ray` | Distributed backend |
| `--ui terminal\|web` | `terminal` | Output UI |
| `--web_port PORT` | `8080` | Port for web UI |
| `--output_file PATH` | *(none)* | Save result matrix to this path |

---

## Matrix File Formats

### `.npz` (recommended)
```python
import numpy as np
np.savez("matrices.npz", A=A, B=B)
```

### `.txt`
Two matrices separated by a blank line, space-delimited:
```
1.0 2.0 3.0
4.0 5.0 6.0

9.0 8.0
7.0 6.0
5.0 4.0
```

### Generate files with gen_data.py
```bash
python gen_data.py --size 1000 --out data/m1000.npz
python gen_data.py --rows_a 500 --cols_a 800 --cols_b 300 --out data/rect.npz
python gen_data.py --size 200 --out data/m200.txt --format txt
```

---

## Distributed Algorithm

```
Matrix A (M×K)  split into N horizontal blocks: A_0, A_1, … A_{N-1}
Matrix B (K×N)  sent in full to every worker

Worker i computes:  C_i = A_i × B

Head node assembles:  C = vstack(C_0, C_1, … C_{N-1})
```

- **N** = number of available worker nodes
- Workers run concurrently; head gathers via `asyncio.gather`
- Order is preserved by index — no sorting needed

---

## Custom TCP Protocol

The custom backend uses a minimal binary framing protocol over asyncio TCP:

```
┌──────────────┬──────────────────────┬─────────────────────┐
│  msg_type    │  payload_length      │  payload            │
│  (4 bytes)   │  (8 bytes, big-end.) │  (N bytes, msgpack) │
└──────────────┴──────────────────────┴─────────────────────┘
```

| Type | Direction | Meaning |
|------|-----------|---------|
| `MSG_TASK (1)` | Head → Worker | Matrix block to multiply |
| `MSG_RESULT (2)` | Worker → Head | Computed result block |
| `MSG_PING (3)` | Head → Worker | Health check |
| `MSG_PONG (4)` | Worker → Head | Alive acknowledgement |
| `MSG_ERROR (5)` | Either | Error notification |

Payload is **msgpack-serialized** (falls back to `pickle` if msgpack is unavailable).
NumPy arrays are transmitted as raw bytes with shape and dtype metadata.

---

## Benchmarking

```bash
# Local only (no workers needed):
python benchmark.py --sizes 200 500 1000 2000

# With distributed workers:
python benchmark.py \
  --sizes 500 1000 2000 5000 \
  --remote_servers 192.168.1.2:9000 192.168.1.3:9000 \
  --backend custom \
  --interface eth0 \
  --output results.json
```

### Test Scenarios

| Scenario | Network Config | Expected |
|---|---|---|
| **Local (baseline)** | N/A | Best single-node performance |
| **Distributed — High BW** | 100 Mbps, 0% loss | Speedup for large matrices |
| **Distributed — Low BW** | 1 Mbps, 0% loss | Slower — transfer dominates |
| **Distributed — Unstable** | 10 Mbps, 7.5% loss | Risk of failure / retries |
| **Distributed — Worst Case** | 0.5 Mbps, 10% loss | Likely slower than local |

---

## Network Simulation

Uses Linux `tc` (Traffic Control) + `netem`. Requires `iproute2` and usually `sudo`:

```bash
# Apply manually:
sudo tc qdisc add dev eth0 root tbf rate 1mbit burst 10kb latency 50ms
sudo tc qdisc add dev eth0 parent 1:1 handle 10: netem loss 5%

# Remove:
sudo tc qdisc del dev eth0 root
```

Or use the Python wrapper:
```python
from network.simulator import NetworkSimulator, PROFILES
sim = NetworkSimulator(interface="eth0")
sim.apply_profile("low_bandwidth")   # 1 Mbps
# ... run benchmark ...
sim.clear()
```

Available profiles: `high_bandwidth`, `low_bandwidth`, `very_low_bandwidth`,
`unstable`, `worst_case`.

---

## Metrics

| Metric | Description |
|---|---|
| **Total Time** | Wall-clock time from start to final matrix assembled |
| **Compute Time** | Pure `A_block @ B` time (summed across workers) |
| **Gather Time** | Time to `vstack` partial results on head |
| **Comm. Overhead** | Serialize + Transfer + Deserialize time |
| **Bytes Sent** | Total bytes dispatched to workers |
| **Bytes Received** | Total bytes received from workers |
| **Speedup Ratio** | `T_local / T_distributed` (set baseline first) |
| **Worker Times** | Individual compute time per worker |

---

## Running Tests

```bash
# All tests:
pytest

# Core only (fast, no sockets):
pytest tests/test_core.py -v

# Distributed (opens real TCP sockets on localhost):
pytest tests/test_distributed.py -v

# With coverage:
pip install pytest-cov
pytest --cov=. --cov-report=term-missing
```

**26 core tests** · **14 distributed tests** · all green ✓

---

## Expected Performance

| Matrix Size | Local (2 cores) | Distributed 2-node (LAN) |
|---|---|---|
| 200 × 200 | ~0.01 s | ~0.05 s (overhead dominates) |
| 500 × 500 | ~0.05 s | ~0.10 s |
| 1000 × 1000 | ~0.3 s | ~0.2 s (starts to pay off) |
| 2000 × 2000 | ~2 s | ~1.1 s |
| 5000 × 5000 | ~30 s | ~16 s |
| 10000 × 10000 | ~240 s | ~125 s |

> Distribution only pays off when compute time >> communication overhead.
> For small matrices (<1000×1000) on a fast LAN, local mode is typically faster.

---

## Dependencies

| Package | Use | Required |
|---|---|---|
| `numpy` | Matrix operations | **Yes** |
| `msgpack` | Fast binary serialization | No (falls back to pickle) |
| `ray` | Ray distributed backend | No (only for `--backend ray`) |
| `fastapi` | Web UI server | No (only for `--ui web`) |
| `uvicorn` | ASGI server for web UI | No (only for `--ui web`) |
| `pytest` | Testing | Dev only |
| `pytest-asyncio` | Async test support | Dev only |
