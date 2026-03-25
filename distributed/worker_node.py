"""
Custom worker node — listens for matrix blocks, computes, and returns results.
Uses asyncio TCP sockets with a simple binary protocol.

Protocol (per message):
  [4 bytes: message type]
  [8 bytes: payload length]
  [N bytes: payload (msgpack-serialized dict)]
"""

import asyncio
import time
import struct
import logging
import numpy as np
from typing import Optional

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

logger = logging.getLogger(__name__)


# Message types
MSG_TASK = 1       # Head → Worker: here's your matrix block
MSG_RESULT = 2     # Worker → Head: here's my result
MSG_PING = 3       # Head → Worker: are you alive?
MSG_PONG = 4       # Worker → Head: yes
MSG_ERROR = 5      # Either direction: error


def encode_message(msg_type: int, payload: bytes) -> bytes:
    """Pack header + payload."""
    header = struct.pack(">IQ", msg_type, len(payload))
    return header + payload


async def read_message(reader: asyncio.StreamReader):
    """Read one message from stream. Returns (msg_type, payload_bytes)."""
    header = await reader.readexactly(12)  # 4 + 8
    msg_type, length = struct.unpack(">IQ", header)
    payload = await reader.readexactly(length)
    return msg_type, payload


def serialize_arrays(**kwargs) -> bytes:
    """Serialize numpy arrays + metadata to bytes."""
    if HAS_MSGPACK:
        data = {}
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray):
                data[key + "_bytes"] = val.tobytes()
                data[key + "_shape"] = list(val.shape)
                data[key + "_dtype"] = val.dtype.str
            else:
                data[key] = val
        return msgpack.packb(data)
    else:
        import pickle
        return pickle.dumps(kwargs)


def deserialize_arrays(payload: bytes) -> dict:
    """Deserialize bytes back to dict with numpy arrays."""
    if HAS_MSGPACK:
        raw = msgpack.unpackb(payload, raw=False)
        result = {}
        # Collect numpy keys
        array_keys = set()
        for k in raw:
            if k.endswith("_bytes"):
                array_keys.add(k[:-6])
        for key in array_keys:
            arr_bytes = raw[key + "_bytes"]
            shape = tuple(raw[key + "_shape"])
            dtype = raw[key + "_dtype"]
            result[key] = np.frombuffer(arr_bytes, dtype=np.dtype(dtype)).reshape(shape)
        # Non-array keys
        for k, v in raw.items():
            if not (k.endswith("_bytes") or k.endswith("_shape") or k.endswith("_dtype")):
                result[k] = v
        return result
    else:
        import pickle
        return pickle.loads(payload)


class WorkerNode:
    """
    Async TCP worker that accepts tasks from the head node.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self):
        """Start listening for incoming connections."""
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        addr = self._server.sockets[0].getsockname()
        logger.info(f"[Worker] Listening on {addr[0]}:{addr[1]}")
        print(f"[Worker] Ready on {addr[0]}:{addr[1]}")
        async with self._server:
            await self._server.serve_forever()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        logger.info(f"[Worker] Connection from {peer}")

        try:
            while True:
                msg_type, payload = await read_message(reader)

                if msg_type == MSG_PING:
                    resp = encode_message(MSG_PONG, b"")
                    writer.write(resp)
                    await writer.drain()

                elif msg_type == MSG_TASK:
                    data = deserialize_arrays(payload)
                    A_block: np.ndarray = data["A_block"]
                    B: np.ndarray = data["B"]
                    task_id: int = data.get("task_id", 0)

                    logger.info(f"[Worker] Task {task_id}: A_block{A_block.shape} x B{B.shape}")

                    t0 = time.perf_counter()
                    result = A_block @ B
                    elapsed = time.perf_counter() - t0

                    resp_payload = serialize_arrays(result=result, task_id=task_id, elapsed=elapsed)
                    resp = encode_message(MSG_RESULT, resp_payload)
                    writer.write(resp)
                    await writer.drain()

                    logger.info(f"[Worker] Task {task_id} done in {elapsed:.4f}s")

                elif msg_type == MSG_ERROR:
                    logger.warning(f"[Worker] Received error from head. Closing.")
                    break

                else:
                    logger.warning(f"[Worker] Unknown message type {msg_type}")

        except asyncio.IncompleteReadError:
            logger.info(f"[Worker] Connection closed by {peer}")
        except Exception as e:
            logger.error(f"[Worker] Error: {e}", exc_info=True)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
