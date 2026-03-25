"""
Web UI — FastAPI + WebSocket server with real-time metrics updates.
Opens a browser-accessible dashboard at http://localhost:<port>
"""

import asyncio
import json
import time
from typing import Optional, Set, Tuple
from pathlib import Path

from core.metrics import ComputeMetrics
from core.matrix_io import format_bytes

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Distributed Matrix Multiplication</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', monospace; background: #0d1117; color: #c9d1d9; min-height: 100vh; }
  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 32px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; color: #58a6ff; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #3fb950; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 24px 32px; }
  @media(max-width:900px){ .container { grid-template-columns: 1fr; } }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
  .card h2 { font-size: 0.85rem; text-transform: uppercase; color: #8b949e; letter-spacing: 1px; margin-bottom: 16px; }
  .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 0.9rem; }
  .metric-row:last-child { border-bottom: none; }
  .metric-label { color: #8b949e; }
  .metric-value { color: #e6edf3; font-weight: 600; font-family: monospace; }
  .metric-value.highlight { color: #58a6ff; }
  .metric-value.good { color: #3fb950; }
  .metric-value.warn { color: #d29922; }
  .progress-wrap { margin: 16px 0; }
  .progress-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 6px; display: flex; justify-content: space-between; }
  .progress-bar { height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #1f6feb, #58a6ff); border-radius: 4px; transition: width 0.3s ease; width: 0%; }
  .log-area { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 12px; height: 220px; overflow-y: auto; font-family: monospace; font-size: 0.8rem; line-height: 1.6; }
  .log-line { color: #8b949e; }
  .log-line .ts { color: #30363d; margin-right: 8px; }
  .log-line .msg { color: #c9d1d9; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
  .badge-local { background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb; }
  .badge-dist { background: #3fb95022; color: #3fb950; border: 1px solid #3fb950; }
  .badge-idle { background: #8b949e22; color: #8b949e; border: 1px solid #8b949e; }
  .worker-bars { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  .worker-bar { flex: 1; min-width: 60px; }
  .worker-bar-label { font-size: 0.7rem; color: #8b949e; text-align: center; margin-bottom: 4px; }
  .worker-bar-bg { height: 40px; background: #21262d; border-radius: 4px; overflow: hidden; position: relative; display: flex; align-items: flex-end; }
  .worker-bar-fill { width: 100%; background: #1f6feb; border-radius: 4px; transition: height 0.4s ease; }
  .worker-bar-val { position: absolute; top: 4px; width: 100%; text-align: center; font-size: 0.7rem; color: #c9d1d9; }
</style>
</head>
<body>
<header>
  <div class="status-dot" id="statusDot"></div>
  <h1>Distributed Matrix Multiplication</h1>
  <span id="modeBadge" class="badge badge-idle">IDLE</span>
</header>
<div class="container">
  <!-- Overview card -->
  <div class="card">
    <h2>Overview</h2>
    <div class="metric-row"><span class="metric-label">Mode</span><span class="metric-value highlight" id="mode">—</span></div>
    <div class="metric-row"><span class="metric-label">Backend</span><span class="metric-value" id="backend">—</span></div>
    <div class="metric-row"><span class="metric-label">Workers</span><span class="metric-value" id="workers">—</span></div>
    <div class="metric-row"><span class="metric-label">Matrix A</span><span class="metric-value" id="matA">—</span></div>
    <div class="metric-row"><span class="metric-label">Matrix B</span><span class="metric-value" id="matB">—</span></div>
    <div class="metric-row"><span class="metric-label">Status</span><span class="metric-value" id="statusText">Waiting...</span></div>
    <div class="progress-wrap" id="progressWrap" style="display:none">
      <div class="progress-label"><span id="progressLabel">Progress</span><span id="progressPct">0%</span></div>
      <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    </div>
  </div>

  <!-- Timing card -->
  <div class="card">
    <h2>Timing Metrics</h2>
    <div class="metric-row"><span class="metric-label">Total Time</span><span class="metric-value highlight" id="totalTime">—</span></div>
    <div class="metric-row"><span class="metric-label">Compute Time</span><span class="metric-value" id="computeTime">—</span></div>
    <div class="metric-row"><span class="metric-label">Gather Time</span><span class="metric-value" id="gatherTime">—</span></div>
    <div class="metric-row"><span class="metric-label">Comm. Overhead</span><span class="metric-value warn" id="commOverhead">—</span></div>
    <div class="metric-row"><span class="metric-label">  Serialize</span><span class="metric-value" id="serTime">—</span></div>
    <div class="metric-row"><span class="metric-label">  Transfer</span><span class="metric-value" id="netTime">—</span></div>
    <div class="metric-row"><span class="metric-label">  Deserialize</span><span class="metric-value" id="deTime">—</span></div>
    <div class="metric-row"><span class="metric-label">Speedup Ratio</span><span class="metric-value good" id="speedup">—</span></div>
  </div>

  <!-- Data transfer card -->
  <div class="card">
    <h2>Data Transfer</h2>
    <div class="metric-row"><span class="metric-label">Bytes Sent</span><span class="metric-value" id="bytesSent">—</span></div>
    <div class="metric-row"><span class="metric-label">Bytes Received</span><span class="metric-value" id="bytesRecv">—</span></div>
    <div id="workerSection" style="display:none">
      <h2 style="margin-top:16px;">Worker Times</h2>
      <div class="worker-bars" id="workerBars"></div>
    </div>
  </div>

  <!-- Log card -->
  <div class="card">
    <h2>Live Log</h2>
    <div class="log-area" id="logArea"></div>
  </div>
</div>

<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
let startTime = null;

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  handleMessage(data);
};

ws.onclose = () => {
  document.getElementById('statusDot').style.background = '#f85149';
  appendLog('Disconnected from server');
};

function handleMessage(data) {
  if (data.type === 'log') {
    appendLog(data.message);
  } else if (data.type === 'matrix_info') {
    document.getElementById('matA').textContent = data.shape_a;
    document.getElementById('matB').textContent = data.shape_b;
  } else if (data.type === 'mode') {
    const m = data.mode.toUpperCase();
    document.getElementById('mode').textContent = m;
    document.getElementById('backend').textContent = data.backend.toUpperCase();
    const badge = document.getElementById('modeBadge');
    badge.textContent = m;
    badge.className = 'badge ' + (data.mode === 'local' ? 'badge-local' : 'badge-dist');
    if (data.workers) document.getElementById('workers').textContent = data.workers;
    startTime = Date.now();
  } else if (data.type === 'progress') {
    const pct = Math.round(data.done / data.total * 100);
    document.getElementById('progressWrap').style.display = '';
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressPct').textContent = pct + '%';
    document.getElementById('progressLabel').textContent = data.status;
    document.getElementById('statusText').textContent = data.status;
  } else if (data.type === 'metrics') {
    updateMetrics(data.metrics);
  }
}

function updateMetrics(m) {
  const set = (id, val) => { if (document.getElementById(id)) document.getElementById(id).textContent = val; };
  set('totalTime', m.total_time_s + ' s');
  set('computeTime', m.compute_time_s + ' s');
  set('gatherTime', m.gather_time_s + ' s');
  set('commOverhead', m.communication_overhead_s + ' s');
  set('serTime', m.serialize_time_s + ' s');
  set('netTime', m.transfer_time_s + ' s');
  set('deTime', m.deserialize_time_s + ' s');
  set('speedup', m.speedup_ratio > 0 ? m.speedup_ratio + '×' : 'N/A');
  set('bytesSent', formatBytes(m.bytes_sent));
  set('bytesRecv', formatBytes(m.bytes_received));
  set('statusText', 'Completed ✓');
  document.getElementById('progressFill').style.width = '100%';
  document.getElementById('progressPct').textContent = '100%';

  if (m.worker_times_s && m.worker_times_s.length > 0) {
    document.getElementById('workerSection').style.display = '';
    const bars = document.getElementById('workerBars');
    bars.innerHTML = '';
    const max = Math.max(...m.worker_times_s, 0.001);
    m.worker_times_s.forEach((t, i) => {
      const h = Math.max(4, Math.round(t / max * 36));
      bars.innerHTML += `<div class="worker-bar">
        <div class="worker-bar-label">W${i+1}</div>
        <div class="worker-bar-bg">
          <div class="worker-bar-fill" style="height:${h}px"></div>
          <div class="worker-bar-val">${t}s</div>
        </div>
      </div>`;
    });
  }
}

function appendLog(msg) {
  const area = document.getElementById('logArea');
  const d = new Date();
  const ts = d.toTimeString().slice(0,8);
  area.innerHTML += `<div class="log-line"><span class="ts">${ts}</span><span class="msg">${msg}</span></div>`;
  area.scrollTop = area.scrollHeight;
}

function formatBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n/1024).toFixed(1) + ' KB';
  if (n < 1073741824) return (n/1048576).toFixed(1) + ' MB';
  return (n/1073741824).toFixed(1) + ' GB';
}
</script>
</body>
</html>
"""


class WebUI:
    """
    FastAPI + WebSocket server as the UI backend.
    Broadcasts log and metric events to all connected browser clients.
    """

    def __init__(self, port: int = 8080):
        if not HAS_FASTAPI:
            raise ImportError(
                "FastAPI and uvicorn are required for web UI.\n"
                "Install with: pip install fastapi uvicorn"
            )
        self.port = port
        self._connections: Set[WebSocket] = set()
        self._start_time = time.perf_counter()
        self._app = self._build_app()
        self._matrix_a = None
        self._matrix_b = None

    def _build_app(self) -> "FastAPI":
        app = FastAPI()

        @app.get("/")
        async def index():
            return HTMLResponse(HTML_TEMPLATE)

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._connections.add(websocket)
            try:
                while True:
                    await websocket.receive_text()  # keep alive
            except WebSocketDisconnect:
                self._connections.discard(websocket)

        return app

    async def _broadcast(self, data: dict):
        dead = set()
        for ws in self._connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self._connections -= dead

    def log(self, message: str):
        elapsed = time.perf_counter() - self._start_time
        print(f"  [{elapsed:7.2f}s] {message}", flush=True)
        asyncio.create_task(self._broadcast({"type": "log", "message": message}))

    def set_matrix_info(self, shape_a: tuple, shape_b: tuple):
        self._matrix_a = shape_a
        self._matrix_b = shape_b
        asyncio.create_task(self._broadcast({
            "type": "matrix_info",
            "shape_a": f"{shape_a[0]}×{shape_a[1]}",
            "shape_b": f"{shape_b[0]}×{shape_b[1]}",
        }))

    def update_progress(self, done: int, total: int, status: str = ""):
        asyncio.create_task(self._broadcast({
            "type": "progress",
            "done": done,
            "total": total,
            "status": status,
        }))

    def show_final_metrics(self, metrics: ComputeMetrics):
        asyncio.create_task(self._broadcast({
            "type": "metrics",
            "metrics": metrics.to_dict(),
        }))
        m = metrics.to_dict()
        print(f"\n  Total: {m['total_time_s']}s | Compute: {m['compute_time_s']}s | "
              f"Overhead: {m['communication_overhead_s']}s | "
              f"Sent: {format_bytes(m['bytes_sent'])} | Recv: {format_bytes(m['bytes_received'])}\n")

    def run_with_task(self, coro):
        """Start web server and run computation task concurrently."""
        import webbrowser

        async def main():
            config = uvicorn.Config(self._app, host="0.0.0.0", port=self.port, log_level="error")
            server = uvicorn.Server(config)
            print(f"\n  Web UI available at: http://localhost:{self.port}\n")
            # Open browser after short delay
            asyncio.create_task(_open_browser(self.port))
            # Run server and computation concurrently
            await asyncio.gather(
                server.serve(),
                coro,
            )

        async def _open_browser(port):
            await asyncio.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        asyncio.run(main())
