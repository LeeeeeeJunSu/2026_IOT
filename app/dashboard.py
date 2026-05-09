from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .core import FingerprintEngine
from .location_monitor import LocationMonitorConfig, StableLocationMonitor


DEFAULT_DASHBOARD_HOST = "0.0.0.0"
DEFAULT_DASHBOARD_PORT = 8000
MAX_PORT_ATTEMPTS = 10


class LocationDashboardThread(threading.Thread):
    def __init__(
        self,
        engine: FingerprintEngine,
        *,
        host: str = DEFAULT_DASHBOARD_HOST,
        port: int = DEFAULT_DASHBOARD_PORT,
        monitor_config: LocationMonitorConfig | None = None,
        led_controller: Any | None = None,
    ) -> None:
        super().__init__(name="location-dashboard-http", daemon=True)
        self.engine = engine
        self.host = host
        self.port = int(port)
        self.monitor = StableLocationMonitor(monitor_config)
        self.led_controller = led_controller
        self.httpd: _DashboardHTTPServer | None = None
        self.ready_event = threading.Event()
        self.error: str | None = None
        self.actual_port: int | None = None

    @property
    def local_url(self) -> str:
        port = self.actual_port or self.port
        return f"http://127.0.0.1:{port}"

    @property
    def lan_url_hint(self) -> str:
        port = self.actual_port or self.port
        return f"http://{socket.gethostname()}.local:{port}"

    def run(self) -> None:
        for offset in range(MAX_PORT_ATTEMPTS):
            candidate_port = self.port + offset
            try:
                self.httpd = _DashboardHTTPServer(
                    (self.host, candidate_port),
                    _DashboardHandler,
                    engine=self.engine,
                    monitor=self.monitor,
                    led_controller=self.led_controller,
                )
            except OSError as exc:
                self.error = str(exc)
                continue

            self.actual_port = candidate_port
            self.ready_event.set()
            self.httpd.serve_forever(poll_interval=0.25)
            return
        self.ready_event.set()

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


class _DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        engine: FingerprintEngine,
        monitor: StableLocationMonitor,
        led_controller: Any | None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.engine = engine
        self.monitor = monitor
        self.led_controller = led_controller


class _DashboardHandler(BaseHTTPRequestHandler):
    server: _DashboardHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"", "/"}:
            self._send_html(_INDEX_HTML)
            return
        if path == "/api/location":
            self._send_json(self._location_payload())
            return
        if path == "/api/snapshot":
            self._send_json(self.server.engine.snapshot())
            return
        if path == "/api/leds":
            self._send_json(self._led_payload())
            return
        self.send_error(404, "Not found")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/leds":
            self.send_error(404, "Not found")
            return
        controller = self.server.led_controller
        if controller is None:
            self._send_json(
                {"enabled": False, "error": "LED controller is disabled."},
                status=503,
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(max(0, min(length, 4096)))
            body = json.loads(raw_body.decode("utf-8") or "{}")
            mode = str(body.get("mode", "auto"))
            if body.get("cell_key") == "*":
                payload = controller.set_all_mode(mode)
            else:
                payload = controller.set_mode(str(body.get("cell_key", "")), mode)
        except Exception as exc:
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}", "leds": self._led_payload()},
                status=400,
            )
            return
        self._send_json(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _location_payload(self) -> dict[str, Any]:
        snapshot = self.server.engine.snapshot()
        payload = self.server.monitor.update(snapshot)
        payload["leds"] = self._led_payload()
        return payload

    def _led_payload(self) -> dict[str, Any]:
        controller = self.server.led_controller
        if controller is None:
            return {
                "enabled": False,
                "auto_enabled": False,
                "backend": "disabled",
                "hardware_ready": False,
                "active_cell_key": None,
                "message": "LED controller is disabled.",
                "zones": [],
            }
        return controller.snapshot()

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_dashboard_for_engine(engine: FingerprintEngine) -> LocationDashboardThread:
    return LocationDashboardThread(engine)


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CSI Location Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #151d2b;
      --muted: #667386;
      --soft: #eef2f6;
      --line: #d6dde8;
      --blue: #2563eb;
      --teal: #0f9f7b;
      --amber: #c77700;
      --red: #b42318;
      --coral: #e6574f;
      --shadow: 0 18px 50px rgba(39, 55, 77, 0.12);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(180deg, #eef3f9 0%, #f9fbfd 54%, #f5f7fa 100%);
    }
    main {
      width: min(1280px, 100%);
      margin: 0 auto;
      padding: 18px;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 21px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 30px;
      border-radius: 999px;
      padding: 4px 11px;
      font-size: 13px;
      font-weight: 800;
      background: #e8eef7;
      color: var(--ink);
      text-transform: uppercase;
      letter-spacing: 0;
      white-space: nowrap;
    }
    .badge.good { background: #dcf7ef; color: #08745a; }
    .badge.warn { background: #fff0d6; color: #955600; }
    .badge.bad { background: #fee4e2; color: var(--red); }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
      gap: 14px;
      margin-bottom: 14px;
    }
    .hero-main, .panel {
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(214, 221, 232, 0.95);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .hero-main {
      min-height: 210px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 168px;
      gap: 16px;
      padding: 22px;
      overflow: hidden;
    }
    .eyebrow {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
      margin-bottom: 9px;
    }
    .location-name {
      font-size: clamp(34px, 7vw, 76px);
      font-weight: 850;
      line-height: 0.95;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }
    .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 16px;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 5px 10px;
      border-radius: 999px;
      background: var(--soft);
      border: 1px solid var(--line);
      font-weight: 700;
      color: #334155;
    }
    .confidence-ring {
      width: 148px;
      height: 148px;
      border-radius: 999px;
      align-self: center;
      justify-self: center;
      display: grid;
      place-items: center;
      background:
        conic-gradient(var(--teal) var(--confidence-deg, 0deg), #e6edf5 0deg);
      position: relative;
    }
    .confidence-ring::after {
      content: "";
      position: absolute;
      inset: 13px;
      border-radius: 999px;
      background: var(--panel);
    }
    .confidence-value {
      position: relative;
      z-index: 1;
      font-size: 29px;
      font-weight: 850;
    }
    .confidence-label {
      position: relative;
      z-index: 1;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .hero-side {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .stat {
      min-height: 100px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      box-shadow: 0 8px 26px rgba(39, 55, 77, 0.08);
    }
    .stat-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .stat-value {
      font-size: 25px;
      line-height: 1.05;
      font-weight: 850;
      overflow-wrap: anywhere;
    }
    .stat-sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 7px;
      overflow-wrap: anywhere;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.42fr) minmax(320px, 0.78fr);
      gap: 14px;
    }
    .panel {
      padding: 16px;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .panel-title {
      font-size: 16px;
      font-weight: 850;
    }
    .panel-note {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    .room-grid {
      display: grid;
      gap: 0;
      min-height: 520px;
      padding: 18px;
      border: 10px solid #c8d3df;
      border-radius: 8px;
      background:
        linear-gradient(90deg, rgba(31, 41, 55, 0.08) 1px, transparent 1px),
        linear-gradient(0deg, rgba(31, 41, 55, 0.08) 1px, transparent 1px),
        #d8e2ec;
      background-size: 34px 34px, 34px 34px, auto;
      box-shadow: inset 0 0 0 2px rgba(255,255,255,0.72);
    }
    .cell {
      min-height: 170px;
      border: 2px solid rgba(129, 145, 166, 0.55);
      border-radius: 0;
      background:
        linear-gradient(90deg, rgba(255,255,255,0.35) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,0.35) 1px, transparent 1px),
        linear-gradient(180deg, #eef3f7, #e2ebf3);
      background-size: 18px 18px, 18px 18px, auto;
      padding: 14px;
      position: relative;
      overflow: hidden;
      display: block;
    }
    .cell::before {
      content: "";
      position: absolute;
      inset: 0;
      opacity: calc(0.12 + var(--heat-ratio, 0) * 0.48);
      background:
        radial-gradient(circle at 50% 55%, rgba(37, 99, 235, 0.42), rgba(15, 159, 123, 0.18) 44%, transparent 70%);
      pointer-events: none;
      transition: opacity 220ms ease;
    }
    .cell::after {
      content: "";
      display: none;
    }
    .cell.best {
      border-color: rgba(37, 99, 235, 0.72);
      box-shadow: inset 0 0 0 3px rgba(37, 99, 235, 0.22), 0 14px 34px rgba(37, 99, 235, 0.18);
    }
    .cell.best .cell-title {
      color: #113c87;
    }
    .cell.led-on {
      box-shadow:
        inset 0 0 0 3px rgba(245, 158, 11, 0.28),
        0 14px 34px rgba(245, 158, 11, 0.16);
    }
    .cell.led-on::before {
      background:
        radial-gradient(circle at 50% 55%, rgba(245, 158, 11, 0.52), rgba(37, 99, 235, 0.20) 45%, transparent 72%);
    }
    .cell-title, .cell-prob, .cell-meta {
      position: relative;
      z-index: 1;
    }
    .cell-title {
      font-size: 14px;
      font-weight: 850;
    }
    .cell-prob {
      position: absolute;
      right: 13px;
      bottom: 12px;
      z-index: 2;
      font-size: 22px;
      line-height: 0.95;
      font-weight: 900;
      letter-spacing: 0;
    }
    .cell-meta {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-top: 8px;
    }
    .led-chip {
      position: absolute;
      z-index: 5;
      left: 12px;
      bottom: 12px;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 26px;
      max-width: calc(100% - 92px);
      padding: 4px 9px;
      border: 1px solid rgba(100, 116, 139, 0.28);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.82);
      color: #334155;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .led-dot {
      flex: 0 0 auto;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #94a3b8;
      box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.18);
    }
    .led-chip.on .led-dot {
      background: #f59e0b;
      box-shadow: 0 0 0 5px rgba(245, 158, 11, 0.18), 0 0 18px rgba(245, 158, 11, 0.7);
    }
    .led-controls {
      position: absolute;
      z-index: 6;
      right: 12px;
      top: 12px;
      display: inline-grid;
      grid-template-columns: repeat(3, 36px);
      gap: 4px;
      padding: 4px;
      border-radius: 8px;
      border: 1px solid rgba(100, 116, 139, 0.22);
      background: rgba(255, 255, 255, 0.88);
      box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
    }
    .led-btn {
      width: 36px;
      height: 28px;
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: #475569;
      font-size: 11px;
      font-weight: 850;
      cursor: pointer;
    }
    .led-btn:hover {
      background: #e2e8f0;
    }
    .led-btn.active {
      background: #1d4ed8;
      color: #ffffff;
    }
    .led-btn.on.active {
      background: #d97706;
    }
    .led-btn.off.active {
      background: #64748b;
    }
    .furniture {
      position: absolute;
      z-index: 1;
      opacity: 0.82;
      border: 2px solid rgba(79, 94, 113, 0.38);
      background: rgba(255, 255, 255, 0.55);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.48);
    }
    .furniture.bed {
      width: 46%;
      height: 34%;
      left: 12%;
      top: 41%;
      border-radius: 7px 7px 3px 3px;
      background: #dbeafe;
    }
    .furniture.bed::before {
      content: "";
      position: absolute;
      left: 7px;
      top: 7px;
      width: 34%;
      height: 26%;
      border-radius: 4px;
      background: rgba(255,255,255,0.8);
    }
    .furniture.sofa {
      width: 54%;
      height: 28%;
      right: 10%;
      top: 44%;
      border-radius: 12px;
      background: #ccfbf1;
    }
    .furniture.sofa::before, .furniture.sofa::after {
      content: "";
      position: absolute;
      top: 8px;
      bottom: 8px;
      width: 2px;
      background: rgba(15, 118, 110, 0.22);
    }
    .furniture.sofa::before { left: 34%; }
    .furniture.sofa::after { right: 34%; }
    .furniture.table {
      width: 34%;
      height: 34%;
      left: 33%;
      top: 40%;
      border-radius: 999px;
      background: #fef3c7;
    }
    .furniture.desk {
      width: 50%;
      height: 25%;
      left: 12%;
      top: 48%;
      border-radius: 5px;
      background: #e0e7ff;
    }
    .furniture.counter {
      width: 24%;
      height: 72%;
      right: 0;
      top: 14%;
      border-radius: 5px 0 0 5px;
      background: #f1f5f9;
    }
    .door {
      position: absolute;
      z-index: 1;
      width: 42px;
      height: 36px;
      left: 12px;
      bottom: -2px;
      border: 3px solid rgba(79, 94, 113, 0.45);
      border-bottom: 0;
      border-radius: 42px 42px 0 0;
      opacity: 0.65;
    }
    .person-marker {
      position: absolute;
      z-index: 4;
      left: 50%;
      top: 52%;
      width: 58px;
      height: 58px;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      background: rgba(37, 99, 235, 0.16);
      box-shadow: 0 0 0 12px rgba(37, 99, 235, 0.08), 0 10px 28px rgba(37, 99, 235, 0.26);
      display: grid;
      place-items: center;
    }
    .person-marker::before {
      content: "";
      width: 24px;
      height: 24px;
      border-radius: 999px;
      background: var(--blue);
      box-shadow: 0 20px 0 5px var(--blue);
      transform: translateY(-9px);
    }
    .person-marker::after {
      content: "";
      position: absolute;
      inset: -11px;
      border-radius: 999px;
      border: 2px solid rgba(37, 99, 235, 0.28);
    }
    .side-stack {
      display: grid;
      gap: 14px;
    }
    .node-list {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .node {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #f8fafc;
      min-height: 74px;
    }
    .node.ok { border-color: rgba(15, 159, 123, 0.45); background: #effbf7; }
    .node.warn { border-color: rgba(199, 119, 0, 0.45); background: #fff8ea; }
    .node.bad { border-color: rgba(180, 35, 24, 0.35); background: #fff1ef; }
    .node-name {
      font-size: 13px;
      font-weight: 850;
      margin-bottom: 7px;
    }
    .node-line {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .logline {
      color: #475569;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      font-size: 13px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .updated {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    @media (max-width: 920px) {
      main { padding: 12px; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .hero { grid-template-columns: 1fr; }
      .hero-main { grid-template-columns: 1fr; }
      .confidence-ring { width: 132px; height: 132px; justify-self: start; }
      .layout { grid-template-columns: 1fr; }
      .room-grid { min-height: 0; }
      .cell { min-height: 150px; }
      .room-grid { padding: 10px; }
      .person-marker { width: 48px; height: 48px; }
    }
    @media (max-width: 560px) {
      .hero-side { grid-template-columns: 1fr 1fr; }
      .stat { min-height: 86px; }
      .stat-value { font-size: 21px; }
      .node-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .panel { padding: 13px; }
    }
  </style>
</head>
<body>
<main>
  <header class="topbar">
    <div>
      <h1>CSI Location Viewer</h1>
      <div class="subtitle">ESP32 CSI realtime room monitor</div>
    </div>
    <div class="badge" id="statusBadge">starting</div>
  </header>

  <section class="hero">
    <section class="hero-main">
      <div>
        <div class="eyebrow">Stable location</div>
        <div class="location-name" id="stableLocation">Waiting</div>
        <div class="hero-meta">
          <span class="pill" id="stableMeta">-</span>
          <span class="pill" id="rawLocation">Raw: -</span>
          <span class="pill" id="rawMeta">-</span>
        </div>
      </div>
      <div class="confidence-ring" id="confidenceRing">
        <div>
          <div class="confidence-value" id="confidenceValue">0%</div>
          <div class="confidence-label">confidence</div>
        </div>
      </div>
    </section>
    <section class="hero-side">
      <div class="stat">
        <div class="stat-label">Nodes</div>
        <div class="stat-value" id="nodeCount">0/0</div>
        <div class="stat-sub" id="nodeMeta">waiting</div>
      </div>
      <div class="stat">
        <div class="stat-label">Packets</div>
        <div class="stat-value" id="packetCount">0</div>
        <div class="stat-sub" id="packetMeta">last packet -</div>
      </div>
      <div class="stat">
        <div class="stat-label">Model</div>
        <div class="stat-value" id="modelState">-</div>
        <div class="stat-sub" id="systemMeta">-</div>
      </div>
      <div class="stat">
        <div class="stat-label">Updated</div>
        <div class="stat-value" id="updatedClock">-</div>
        <div class="stat-sub">auto refresh 0.5s</div>
      </div>
    </section>
  </section>

  <section class="layout">
    <section class="panel">
      <div class="panel-head">
        <div class="panel-title">Room Probability Map</div>
        <div class="panel-note" id="gridMeta">-</div>
      </div>
      <div class="room-grid" id="grid"></div>
    </section>
    <section class="side-stack">
      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">ESP32 Nodes</div>
          <div class="panel-note" id="nodeSummary">-</div>
        </div>
        <div class="node-list" id="nodes"></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">Signal Feed</div>
          <div class="updated" id="feedState">-</div>
        </div>
        <div class="logline" id="udpStatus">-</div>
        <div class="logline" id="message" style="margin-top:8px;">-</div>
      </section>
    </section>
  </section>
</main>
<script>
const fmtPct = value => `${Math.round((Number(value) || 0) * 100)}%`;
const fmtAge = value => value == null ? "-" : `${Math.round(Number(value))} ms`;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const statusClass = status => {
  if (status === "occupied" || status === "empty_room") return "good";
  if (status === "stabilizing" || status === "warming_up" || status === "low_confidence") return "warn";
  return "bad";
};
const statusText = status => String(status || "-").replaceAll("_", " ");

async function refresh() {
  try {
    const response = await fetch("/api/location", {cache: "no-store"});
    const data = await response.json();
    render(data);
  } catch (error) {
    const badge = document.getElementById("statusBadge");
    badge.className = "badge bad";
    badge.textContent = "offline";
    document.getElementById("feedState").textContent = "disconnected";
  }
}

function render(data) {
  const stable = data.stable || {};
  const raw = data.raw || {};
  const health = data.health || {};
  const metrics = data.metrics || {};
  const probability = Number(stable.probability || raw.probability || 0);
  const stableLabel = stable.label_display || (data.status === "stabilizing" ? "Stabilizing" : "Waiting");
  const rawLabel = raw.label_display || "-";

  document.getElementById("stableLocation").textContent = stableLabel;
  document.getElementById("stableMeta").textContent = `${fmtPct(stable.probability)} stable / ${Number(stable.age_seconds || 0).toFixed(1)}s`;
  document.getElementById("rawLocation").textContent = `Raw: ${rawLabel}`;
  document.getElementById("rawMeta").textContent = `${fmtPct(raw.probability)} live`;
  document.getElementById("confidenceRing").style.setProperty("--confidence-deg", `${clamp(probability, 0, 1) * 360}deg`);
  document.getElementById("confidenceValue").textContent = fmtPct(probability);

  const badge = document.getElementById("statusBadge");
  badge.className = `badge ${statusClass(data.status)}`;
  badge.textContent = statusText(data.status);

  document.getElementById("nodeCount").textContent = `${health.active_nodes || 0}/${health.required_nodes || 0}`;
  document.getElementById("nodeMeta").textContent = `${health.min_active_nodes || 0} nodes needed`;
  document.getElementById("packetCount").textContent = metrics.packet_count ?? "-";
  document.getElementById("packetMeta").textContent = `last packet ${fmtAge(metrics.last_packet_age_ms)}`;
  document.getElementById("modelState").textContent = health.model_ready ? "Ready" : "No model";
  document.getElementById("systemMeta").textContent = health.prediction_ready ? "inference active" : "warming up";
  document.getElementById("updatedClock").textContent = new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  document.getElementById("udpStatus").textContent = data.udp_status || "-";
  document.getElementById("message").textContent = data.status_message || "-";
  document.getElementById("feedState").textContent = raw.usable ? "live" : "waiting";

  renderGrid(data);
  renderNodes(data.nodes || []);
}

function renderGrid(data) {
  const gridInfo = data.grid || {};
  const cells = data.cells || [];
  const ledZones = ((data.leds || {}).zones) || [];
  const ledsByCell = Object.fromEntries(ledZones.map(zone => [zone.cell_key, zone]));
  const grid = document.getElementById("grid");
  const cols = Math.max(1, Number(gridInfo.cols || 1));
  const rows = Math.max(1, Number(gridInfo.rows || 1));
  grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  grid.innerHTML = "";
  const ledSummary = data.leds?.enabled ? `${ledZones.filter(zone => zone.is_on).length} LED on` : "LED disabled";
  document.getElementById("gridMeta").textContent = `${cols} x ${rows} cells / ${ledSummary}`;
  for (const cell of cells) {
    const probability = Number(cell.probability || 0);
    const led = ledsByCell[cell.cell_key] || null;
    const ledMode = led?.mode || "auto";
    const ledOn = Boolean(led?.is_on);
    const item = document.createElement("article");
    item.className = `cell ${cell.is_best ? "best" : ""} ${ledOn ? "led-on" : ""}`;
    item.style.setProperty("--heat", `${clamp(probability * 100, 0, 100)}%`);
    item.style.setProperty("--heat-ratio", `${clamp(probability, 0, 1)}`);
    item.innerHTML = `
      <div>
        <div class="cell-title">Cell (${Number(cell.grid_x) + 1}, ${Number(cell.grid_y) + 1})</div>
        <div class="cell-meta">${cell.trained ? "trained" : "not trained"} / ${cell.window_sample_count || 0} windows</div>
      </div>
      ${led ? `
        <div class="led-chip ${ledOn ? "on" : ""}" title="GPIO ${led.pin_bcm}">
          <span class="led-dot"></span>
          <span>GPIO ${led.pin_bcm} ${ledOn ? "on" : "off"}</span>
        </div>
        <div class="led-controls" aria-label="LED control for ${led.label}">
          ${ledButton("auto", ledMode)}
          ${ledButton("on", ledMode)}
          ${ledButton("off", ledMode)}
        </div>
      ` : ""}
      ${roomDetails(cell, cols, rows)}
      ${cell.is_best ? '<div class="person-marker" aria-label="Current person location"></div>' : ''}
      <div>
        <div class="cell-prob">${fmtPct(probability)}</div>
        <div class="cell-meta">${cell.is_best ? "current best" : "probability"}</div>
      </div>
    `;
    item.querySelectorAll("[data-led-mode]").forEach(button => {
      button.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        setLedMode(cell.cell_key, button.dataset.ledMode || "auto");
      });
    });
    grid.appendChild(item);
  }
}

function ledButton(mode, activeMode) {
  const label = mode === "auto" ? "A" : (mode === "on" ? "On" : "Off");
  const title = mode === "auto" ? "Follow location" : `Turn ${mode}`;
  const active = mode === activeMode ? "active" : "";
  return `<button class="led-btn ${mode} ${active}" data-led-mode="${mode}" title="${title}">${label}</button>`;
}

async function setLedMode(cellKey, mode) {
  try {
    await fetch("/api/leds", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({cell_key: cellKey, mode}),
    });
    refresh();
  } catch (error) {
    const badge = document.getElementById("statusBadge");
    badge.className = "badge bad";
    badge.textContent = "led error";
  }
}

function roomDetails(cell, cols, rows) {
  const x = Number(cell.grid_x || 0);
  const y = Number(cell.grid_y || 0);
  const index = y * cols + x;
  const kind = ["bed", "table", "sofa", "desk", "counter", "table"][index % 6];
  const door = y === rows - 1 && x === 0 ? '<div class="door"></div>' : '';
  return `<div class="furniture ${kind}"></div>${door}`;
}

function renderNodes(nodes) {
  const body = document.getElementById("nodes");
  body.innerHTML = "";
  document.getElementById("nodeSummary").textContent = `${nodes.length} tracked`;
  for (const node of nodes) {
    const age = Number(node.age_ms ?? 999999);
    const snr = Number(node.snr_db || 0);
    const level = age > 1500 ? "bad" : (snr < 35 ? "warn" : "ok");
    const item = document.createElement("div");
    item.className = `node ${level}`;
    item.innerHTML = `
      <div class="node-name">${node.label || `ESP ${node.node_id}`}</div>
      <div class="node-line">age ${fmtAge(node.age_ms)}</div>
      <div class="node-line">RSSI ${Number(node.rssi_dbm || 0).toFixed(0)} / SNR ${snr.toFixed(0)}</div>
    `;
    body.appendChild(item);
  }
}

refresh();
setInterval(refresh, 500);
</script>
</body>
</html>
"""
