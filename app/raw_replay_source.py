from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPLAY_SOURCE_PREFIX = "raw-replay:"


@dataclass(frozen=True)
class RawReplayConfig:
    raw_dir: Path
    fallback_after_seconds: float = 15.0
    replay_speedup: float = 20.0
    loop: bool = True
    external_signal_grace_ms: float = 1500.0


class RawDataReplayThread(threading.Thread):
    def __init__(self, engine: Any, config: RawReplayConfig) -> None:
        super().__init__(name="raw-data-live-replay", daemon=True)
        self.engine = engine
        self.config = config
        self.stop_event = threading.Event()
        self.active = False
        self.status_message = "Raw replay fallback is armed."
        self.armed_at = time.monotonic()

    def run(self) -> None:
        sessions = self._session_paths()
        if not sessions:
            self.status_message = f"No raw replay sessions found in {self.config.raw_dir}."
            return

        while not self.stop_event.is_set():
            if self._external_signal_active():
                self.active = False
                self.status_message = "Real ESP32 signal detected; raw replay is paused."
                self.stop_event.wait(0.5)
                continue

            if not self._fallback_delay_elapsed():
                remaining = max(0.0, self.config.fallback_after_seconds - self._last_packet_age())
                self.status_message = f"Waiting {remaining:.1f}s before raw replay fallback."
                self.stop_event.wait(0.5)
                continue

            self.active = True
            self.status_message = "Raw replay fallback is streaming saved CSI sessions."
            self._set_udp_status("No live ESP32 signal; streaming saved raw CSI data.")
            for session_path in sessions:
                if self.stop_event.is_set() or self._external_signal_active():
                    break
                self._replay_session(session_path)

            if not self.config.loop:
                break

        self.active = False

    def stop(self) -> None:
        self.stop_event.set()

    def _session_paths(self) -> list[Path]:
        if not self.config.raw_dir.exists():
            return []
        paths = [
            path
            for path in self.config.raw_dir.glob("*.jsonl")
            if path.name != "sessions.jsonl"
        ]
        return sorted(paths, key=_session_sort_key)

    def _replay_session(self, path: Path) -> None:
        first_ts: float | None = None
        session_started_at = time.perf_counter()
        packet_count = 0
        self._set_udp_status(f"Raw replay: {path.name}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if self.stop_event.is_set() or self._external_signal_active():
                    return
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("record_type") != "packet" or not record.get("valid_adr018"):
                    continue
                raw_b64 = record.get("raw_b64")
                if not isinstance(raw_b64, str) or not raw_b64:
                    continue
                captured_at = _float(record.get("captured_at_unix"))
                if first_ts is None:
                    first_ts = captured_at
                    session_started_at = time.perf_counter()
                target_elapsed = max(0.0, captured_at - first_ts) / self._speedup()
                while not self.stop_event.is_set():
                    current_elapsed = time.perf_counter() - session_started_at
                    wait_seconds = target_elapsed - current_elapsed
                    if wait_seconds <= 0.0:
                        break
                    self.stop_event.wait(min(wait_seconds, 0.05))
                try:
                    payload = base64.b64decode(raw_b64)
                except ValueError:
                    continue
                node_id = int(_float(record.get("node_id")))
                source = f"{REPLAY_SOURCE_PREFIX}{path.name}:node{node_id}"
                self.engine.process_packet(payload, source)
                packet_count += 1
        self._set_udp_status(f"Raw replay completed {path.name}: {packet_count} packets.")

    def _fallback_delay_elapsed(self) -> bool:
        snapshot = self.engine.snapshot()
        metrics = snapshot.get("metrics", {})
        if not isinstance(metrics, dict):
            return True
        packet_count = int(_float(metrics.get("packet_count")))
        last_age = metrics.get("last_packet_age_ms")
        if packet_count <= 0 or last_age is None:
            return time.monotonic() - self.armed_at >= self.config.fallback_after_seconds
        return float(last_age) >= self.config.fallback_after_seconds * 1000.0

    def _last_packet_age(self) -> float:
        snapshot = self.engine.snapshot()
        metrics = snapshot.get("metrics", {})
        if not isinstance(metrics, dict):
            return time.monotonic() - self.armed_at
        age_ms = metrics.get("last_packet_age_ms")
        if age_ms is None:
            return time.monotonic() - self.armed_at
        return max(0.0, _float(age_ms) / 1000.0)

    def _external_signal_active(self) -> bool:
        snapshot = self.engine.snapshot()
        nodes = snapshot.get("nodes", [])
        if not isinstance(nodes, list):
            return False
        for node in nodes:
            if not isinstance(node, dict):
                continue
            source = str(node.get("source", ""))
            age_ms = node.get("age_ms")
            if source.startswith(REPLAY_SOURCE_PREFIX):
                continue
            if age_ms is None:
                continue
            if _float(age_ms) <= self.config.external_signal_grace_ms:
                return True
        return False

    def _set_udp_status(self, message: str) -> None:
        try:
            self.engine.set_udp_status(message)
        except Exception:
            pass

    def _speedup(self) -> float:
        return max(1.0, float(self.config.replay_speedup))


def _session_sort_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if "empty_room" in name:
        return (0, name)
    return (1, name)


def _float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0
