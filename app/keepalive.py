from __future__ import annotations

import subprocess
import sys
import threading
import time

from .core import FingerprintEngine


class KeepalivePingThread(threading.Thread):
    def __init__(self, engine: FingerprintEngine) -> None:
        super().__init__(name="fingerprint-keepalive-ping", daemon=True)
        self.engine = engine
        self.stop_event = threading.Event()

    def run(self) -> None:
        next_targets_refresh = 0.0
        targets: list[str] = []
        pings_per_second = 0.0

        while not self.stop_event.is_set():
            now = time.time()
            if now >= next_targets_refresh:
                snapshot = self.engine.snapshot()
                pings_per_second = float(snapshot["host"].get("keepalive_pings_per_second", 0.0))
                targets = self._extract_targets(snapshot.get("nodes", []))
                next_targets_refresh = now + 1.0

            if pings_per_second <= 0.0 or not targets:
                self.stop_event.wait(0.25)
                continue

            interval = max(0.05, 1.0 / pings_per_second)
            timeout_ms = max(150, min(1000, int(interval * 1000.0)))
            started_at = time.time()

            for target in targets:
                if self.stop_event.is_set():
                    return
                self._ping_once(target, timeout_ms)

            elapsed = time.time() - started_at
            self.stop_event.wait(max(0.0, interval - elapsed))

    def stop(self) -> None:
        self.stop_event.set()

    @staticmethod
    def _extract_targets(nodes: list[dict[str, object]]) -> list[str]:
        targets: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            source = str(node.get("source", ""))
            if not source:
                continue
            host = source.split(":", 1)[0].strip()
            if not host or host in seen:
                continue
            seen.add(host)
            targets.append(host)
        return targets

    @staticmethod
    def _ping_once(target: str, timeout_ms: int) -> None:
        if sys.platform == "win32":
            cmd = ["ping", "-n", "1", "-w", str(timeout_ms), target]
        else:
            timeout_seconds = max(1, int((timeout_ms + 999) / 1000))
            cmd = ["ping", "-c", "1", "-W", str(timeout_seconds), target]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=max(1.5, timeout_ms / 1000.0 + 0.5),
            )
        except (OSError, subprocess.SubprocessError):
            return
