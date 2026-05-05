from __future__ import annotations

import threading
import time

from .core import FingerprintEngine


class EngineRuntimeThread(threading.Thread):
    def __init__(self, engine: FingerprintEngine) -> None:
        super().__init__(name="fingerprint-engine-runtime", daemon=True)
        self.engine = engine
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.engine.run_runtime_tick()
            self.stop_event.wait(self.engine.runtime_tick_seconds)

    def stop(self) -> None:
        self.stop_event.set()
