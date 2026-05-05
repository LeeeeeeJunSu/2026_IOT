from __future__ import annotations

import sys
import threading
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.raw_capture import RawCaptureEngine
from app.raw_gui import RawCaptureAppWindow
from app.receiver import build_receivers_for_config
from app.stimulus import build_stimulus_for_config


class RawRuntimeThread(threading.Thread):
    def __init__(self, engine: RawCaptureEngine) -> None:
        super().__init__(name="raw-capture-runtime", daemon=True)
        self.engine = engine
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.engine.run_runtime_tick()
            self.stop_event.wait(self.engine.runtime_tick_seconds)

    def stop(self) -> None:
        self.stop_event.set()


def main() -> int:
    workspace = Path(__file__).resolve().parent
    engine = RawCaptureEngine(workspace)
    receiver = build_receivers_for_config(
        engine,
        engine.system_config,
        include_payload=True,
    )
    stimulus = build_stimulus_for_config(engine.system_config)
    runtime = RawRuntimeThread(engine)
    receiver.start()
    stimulus.start()
    runtime.start()
    window = RawCaptureAppWindow(engine, receiver)
    try:
        window.run()
    finally:
        engine.stop_capture()
        receiver.stop()
        stimulus.stop()
        runtime.stop()
        receiver.join(timeout=1.5)
        stimulus.join(timeout=1.5)
        runtime.join(timeout=1.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
