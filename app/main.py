from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import FingerprintEngine
from app.gui import FingerprintAppWindow
from app.receiver import build_receivers_for_config
from app.runtime import EngineRuntimeThread
from app.stimulus import build_stimulus_for_config


def main() -> int:
    workspace = Path(__file__).resolve().parent
    engine = FingerprintEngine(workspace)
    receiver = build_receivers_for_config(engine, engine.system_config)
    stimulus = build_stimulus_for_config(engine.system_config)
    runtime = EngineRuntimeThread(engine)
    receiver.start()
    stimulus.start()
    runtime.start()
    window = FingerprintAppWindow(engine, receiver)
    try:
        window.run()
    finally:
        receiver.stop()
        stimulus.stop()
        runtime.stop()
        receiver.join(timeout=1.5)
        stimulus.join(timeout=1.5)
        runtime.join(timeout=1.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
