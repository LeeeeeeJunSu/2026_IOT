from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import FingerprintEngine
from app.dashboard import build_dashboard_for_engine
from app.receiver import build_receivers_for_config
from app.runtime import EngineRuntimeThread
from app.stimulus import build_stimulus_for_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CSI location dashboard.")
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Active model to use. Examples: VariableNodeAggregateExtraTrees, "
            "DeepCNNV1, DeepGRUV1, cnn, gru."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print loadable models and exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(__file__).resolve().parent
    engine = FingerprintEngine(workspace)
    if args.list_models:
        models = engine.available_model_names()
        if not models:
            print("No trained models found.", flush=True)
            return 1
        print("Available models:", flush=True)
        for model_name in models:
            marker = "*" if model_name == engine.active_model_name else " "
            print(f"{marker} {model_name}", flush=True)
        return 0
    if args.model:
        engine.set_active_model(args.model)
    receiver = build_receivers_for_config(engine, engine.system_config)
    stimulus = build_stimulus_for_config(engine.system_config)
    runtime = EngineRuntimeThread(engine)
    dashboard = build_dashboard_for_engine(engine)

    receiver.start()
    stimulus.start()
    runtime.start()
    dashboard.start()
    dashboard.ready_event.wait(timeout=1.5)
    if dashboard.actual_port is None:
        print(f"Location dashboard failed to start: {dashboard.error}", flush=True)
        receiver.stop()
        stimulus.stop()
        runtime.stop()
        dashboard.stop()
        return 1

    print(
        "Location dashboard: "
        f"{dashboard.local_url} "
        f"(LAN hint: {dashboard.lan_url_hint}) "
        f"active_model={engine.active_model_name}",
        flush=True,
    )
    print("Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        stimulus.stop()
        runtime.stop()
        dashboard.stop()
        receiver.join(timeout=1.5)
        stimulus.join(timeout=1.5)
        runtime.join(timeout=1.5)
        dashboard.join(timeout=1.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
