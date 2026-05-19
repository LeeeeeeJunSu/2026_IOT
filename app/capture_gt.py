from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.raw_capture import RawCaptureEngine
from app.receiver import build_receivers_for_config
from app.stimulus import build_stimulus_for_config


class RawRuntimeThread(threading.Thread):
    def __init__(self, engine: RawCaptureEngine) -> None:
        super().__init__(name="gt-capture-runtime", daemon=True)
        self.engine = engine
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.engine.run_runtime_tick()
            self.stop_event.wait(self.engine.runtime_tick_seconds)

    def stop(self) -> None:
        self.stop_event.set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture raw CSI packets from all configured ESP32 nodes with a GT location."
    )
    parser.add_argument(
        "--gt",
        type=int,
        choices=range(0, 7),
        required=True,
        metavar="{0,1,2,3,4,5,6}",
        help="Ground-truth location number to store in JSONL records.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Capture duration. Defaults to Config/system_config.json capture_seconds. Use 0 for manual Ctrl+C stop.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=10.0,
        help="Seconds to wait after node readiness before recording starts. Defaults to 10.",
    )
    parser.add_argument(
        "--min-active-nodes",
        type=int,
        default=None,
        help="Wait until at least this many nodes are active before capture. Defaults to all enabled nodes.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for active nodes before starting anyway. Use 0 to skip waiting.",
    )
    parser.add_argument(
        "--node-fresh-seconds",
        type=float,
        default=3.0,
        help="A node is active when its last packet is newer than this many seconds.",
    )
    parser.add_argument(
        "--no-stimulus",
        action="store_true",
        help="Disable the UDP broadcast stimulus during capture.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(__file__).resolve().parent
    engine = RawCaptureEngine(workspace)
    enabled_node_count = len(list(engine.system_config.enabled_nodes()))
    min_active_nodes = (
        enabled_node_count
        if args.min_active_nodes is None
        else max(1, int(args.min_active_nodes))
    )

    receiver = build_receivers_for_config(
        engine,
        engine.system_config,
        include_payload=True,
    )
    stimulus = (
        None
        if args.no_stimulus
        else build_stimulus_for_config(engine.system_config)
    )
    runtime = RawRuntimeThread(engine)

    receiver.start()
    if stimulus is not None:
        stimulus.start()
    runtime.start()

    exit_code = 0
    try:
        if args.wait_timeout > 0 and min_active_nodes > 0:
            _wait_for_nodes(
                engine,
                min_active_nodes=min_active_nodes,
                timeout_seconds=float(args.wait_timeout),
                fresh_seconds=max(0.1, float(args.node_fresh_seconds)),
            )

        engine.start_ground_truth_capture(
            int(args.gt),
            duration_seconds=args.seconds,
            start_delay_seconds=max(0.0, float(args.start_delay)),
        )
        exit_code = _monitor_capture(engine)
    finally:
        engine.stop_capture()
        receiver.stop()
        if stimulus is not None:
            stimulus.stop()
        runtime.stop()
        receiver.join(timeout=1.5)
        if stimulus is not None:
            stimulus.join(timeout=1.5)
        runtime.join(timeout=1.5)
    return exit_code


def _wait_for_nodes(
    engine: RawCaptureEngine,
    *,
    min_active_nodes: int,
    timeout_seconds: float,
    fresh_seconds: float,
) -> None:
    deadline = time.time() + timeout_seconds
    last_reported_count = -1
    while time.time() < deadline:
        snapshot = engine.snapshot()
        active_nodes = _active_node_ids(snapshot, fresh_seconds)
        if len(active_nodes) != last_reported_count:
            last_reported_count = len(active_nodes)
            print(
                f"Waiting for nodes: {len(active_nodes)}/{min_active_nodes} active "
                f"{active_nodes}",
                flush=True,
            )
        if len(active_nodes) >= min_active_nodes:
            return
        time.sleep(0.5)

    snapshot = engine.snapshot()
    active_nodes = _active_node_ids(snapshot, fresh_seconds)
    print(
        f"Node wait timeout: starting capture with {len(active_nodes)}/{min_active_nodes} "
        f"active nodes {active_nodes}",
        flush=True,
    )


def _monitor_capture(engine: RawCaptureEngine) -> int:
    output_path = ""
    try:
        while True:
            snapshot = engine.snapshot()
            capture = snapshot["capture"]
            if isinstance(capture, dict) and capture.get("active"):
                output_path = str(capture.get("path") or output_path)
                remaining = capture.get("remaining_seconds")
                remaining_text = (
                    "manual" if remaining is None else f"{float(remaining):.1f}s left"
                )
                print(
                    "capturing: "
                    f"saved={capture.get('saved_packets', 0)} "
                    f"valid={capture.get('valid_packets', 0)} "
                    f"nodes={capture.get('packets_by_node', [])} "
                    f"{remaining_text}",
                    flush=True,
                )
                time.sleep(1.0)
                continue
            print(f"Capture finished: {output_path}", flush=True)
            return 0
    except KeyboardInterrupt:
        print("Stopping capture...", flush=True)
        return 130


def _active_node_ids(snapshot: dict[str, object], fresh_seconds: float) -> list[int]:
    nodes = snapshot.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    active: list[int] = []
    fresh_ms = fresh_seconds * 1000.0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        age_ms = node.get("age_ms")
        node_id = node.get("node_id")
        if (
            isinstance(age_ms, (int, float))
            and isinstance(node_id, int)
            and age_ms <= fresh_ms
        ):
            active.append(int(node_id))
    return sorted(active)


if __name__ == "__main__":
    raise SystemExit(main())
