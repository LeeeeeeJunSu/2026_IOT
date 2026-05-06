from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Config.config_loader import load_system_config, save_system_config

from app.core import EMPTY_ROOM_CLASS_KEY, FingerprintEngine
from app.receiver import build_receivers_for_config
from app.runtime import EngineRuntimeThread


@dataclass(frozen=True)
class ReplayPacket:
    captured_at: float
    node_id: int
    payload: bytes


@dataclass(frozen=True)
class ReplaySession:
    session_id: str
    kind: str
    label: str
    grid_x: int | None
    grid_y: int | None
    path: Path
    packets: tuple[ReplayPacket, ...]

    @property
    def duration_seconds(self) -> float:
        if len(self.packets) < 2:
            return 0.0
        return max(0.0, self.packets[-1].captured_at - self.packets[0].captured_at)

    @property
    def expected_label_key(self) -> str:
        if self.kind == "empty_room":
            return EMPTY_ROOM_CLASS_KEY
        if self.grid_x is None or self.grid_y is None:
            raise RuntimeError(f"Cell session {self.path} is missing grid coordinates.")
        return f"{self.grid_x},{self.grid_y}"

    @property
    def display_label(self) -> str:
        if self.kind == "empty_room":
            return "Empty Room"
        return f"Cell ({int(self.grid_x) + 1}, {int(self.grid_y) + 1})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay app/raw_data as real UDP traffic and run an end-to-end smoke test "
            "through baseline capture, Learn capture, model training, model reload, "
            "and live inference."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw_data",
        help="Directory containing captured raw JSONL sessions.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw_udp_replay_smoke_report.json",
        help="Where to write the smoke-test report JSON.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Optional isolated workspace root for the smoke test.",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=10.0,
        help="Replay speedup factor. 10 means 60s raw traffic replays in about 6s.",
    )
    parser.add_argument(
        "--capture-padding-seconds",
        type=float,
        default=0.20,
        help="Extra capture headroom added on top of the replay duration.",
    )
    parser.add_argument(
        "--receiver-ready-seconds",
        type=float,
        default=0.30,
        help="How long to wait after starting UDP receiver threads.",
    )
    parser.add_argument(
        "--prediction-settle-seconds",
        type=float,
        default=0.25,
        help="Extra settle time after replay finishes before reading the final prediction.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_progress(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def display_label(label_key: str | None) -> str:
    if not label_key:
        return ""
    if label_key == EMPTY_ROOM_CLASS_KEY:
        return "Empty Room"
    grid_x, grid_y = [int(value) for value in label_key.split(",", 1)]
    return f"Cell ({grid_x + 1}, {grid_y + 1})"


def load_replay_sessions(raw_dir: Path) -> list[ReplaySession]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    sessions: list[ReplaySession] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        if path.name == "sessions.jsonl":
            continue
        packets: list[ReplayPacket] = []
        session_meta: dict[str, Any] | None = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record_type = str(record.get("record_type", ""))
                if record_type == "packet" and bool(record.get("valid_adr018", False)):
                    raw_b64 = record.get("raw_b64")
                    if not isinstance(raw_b64, str) or not raw_b64:
                        continue
                    packets.append(
                        ReplayPacket(
                            captured_at=float(record.get("captured_at_unix", 0.0)),
                            node_id=int(record.get("node_id", 0)),
                            payload=base64.b64decode(raw_b64),
                        )
                    )
                elif record_type in {"session_armed", "session_start", "session_end"}:
                    session_meta = record
        if session_meta is None or not packets:
            continue
        packets.sort(key=lambda item: item.captured_at)
        sessions.append(
            ReplaySession(
                session_id=str(session_meta.get("session_id", path.stem)),
                kind=str(session_meta.get("kind", "")),
                label=str(session_meta.get("label", path.stem)),
                grid_x=(
                    None
                    if session_meta.get("grid_x") is None
                    else int(session_meta.get("grid_x", 0))
                ),
                grid_y=(
                    None
                    if session_meta.get("grid_y") is None
                    else int(session_meta.get("grid_y", 0))
                ),
                path=path.resolve(),
                packets=tuple(packets),
            )
        )
    if not sessions:
        raise RuntimeError(f"No replayable raw sessions found in {raw_dir}")
    return sessions


def split_sessions(
    sessions: list[ReplaySession],
) -> tuple[ReplaySession, list[ReplaySession]]:
    empty_room_sessions = [session for session in sessions if session.kind == "empty_room"]
    if len(empty_room_sessions) != 1:
        raise RuntimeError(
            f"Expected exactly one empty-room session, found {len(empty_room_sessions)}."
        )
    cell_sessions = [session for session in sessions if session.kind != "empty_room"]
    if not cell_sessions:
        raise RuntimeError("No cell sessions found in raw data.")
    cell_sessions.sort(
        key=lambda session: (
            -1 if session.grid_y is None else int(session.grid_y),
            -1 if session.grid_x is None else int(session.grid_x),
        )
    )
    return empty_room_sessions[0], cell_sessions


def create_isolated_workspace(
    *,
    repo_root: Path,
    report_path: Path,
    workspace_root: Path | None,
    baseline_session: ReplaySession,
    cell_sessions: list[ReplaySession],
    speedup: float,
    capture_padding_seconds: float,
) -> dict[str, Any]:
    if speedup <= 0.0:
        raise RuntimeError("Speedup must be positive.")

    if workspace_root is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        workspace_root = report_path.parent / f"raw_udp_replay_workspace_{timestamp}"
    workspace_root = workspace_root.resolve()
    app_workspace = workspace_root / "app"
    data_dir = app_workspace / "data"
    config_dir = workspace_root / "Config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    config = load_system_config(repo_root / "Config" / "system_config.json")
    max_grid_x = max(int(session.grid_x or 0) for session in cell_sessions)
    max_grid_y = max(int(session.grid_y or 0) for session in cell_sessions)
    config.grid.cols = max_grid_x + 1
    config.grid.rows = max_grid_y + 1

    config.host.listen_host = "127.0.0.1"
    config.host.target_ip = "127.0.0.1"
    config.host.udp_port = 5600

    enabled_node_ids = sorted(
        {
            int(packet.node_id)
            for session in [baseline_session, *cell_sessions]
            for packet in session.packets
        }
    )
    enabled_set = set(enabled_node_ids)
    for node in config.nodes:
        node.enabled = node.node_id in enabled_set
        if node.enabled:
            node.target_port = 5600 + int(node.node_id)

    baseline_replay_seconds = max(
        1.25,
        baseline_session.duration_seconds / speedup,
    )
    cell_replay_seconds = max(
        1.25,
        max(session.duration_seconds for session in cell_sessions) / speedup,
    )
    config.fingerprinting.baseline_capture_seconds = max(
        config.fingerprinting.window_seconds + 0.1,
        baseline_replay_seconds + capture_padding_seconds,
    )
    config.fingerprinting.capture_seconds = max(
        config.fingerprinting.window_seconds + 0.1,
        cell_replay_seconds + capture_padding_seconds,
    )
    config.fingerprinting.baseline_start_delay_seconds = 0.0
    config.fingerprinting.capture_auto_extend_seconds = 0.0
    config.fingerprinting.capture_extend_step_seconds = 0.25
    config.fingerprinting.live_probability_smoothing_seconds = 0.0
    config.fingerprinting.best_cell_switch_margin = 0.0
    config.fingerprinting.best_cell_switch_delay_seconds = 0.0
    config.fingerprinting.prediction_stale_grace_seconds = 2.0
    config.fingerprinting.smoothing_half_window = 0

    config_path = config_dir / "system_config.json"
    save_system_config(config_path, config)

    node_ports = {
        int(node.node_id): int(config.node_target_port(node))
        for node in config.enabled_nodes()
    }
    return {
        "workspace_root": str(workspace_root),
        "app_workspace": str(app_workspace),
        "config_path": str(config_path),
        "node_ports": node_ports,
        "baseline_replay_seconds": baseline_replay_seconds,
        "cell_replay_seconds": cell_replay_seconds,
    }


def wait_for_snapshot(
    engine: FingerprintEngine,
    predicate,
    *,
    timeout: float,
    poll_seconds: float = 0.05,
) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_snapshot: dict[str, Any] | None = None
    while time.time() < deadline:
        last_snapshot = engine.snapshot()
        if predicate(last_snapshot):
            return last_snapshot
        time.sleep(poll_seconds)
    if last_snapshot is None:
        last_snapshot = engine.snapshot()
    raise TimeoutError(
        f"Timed out after {timeout:.1f}s. Last status: {last_snapshot.get('status_message', '')}"
    )


def snapshot_cell(snapshot: dict[str, Any], grid_x: int, grid_y: int) -> dict[str, Any]:
    for cell in snapshot["cells"]:
        if cell["grid_x"] == grid_x and cell["grid_y"] == grid_y:
            return cell
    raise KeyError(f"Cell ({grid_x}, {grid_y}) not found in snapshot.")


def start_runtime(
    app_workspace: Path,
    *,
    receiver_ready_seconds: float,
) -> tuple[FingerprintEngine, Any, EngineRuntimeThread]:
    engine = FingerprintEngine(app_workspace)
    receiver_group = build_receivers_for_config(
        engine,
        engine.system_config,
        multiprocessing=False,
    )
    runtime = EngineRuntimeThread(engine)
    receiver_group.start()
    runtime.start()
    ready_snapshot = wait_for_snapshot(
        engine,
        lambda snap: (
            "Listening on" in str(snap["udp_status"])
            or "UDP bind failed" in str(snap["udp_status"])
        ),
        timeout=max(1.0, receiver_ready_seconds + 1.0),
    )
    udp_status = str(ready_snapshot["udp_status"])
    if "UDP bind failed" in udp_status:
        raise RuntimeError(udp_status)
    time.sleep(max(0.0, receiver_ready_seconds))
    return engine, receiver_group, runtime


def stop_runtime(receiver_group: Any | None, runtime: EngineRuntimeThread | None) -> None:
    if runtime is not None:
        runtime.stop()
    if receiver_group is not None:
        receiver_group.stop()
    if receiver_group is not None:
        receiver_group.join(timeout=2.0)
    if runtime is not None:
        runtime.join(timeout=2.0)


def replay_session(
    session: ReplaySession,
    *,
    host: str,
    ports_by_node: dict[int, int],
    speedup: float,
) -> dict[str, Any]:
    if speedup <= 0.0:
        raise RuntimeError("Speedup must be positive.")
    if not session.packets:
        raise RuntimeError(f"Replay session {session.path} is empty.")

    start_perf = time.perf_counter()
    start_capture_time = session.packets[0].captured_at
    sent_packets = 0
    sent_by_node: dict[int, int] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for packet in session.packets:
            target_elapsed = (packet.captured_at - start_capture_time) / speedup
            while True:
                remaining = target_elapsed - (time.perf_counter() - start_perf)
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, 0.002))
            port = ports_by_node.get(packet.node_id)
            if port is None:
                continue
            sock.sendto(packet.payload, (host, port))
            sent_packets += 1
            sent_by_node[packet.node_id] = sent_by_node.get(packet.node_id, 0) + 1

    return {
        "duration_seconds": time.perf_counter() - start_perf,
        "packet_count": sent_packets,
        "packets_by_node": sent_by_node,
    }


def summarize_prediction(snapshot: dict[str, Any]) -> dict[str, Any]:
    entries = [
        {
            "label_key": cell["cell_key"],
            "label_display": display_label(str(cell["cell_key"])),
            "probability": float(cell["probability"]),
            "is_best": bool(cell["is_best"]),
        }
        for cell in snapshot["cells"]
    ]
    empty_room = snapshot["empty_room"]
    entries.append(
        {
            "label_key": EMPTY_ROOM_CLASS_KEY,
            "label_display": "Empty Room",
            "probability": float(empty_room["probability"]),
            "is_best": bool(empty_room["is_best"]),
        }
    )
    entries.sort(key=lambda item: float(item["probability"]), reverse=True)
    return {
        "best_label_key": snapshot["prediction"]["best_label_key"],
        "best_label_display": snapshot["prediction"]["best_label_display"],
        "best_probability": float(snapshot["prediction"]["best_probability"]),
        "top_probabilities": entries[:4],
    }


def label_probability(snapshot: dict[str, Any], label_key: str) -> float:
    if label_key == EMPTY_ROOM_CLASS_KEY:
        return float(snapshot["empty_room"]["probability"])
    for cell in snapshot["cells"]:
        if cell["cell_key"] == label_key:
            return float(cell["probability"])
    return 0.0


def build_prediction_trace_row(
    snapshot: dict[str, Any],
    *,
    expected_label_key: str,
) -> dict[str, Any]:
    prediction_summary = summarize_prediction(snapshot)
    return {
        "captured_at": time.time(),
        "best_label_key": prediction_summary["best_label_key"],
        "best_label_display": prediction_summary["best_label_display"],
        "best_probability": prediction_summary["best_probability"],
        "expected_probability": label_probability(snapshot, expected_label_key),
    }


def run_training_pass(
    *,
    app_workspace: Path,
    baseline_session: ReplaySession,
    cell_sessions: list[ReplaySession],
    node_ports: dict[int, int],
    speedup: float,
    receiver_ready_seconds: float,
) -> dict[str, Any]:
    engine: FingerprintEngine | None = None
    receiver_group: Any | None = None
    runtime: EngineRuntimeThread | None = None
    try:
        engine, receiver_group, runtime = start_runtime(
            app_workspace,
            receiver_ready_seconds=receiver_ready_seconds,
        )

        print_progress("Training phase: starting empty-room baseline replay.")
        engine.start_baseline_capture(reset_training=True)
        time.sleep(0.05)
        baseline_replay = replay_session(
            baseline_session,
            host="127.0.0.1",
            ports_by_node=node_ports,
            speedup=speedup,
        )
        baseline_snapshot = wait_for_snapshot(
            engine,
            lambda snap: (not snap["capture"]["active"] and snap["baseline"]["ready"]),
            timeout=engine.baseline_capture_seconds + 5.0,
        )

        capture_rows: list[dict[str, Any]] = []
        for session in cell_sessions:
            if session.grid_x is None or session.grid_y is None:
                raise RuntimeError(f"Missing grid coordinates for {session.path}")
            before_snapshot = engine.snapshot()
            previous_capture_count = snapshot_cell(
                before_snapshot,
                session.grid_x,
                session.grid_y,
            )["capture_count"]
            print_progress(f"Training phase: replaying {session.display_label}.")
            engine.start_capture(session.grid_x, session.grid_y)
            time.sleep(0.05)
            replay_stats = replay_session(
                session,
                host="127.0.0.1",
                ports_by_node=node_ports,
                speedup=speedup,
            )
            finished_snapshot = wait_for_snapshot(
                engine,
                lambda snap: (
                    not snap["capture"]["active"]
                    and snapshot_cell(snap, session.grid_x, session.grid_y)["capture_count"]
                    > previous_capture_count
                ),
                timeout=engine.capture_seconds + 5.0,
            )
            cell = snapshot_cell(finished_snapshot, session.grid_x, session.grid_y)
            capture_rows.append(
                {
                    "label": session.display_label,
                    "cell_key": session.expected_label_key,
                    "packet_count": replay_stats["packet_count"],
                    "replay_duration_seconds": replay_stats["duration_seconds"],
                    "capture_count": cell["capture_count"],
                    "window_sample_count": cell["window_sample_count"],
                    "observed_window_ratio": float(cell["observed_window_ratio"]),
                    "status_message": finished_snapshot["status_message"],
                }
            )

        train_status = engine.train_models()
        train_snapshot = engine.snapshot()
        return {
            "baseline": {
                "packet_count": baseline_replay["packet_count"],
                "replay_duration_seconds": baseline_replay["duration_seconds"],
                "captured_nodes": baseline_snapshot["baseline"]["captured_nodes"],
                "required_nodes": baseline_snapshot["baseline"]["required_nodes"],
                "empty_room_windows": baseline_snapshot["empty_room"]["window_sample_count"],
                "status_message": baseline_snapshot["status_message"],
            },
            "captures": capture_rows,
            "train_status": train_status,
            "training_snapshot": {
                "trained_cells": train_snapshot["training"]["trained_cells"],
                "trained_classes": train_snapshot["training"]["trained_classes"],
                "total_classes": train_snapshot["training"]["total_classes"],
                "dataset_samples": train_snapshot["training"]["dataset_samples"],
                "empty_room_samples": train_snapshot["training"]["empty_room_samples"],
                "available_models": list(train_snapshot["training"]["available_models"]),
                "active_model": train_snapshot["training"]["active_model"],
            },
        }
    finally:
        stop_runtime(receiver_group, runtime)


def run_inference_check(
    *,
    app_workspace: Path,
    session: ReplaySession,
    node_ports: dict[int, int],
    speedup: float,
    receiver_ready_seconds: float,
    prediction_settle_seconds: float,
) -> dict[str, Any]:
    engine: FingerprintEngine | None = None
    receiver_group: Any | None = None
    runtime: EngineRuntimeThread | None = None
    try:
        engine, receiver_group, runtime = start_runtime(
            app_workspace,
            receiver_ready_seconds=receiver_ready_seconds,
        )
        expected_label_key = session.expected_label_key
        replay_result: dict[str, Any] = {}
        replay_error: dict[str, BaseException] = {}

        def _sender() -> None:
            try:
                replay_result["stats"] = replay_session(
                    session,
                    host="127.0.0.1",
                    ports_by_node=node_ports,
                    speedup=speedup,
                )
            except BaseException as exc:  # noqa: BLE001
                replay_error["exc"] = exc

        sender = threading.Thread(target=_sender, name="raw-udp-replay-sender", daemon=True)
        sender.start()

        prediction_trace: list[dict[str, Any]] = []
        while sender.is_alive():
            snapshot = engine.snapshot()
            if snapshot["prediction"]["ready"]:
                prediction_trace.append(
                    build_prediction_trace_row(
                        snapshot,
                        expected_label_key=expected_label_key,
                    )
                )
            time.sleep(0.10)
        sender.join()
        if "exc" in replay_error:
            raise replay_error["exc"]

        settle_deadline = time.time() + max(0.0, prediction_settle_seconds)
        while time.time() < settle_deadline:
            snapshot = engine.snapshot()
            if snapshot["prediction"]["ready"]:
                prediction_trace.append(
                    build_prediction_trace_row(
                        snapshot,
                        expected_label_key=expected_label_key,
                    )
                )
            time.sleep(0.10)

        replay_stats = dict(replay_result["stats"])
        prediction_snapshot = wait_for_snapshot(
            engine,
            lambda snap: bool(snap["prediction"]["ready"]),
            timeout=max(4.0, replay_stats["duration_seconds"] + 3.0),
        )
        prediction_trace.append(
            build_prediction_trace_row(
                prediction_snapshot,
                expected_label_key=expected_label_key,
            )
        )
        prediction_summary = summarize_prediction(prediction_snapshot)
        matched = prediction_summary["best_label_key"] == expected_label_key
        trace_counts: dict[str, int] = {}
        expected_hits = 0
        peak_expected_probability = 0.0
        for row in prediction_trace:
            label_key = str(row["best_label_key"])
            trace_counts[label_key] = trace_counts.get(label_key, 0) + 1
            if label_key == expected_label_key:
                expected_hits += 1
            peak_expected_probability = max(
                peak_expected_probability,
                float(row["expected_probability"]),
            )
        majority_label_key = max(
            trace_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        return {
            "session_id": session.session_id,
            "label": session.display_label,
            "expected_label_key": expected_label_key,
            "expected_label_display": display_label(expected_label_key),
            "replay_duration_seconds": replay_stats["duration_seconds"],
            "packet_count": replay_stats["packet_count"],
            "prediction_ready": bool(prediction_snapshot["prediction"]["ready"]),
            "best_label_key": prediction_summary["best_label_key"],
            "best_label_display": prediction_summary["best_label_display"],
            "best_probability": prediction_summary["best_probability"],
            "matched": matched,
            "final_expected_probability": label_probability(
                prediction_snapshot,
                expected_label_key,
            ),
            "majority_label_key": majority_label_key,
            "majority_label_display": display_label(majority_label_key),
            "matched_majority": majority_label_key == expected_label_key,
            "matched_any": expected_hits > 0,
            "matched_trace_fraction": (
                float(expected_hits) / float(len(prediction_trace))
                if prediction_trace
                else 0.0
            ),
            "peak_expected_probability": peak_expected_probability,
            "trace_sample_count": len(prediction_trace),
            "top_probabilities": prediction_summary["top_probabilities"],
            "status_message": prediction_snapshot["status_message"],
        }
    finally:
        stop_runtime(receiver_group, runtime)


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    sessions = load_replay_sessions(args.raw_dir.resolve())
    baseline_session, cell_sessions = split_sessions(sessions)
    workspace_info = create_isolated_workspace(
        repo_root=REPO_ROOT,
        report_path=report_path,
        workspace_root=args.workspace_root,
        baseline_session=baseline_session,
        cell_sessions=cell_sessions,
        speedup=float(args.speedup),
        capture_padding_seconds=float(args.capture_padding_seconds),
    )
    app_workspace = Path(str(workspace_info["app_workspace"]))
    node_ports = {
        int(node_id): int(port)
        for node_id, port in dict(workspace_info["node_ports"]).items()
    }

    report: dict[str, Any] = {
        "started_at": now_iso(),
        "raw_dir": str(args.raw_dir.resolve()),
        "workspace": workspace_info,
        "speedup": float(args.speedup),
        "sessions": [
            {
                "session_id": session.session_id,
                "kind": session.kind,
                "label": session.display_label,
                "expected_label_key": session.expected_label_key,
                "duration_seconds": session.duration_seconds,
                "packet_count": len(session.packets),
                "path": str(session.path),
            }
            for session in [baseline_session, *cell_sessions]
        ],
    }

    try:
        training_report = run_training_pass(
            app_workspace=app_workspace,
            baseline_session=baseline_session,
            cell_sessions=cell_sessions,
            node_ports=node_ports,
            speedup=float(args.speedup),
            receiver_ready_seconds=float(args.receiver_ready_seconds),
        )
        report["training"] = training_report

        inference_rows: list[dict[str, Any]] = []
        for session in [baseline_session, *cell_sessions]:
            print_progress(f"Inference phase: replaying {session.display_label}.")
            inference_rows.append(
                run_inference_check(
                    app_workspace=app_workspace,
                    session=session,
                    node_ports=node_ports,
                    speedup=float(args.speedup),
                    receiver_ready_seconds=float(args.receiver_ready_seconds),
                    prediction_settle_seconds=float(args.prediction_settle_seconds),
                )
            )
        report["inference_checks"] = inference_rows
        matched_count = sum(1 for row in inference_rows if row["matched"])
        matched_majority_count = sum(
            1 for row in inference_rows if row["matched_majority"]
        )
        report["summary"] = {
            "matched_final_sessions": matched_count,
            "matched_majority_sessions": matched_majority_count,
            "total_sessions": len(inference_rows),
            "all_final_matched": matched_count == len(inference_rows),
            "all_majority_matched": matched_majority_count == len(inference_rows),
        }
        report["success"] = matched_majority_count == len(inference_rows)
        return_code = 0 if report["success"] else 1
        print_progress(
            f"Smoke test finished: final-match {matched_count}/{len(inference_rows)}, "
            f"majority-match {matched_majority_count}/{len(inference_rows)}."
        )
    except Exception as exc:  # noqa: BLE001
        report["success"] = False
        report["error"] = repr(exc)
        return_code = 1
        print_progress(f"Smoke test failed: {exc!r}")
    finally:
        report["finished_at"] = now_iso()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_progress(f"Wrote report to {report_path}")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
