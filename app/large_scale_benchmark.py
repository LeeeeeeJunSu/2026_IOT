from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Config.config_loader import load_system_config, save_system_config

from app.core import FingerprintEngine
from app.receiver import UdpReceiverThread
from app.runtime import EngineRuntimeThread
from simulator.engine import Cell
from simulator.fingerprint import RoomFingerprintLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a large-scale synthetic benchmark over a bigger grid, train on many "
            "cells, and measure inference bottlenecks on the live pipeline."
        )
    )
    parser.add_argument("--cols", type=int, default=6, help="Grid width for the benchmark.")
    parser.add_argument("--rows", type=int, default=6, help="Grid height for the benchmark.")
    parser.add_argument(
        "--captures-per-cell",
        type=int,
        default=2,
        help="How many Learn captures to collect for each cell.",
    )
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=3.0,
        help="Per-cell capture duration during dataset collection.",
    )
    parser.add_argument(
        "--baseline-capture-seconds",
        type=float,
        default=3.0,
        help="Baseline capture duration.",
    )
    parser.add_argument(
        "--training-tick-hz",
        type=float,
        default=40.0,
        help="Synthetic sender tick rate while collecting training data.",
    )
    parser.add_argument(
        "--training-burst-size",
        type=int,
        default=2,
        help="How many frames per node to emit per training tick.",
    )
    parser.add_argument(
        "--stress-tick-hz",
        type=float,
        default=100.0,
        help="Synthetic sender tick rate during inference stress tests.",
    )
    parser.add_argument(
        "--stress-burst-size",
        type=int,
        default=10,
        help="How many frames per node to emit per stress-test tick.",
    )
    parser.add_argument(
        "--stress-duration-seconds",
        type=float,
        default=5.0,
        help="How long to stress each inference mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260419,
        help="Deterministic seed for the synthetic fingerprint generator.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "large_scale_benchmark_workspace",
        help="Workspace directory where benchmark config, dataset, and model artifacts are written.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "large_scale_benchmark_report.json",
        help="Path to the benchmark report JSON.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_progress(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def snapshot_cell(snapshot: dict[str, Any], grid_x: int, grid_y: int) -> dict[str, Any]:
    for cell in snapshot["cells"]:
        if cell["grid_x"] == grid_x and cell["grid_y"] == grid_y:
            return cell
    raise KeyError(f"Cell ({grid_x}, {grid_y}) not found in snapshot.")


def wait_for_snapshot(
    engine: FingerprintEngine,
    predicate,
    *,
    timeout: float,
    poll_seconds: float = 0.10,
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


def prepare_workspace(args: argparse.Namespace) -> tuple[Path, Path]:
    workspace_root = args.workspace
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    app_dir = workspace_root / "app"
    config_dir = workspace_root / "Config"
    data_dir = app_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    config = load_system_config(REPO_ROOT / "Config" / "system_config.json")
    config.host.listen_host = "0.0.0.0"
    config.host.target_ip = "127.0.0.1"
    config.host.udp_port = 5505
    config.grid.cols = max(1, int(args.cols))
    config.grid.rows = max(1, int(args.rows))
    config.fingerprinting.capture_seconds = max(1.0, float(args.capture_seconds))
    config.fingerprinting.window_sample_count = 9
    config.fingerprinting.window_step_samples = 1
    config.fingerprinting.smoothing_half_window = 0
    config.fingerprinting.window_seconds = (
        float(config.fingerprinting.window_sample_count)
        / config.fingerprinting.effective_packets_per_second
    )
    config.fingerprinting.window_step_seconds = (
        float(config.fingerprinting.window_step_samples)
        / config.fingerprinting.effective_packets_per_second
    )
    config.fingerprinting.baseline_capture_seconds = max(
        1.0,
        float(args.baseline_capture_seconds),
    )
    config.fingerprinting.baseline_start_delay_seconds = 0.0
    config.fingerprinting.live_probability_smoothing_seconds = 0.75
    config.fingerprinting.best_cell_switch_margin = 0.05
    config.fingerprinting.best_cell_switch_delay_seconds = 0.5
    config.simulation.tick_hz = max(1.0, float(args.training_tick_hz))
    config.simulation.frame_burst_size = max(1, int(args.training_burst_size))
    save_system_config(config_dir / "system_config.json", config)
    return workspace_root, config_dir / "system_config.json"


class SyntheticTrafficEmitter:
    def __init__(self, config_path: Path, seed: int) -> None:
        self.config = load_system_config(config_path)
        self.library = RoomFingerprintLibrary(config=self.config, seed=seed)
        self.destination = (self.config.host.target_ip, int(self.config.host.udp_port))
        self.sequence_by_node = {
            node.node_id: 0 for node in self.config.enabled_nodes()
        }
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self.socket.close()

    def emit_cell(
        self,
        *,
        cell: Cell,
        duration_seconds: float,
        tick_hz: float,
        frame_burst_size: int,
    ) -> None:
        tick_hz = max(1.0, float(tick_hz))
        frame_burst_size = max(1, int(frame_burst_size))
        tick_interval = 1.0 / tick_hz
        start = time.monotonic()
        next_tick = start
        tick_index = 0

        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= duration_seconds:
                break
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue

            for burst_index in range(frame_burst_size):
                for node in self.config.enabled_nodes():
                    sequence = self.sequence_by_node[node.node_id]
                    frame = self.library.build_frame(
                        node_id=node.node_id,
                        cell=cell,
                        sequence=sequence,
                        frame_index=tick_index,
                        burst_index=burst_index,
                    )
                    self.socket.sendto(frame.to_bytes(), self.destination)
                    self.sequence_by_node[node.node_id] = sequence + 1

            tick_index += 1
            next_tick += tick_interval
            if now - next_tick > tick_interval * 4:
                next_tick = now + tick_interval


def wait_for_udp_ready(engine: FingerprintEngine, timeout: float = 5.0) -> dict[str, Any]:
    required_nodes = set(engine.required_node_ids)
    return wait_for_snapshot(
        engine,
        lambda snap: required_nodes.issubset({node["node_id"] for node in snap["nodes"]}),
        timeout=timeout,
    )


def measure_inference_throughput(
    engine: FingerprintEngine,
    emitter: SyntheticTrafficEmitter,
    *,
    model_name: str | None,
    cell: Cell,
    stress_duration_seconds: float,
    tick_hz: float,
    frame_burst_size: int,
) -> dict[str, Any]:
    with engine.lock:
        engine.active_model_name = model_name
        engine._clear_prediction_locked()
        engine._schedule_inference_now_locked()

    time.sleep(0.5)
    start_snapshot = engine.snapshot()
    start_packets = int(start_snapshot["metrics"]["packet_count"])
    started_at = time.perf_counter()
    emitter.emit_cell(
        cell=cell,
        duration_seconds=stress_duration_seconds,
        tick_hz=tick_hz,
        frame_burst_size=frame_burst_size,
    )
    elapsed = time.perf_counter() - started_at
    final_snapshot = engine.snapshot()
    processed_packets = int(final_snapshot["metrics"]["packet_count"]) - start_packets
    result = {
        "model_name": model_name or "no_model",
        "processed_packets": processed_packets,
        "elapsed_seconds": round(elapsed, 3),
        "processed_packets_per_second": round(processed_packets / max(elapsed, 1e-9), 1),
        "inference_interval_seconds": round(
            float(final_snapshot["metrics"]["inference_interval_seconds"]),
            3,
        ),
        "last_inference_duration_ms": round(
            float(final_snapshot["metrics"]["last_inference_duration_ms"]),
            3,
        ),
        "last_inference_age_ms": round(
            float(final_snapshot["metrics"]["last_inference_age_ms"] or 0.0),
            3,
        ),
        "inference_cycle_count": int(final_snapshot["metrics"]["inference_cycle_count"]),
        "prediction_ready": bool(final_snapshot["prediction"]["ready"]),
        "best_cell_key": final_snapshot["prediction"]["best_cell_key"],
        "best_probability": round(
            float(final_snapshot["prediction"]["best_probability"]),
            6,
        ),
    }
    return result


def summarize_dataset(snapshot: dict[str, Any]) -> dict[str, Any]:
    trained_cells = [cell for cell in snapshot["cells"] if cell["trained"]]
    return {
        "trained_cells": len(trained_cells),
        "total_cells": int(snapshot["training"]["total_cells"]),
        "dataset_samples": int(snapshot["training"]["dataset_samples"]),
        "per_cell_samples": [
            {
                "grid_x": int(cell["grid_x"]),
                "grid_y": int(cell["grid_y"]),
                "window_sample_count": int(cell["window_sample_count"]),
                "capture_count": int(cell["capture_count"]),
            }
            for cell in trained_cells
        ],
    }


def main() -> int:
    args = parse_args()
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    workspace_root, config_path = prepare_workspace(args)
    emitter = SyntheticTrafficEmitter(config_path, seed=args.seed)
    engine = FingerprintEngine(workspace_root / "app")
    receiver = UdpReceiverThread(
        engine,
        engine.system_config.host.listen_host,
        engine.system_config.host.udp_port,
    )
    runtime = EngineRuntimeThread(engine)

    report: dict[str, Any] = {
        "started_at": now_iso(),
        "workspace": str(workspace_root),
        "config_path": str(config_path),
        "config": {
            "cols": args.cols,
            "rows": args.rows,
            "captures_per_cell": args.captures_per_cell,
            "capture_seconds": args.capture_seconds,
            "baseline_capture_seconds": args.baseline_capture_seconds,
            "training_tick_hz": args.training_tick_hz,
            "training_burst_size": args.training_burst_size,
            "stress_tick_hz": args.stress_tick_hz,
            "stress_burst_size": args.stress_burst_size,
            "stress_duration_seconds": args.stress_duration_seconds,
        },
        "captures": [],
    }

    receiver.start()
    runtime.start()
    try:
        print_progress(
            f"Benchmark workspace ready at {workspace_root}. "
            f"Grid={args.cols}x{args.rows}, captures_per_cell={args.captures_per_cell}."
        )

        print_progress("Priming UDP receiver with a short synthetic burst.")
        emitter.emit_cell(
            cell=Cell(0, 0),
            duration_seconds=0.6,
            tick_hz=args.training_tick_hz,
            frame_burst_size=args.training_burst_size,
        )
        udp_snapshot = wait_for_udp_ready(engine)
        report["udp_ready"] = {
            "nodes": [int(node["node_id"]) for node in udp_snapshot["nodes"]],
            "packet_count": int(udp_snapshot["metrics"]["packet_count"]),
            "udp_status": udp_snapshot["udp_status"],
        }
        print_progress(
            f"UDP ready with nodes {[node['node_id'] for node in udp_snapshot['nodes']]}."
        )

        print_progress("Capturing baseline.")
        engine.start_baseline_capture(reset_training=True)
        emitter.emit_cell(
            cell=Cell(0, 0),
            duration_seconds=float(args.baseline_capture_seconds) + 0.25,
            tick_hz=args.training_tick_hz,
            frame_burst_size=args.training_burst_size,
        )
        baseline_snapshot = wait_for_snapshot(
            engine,
            lambda snap: (not snap["capture"]["active"] and snap["baseline"]["ready"]),
            timeout=float(args.baseline_capture_seconds) + 5.0,
        )
        report["baseline"] = {
            "captured_nodes": int(baseline_snapshot["baseline"]["captured_nodes"]),
            "required_nodes": int(baseline_snapshot["baseline"]["required_nodes"]),
            "status_message": baseline_snapshot["status_message"],
        }
        print_progress(
            f"Baseline ready with {baseline_snapshot['baseline']['captured_nodes']}/"
            f"{baseline_snapshot['baseline']['required_nodes']} nodes."
        )

        total_capture_jobs = int(args.cols) * int(args.rows) * int(args.captures_per_cell)
        capture_index = 0
        for capture_round in range(int(args.captures_per_cell)):
            for grid_y in range(int(args.rows)):
                for grid_x in range(int(args.cols)):
                    capture_index += 1
                    pre_snapshot = engine.snapshot()
                    before = snapshot_cell(pre_snapshot, grid_x, grid_y)["capture_count"]
                    print_progress(
                        f"Collecting capture {capture_index}/{total_capture_jobs} "
                        f"for cell ({grid_x + 1}, {grid_y + 1}), round {capture_round + 1}."
                    )
                    engine.start_capture(grid_x, grid_y)
                    emitter.emit_cell(
                        cell=Cell(grid_x, grid_y),
                        duration_seconds=float(args.capture_seconds) + 0.25,
                        tick_hz=args.training_tick_hz,
                        frame_burst_size=args.training_burst_size,
                    )
                    post_snapshot = wait_for_snapshot(
                        engine,
                        lambda snap, gx=grid_x, gy=grid_y, prev=before: (
                            not snap["capture"]["active"]
                            and snapshot_cell(snap, gx, gy)["capture_count"] > prev
                        ),
                        timeout=float(args.capture_seconds) + 5.0,
                    )
                    cell_state = snapshot_cell(post_snapshot, grid_x, grid_y)
                    report["captures"].append(
                        {
                            "grid_x": grid_x,
                            "grid_y": grid_y,
                            "round": capture_round + 1,
                            "window_sample_count": int(cell_state["window_sample_count"]),
                            "capture_count": int(cell_state["capture_count"]),
                            "total_frames": int(cell_state["total_frames"]),
                        }
                    )

        print_progress("Training models on the large synthetic dataset.")
        train_started = time.perf_counter()
        train_status = engine.train_models()
        train_elapsed = time.perf_counter() - train_started
        trained_snapshot = engine.snapshot()
        report["training"] = {
            "status_message": train_status,
            "elapsed_seconds": round(train_elapsed, 3),
            "trained_model_count": int(trained_snapshot["training"]["trained_model_count"]),
            "available_models": list(trained_snapshot["training"]["available_models"]),
            "dataset": summarize_dataset(trained_snapshot),
        }
        print_progress(
            f"Training completed in {train_elapsed:.2f}s with "
            f"{trained_snapshot['training']['dataset_samples']} samples."
        )

        stress_cell = Cell(args.cols - 1, args.rows - 1)
        report["stress"] = {
            "cell": {"x": stress_cell.x, "y": stress_cell.y},
            "results": [],
        }
        for model_name in [None, *engine.MODEL_ORDER]:
            label = model_name or "no_model"
            print_progress(f"Running inference stress test for {label}.")
            result = measure_inference_throughput(
                engine,
                emitter,
                model_name=model_name,
                cell=stress_cell,
                stress_duration_seconds=float(args.stress_duration_seconds),
                tick_hz=float(args.stress_tick_hz),
                frame_burst_size=int(args.stress_burst_size),
            )
            report["stress"]["results"].append(result)
            print_progress(
                f"{label}: {result['processed_packets_per_second']} pps, "
                f"inference={result['last_inference_duration_ms']} ms."
            )

        report["success"] = True
        report["finished_at"] = now_iso()
        return_code = 0
    except Exception as exc:  # noqa: BLE001
        report["success"] = False
        report["finished_at"] = now_iso()
        report["error"] = repr(exc)
        print_progress(f"Benchmark failed: {exc!r}")
        return_code = 1
    finally:
        receiver.stop()
        runtime.stop()
        receiver.join(timeout=1.5)
        runtime.join(timeout=1.5)
        emitter.close()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_progress(f"Wrote benchmark report to {report_path}.")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
