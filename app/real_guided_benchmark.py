from __future__ import annotations

import argparse
import json
import shutil
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Guide a real ESP32 multi-cell capture session, train models, and benchmark "
            "live inference throughput on real traffic."
        )
    )
    parser.add_argument("--cols", type=int, default=4, help="Grid width for the guided run.")
    parser.add_argument("--rows", type=int, default=4, help="Grid height for the guided run.")
    parser.add_argument(
        "--captures-per-cell",
        type=int,
        default=2,
        help="How many Learn captures to collect for each real cell.",
    )
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=5.0,
        help="Per-cell capture duration.",
    )
    parser.add_argument(
        "--move-seconds",
        type=float,
        default=5.0,
        help="How much time to give the operator to move before each capture starts.",
    )
    parser.add_argument(
        "--baseline-capture-seconds",
        type=float,
        default=4.0,
        help="Baseline capture duration.",
    )
    parser.add_argument(
        "--baseline-move-seconds",
        type=float,
        default=8.0,
        help="How much time to leave the room before baseline capture begins.",
    )
    parser.add_argument(
        "--stress-duration-seconds",
        type=float,
        default=6.0,
        help="How long to measure live throughput per inference mode.",
    )
    parser.add_argument(
        "--baseline-retries",
        type=int,
        default=3,
        help="How many times to retry the real baseline capture before aborting.",
    )
    parser.add_argument(
        "--capture-retries",
        type=int,
        default=3,
        help="How many times to retry each cell capture before aborting.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "real_guided_benchmark_workspace",
        help="Workspace where benchmark datasets/models/logs are written.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "real_guided_benchmark_report.json",
        help="Where to write the benchmark report JSON.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_progress(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


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


def snapshot_cell(snapshot: dict[str, Any], grid_x: int, grid_y: int) -> dict[str, Any]:
    for cell in snapshot["cells"]:
        if cell["grid_x"] == grid_x and cell["grid_y"] == grid_y:
            return cell
    raise KeyError(f"Cell ({grid_x}, {grid_y}) not found in snapshot.")


def sleep_with_updates(seconds: float, message: str) -> None:
    remaining = max(0.0, float(seconds))
    if remaining <= 0.0:
        return
    while remaining > 0.0:
        print_progress(f"{message} | {remaining:.1f}s remaining")
        sleep_chunk = min(1.0, remaining)
        time.sleep(sleep_chunk)
        remaining -= sleep_chunk


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
    config.grid.cols = max(1, int(args.cols))
    config.grid.rows = max(1, int(args.rows))
    config.fingerprinting.capture_seconds = max(1.0, float(args.capture_seconds))
    config.fingerprinting.effective_packets_per_second = 10.0
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
    config.fingerprinting.baseline_start_delay_seconds = max(
        0.0,
        float(args.baseline_move_seconds),
    )
    config.fingerprinting.live_probability_smoothing_seconds = 0.75
    config.fingerprinting.best_cell_switch_margin = 0.05
    config.fingerprinting.best_cell_switch_delay_seconds = 0.5
    config.fingerprinting.capture_auto_extend_seconds = 6.0
    config.fingerprinting.capture_extend_step_seconds = 2.0
    config.fingerprinting.minimum_observed_windows = 8
    config.fingerprinting.minimum_observed_window_ratio = 0.2
    save_system_config(config_dir / "system_config.json", config)
    return workspace_root, config_dir / "system_config.json"


def wait_for_udp_ready(engine: FingerprintEngine, timeout: float = 10.0) -> dict[str, Any]:
    required_nodes = set(engine.required_node_ids)
    return wait_for_snapshot(
        engine,
        lambda snap: required_nodes.issubset({node["node_id"] for node in snap["nodes"]}),
        timeout=timeout,
    )


def wait_for_packet_progress(
    engine: FingerprintEngine,
    *,
    baseline_packet_count: int,
    timeout: float = 3.0,
    poll_seconds: float = 0.10,
) -> dict[str, Any]:
    return wait_for_snapshot(
        engine,
        lambda snap: int(snap["metrics"]["packet_count"]) > int(baseline_packet_count),
        timeout=timeout,
        poll_seconds=poll_seconds,
    )


def measure_throughput(
    engine: FingerprintEngine,
    *,
    model_name: str | None,
    duration_seconds: float,
) -> dict[str, Any]:
    before_switch_snapshot = engine.snapshot()
    baseline_packet_count = int(before_switch_snapshot["metrics"]["packet_count"])
    with engine.lock:
        engine.active_model_name = model_name
        engine._clear_prediction_locked()
        engine._schedule_inference_now_locked()

    label = model_name or "no_model"
    print_progress(
        f"Starting throughput measurement for {label}. "
        f"Keep the real setup unchanged for {duration_seconds:.1f}s."
    )
    try:
        warm_snapshot = wait_for_packet_progress(
            engine,
            baseline_packet_count=baseline_packet_count,
            timeout=max(2.0, float(duration_seconds)),
        )
    except TimeoutError:
        print_progress(
            f"No fresh packets were observed before measuring {label}; "
            "continuing with a best-effort throughput sample."
        )
        warm_snapshot = engine.snapshot()
    start_snapshot = warm_snapshot
    start_packets = int(start_snapshot["metrics"]["packet_count"])
    started_at = time.perf_counter()
    time.sleep(duration_seconds)
    elapsed = time.perf_counter() - started_at
    final_snapshot = engine.snapshot()
    processed_packets = int(final_snapshot["metrics"]["packet_count"]) - start_packets
    return {
        "model_name": label,
        "processed_packets": processed_packets,
        "elapsed_seconds": round(elapsed, 3),
        "processed_packets_per_second": round(processed_packets / max(elapsed, 1e-9), 1),
        "last_inference_duration_ms": round(
            float(final_snapshot["metrics"]["last_inference_duration_ms"]),
            3,
        ),
        "last_inference_age_ms": round(
            float(final_snapshot["metrics"]["last_inference_age_ms"] or 0.0),
            3,
        ),
        "inference_interval_seconds": round(
            float(final_snapshot["metrics"]["inference_interval_seconds"]),
            3,
        ),
        "prediction_ready": bool(final_snapshot["prediction"]["ready"]),
        "best_cell_key": final_snapshot["prediction"]["best_cell_key"],
        "best_probability": round(
            float(final_snapshot["prediction"]["best_probability"]),
            6,
        ),
    }


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
                "capture_count": int(cell["capture_count"]),
                "window_sample_count": int(cell["window_sample_count"]),
                "observed_window_count": int(cell.get("observed_window_count", 0)),
                "generated_window_count": int(cell.get("generated_window_count", 0)),
                "observed_window_ratio": round(
                    float(cell.get("observed_window_ratio", 0.0)),
                    3,
                ),
            }
            for cell in trained_cells
        ],
    }


def main() -> int:
    args = parse_args()
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_root, config_path = prepare_workspace(args)

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
            "move_seconds": args.move_seconds,
            "baseline_capture_seconds": args.baseline_capture_seconds,
            "baseline_move_seconds": args.baseline_move_seconds,
            "stress_duration_seconds": args.stress_duration_seconds,
            "baseline_retries": args.baseline_retries,
            "capture_retries": args.capture_retries,
        },
        "captures": [],
    }

    receiver.start()
    runtime.start()
    try:
        print_progress(
            f"Real guided benchmark workspace ready at {workspace_root}. "
            f"Grid={args.cols}x{args.rows}, captures_per_cell={args.captures_per_cell}."
        )
        print_progress(
            "Waiting for real ESP32 traffic on the configured UDP port. "
            "Make sure the nodes are powered and sending."
        )
        udp_snapshot = wait_for_udp_ready(engine, timeout=15.0)
        report["udp_ready"] = {
            "nodes": [int(node["node_id"]) for node in udp_snapshot["nodes"]],
            "packet_count": int(udp_snapshot["metrics"]["packet_count"]),
            "udp_status": udp_snapshot["udp_status"],
        }
        print_progress(
            f"UDP ready with nodes {[node['node_id'] for node in udp_snapshot['nodes']]}."
        )

        print_progress(
            "Baseline capture phase: leave the room now so the receiver sees an empty room."
        )
        baseline_timeout = (
            float(args.baseline_move_seconds) + float(args.baseline_capture_seconds) + 5.0
        )
        baseline_snapshot: dict[str, Any] | None = None
        baseline_error: Exception | None = None
        for attempt in range(1, max(1, int(args.baseline_retries)) + 1):
            try:
                print_progress(
                    f"Starting baseline attempt {attempt}/{max(1, int(args.baseline_retries))}."
                )
                engine.start_baseline_capture(reset_training=(attempt == 1))
                baseline_snapshot = wait_for_snapshot(
                    engine,
                    lambda snap: (not snap["capture"]["active"] and snap["baseline"]["ready"]),
                    timeout=baseline_timeout,
                )
                break
            except Exception as exc:  # noqa: BLE001
                baseline_error = exc
                failed_snapshot = engine.snapshot()
                print_progress(
                    f"Baseline attempt {attempt} failed: "
                    f"{failed_snapshot.get('status_message', repr(exc))}"
                )
                if attempt >= max(1, int(args.baseline_retries)):
                    raise
                sleep_with_updates(
                    2.0,
                    "Preparing to retry baseline capture",
                )
        if baseline_snapshot is None:
            raise RuntimeError(f"Baseline capture did not complete: {baseline_error!r}")
        report["baseline"] = {
            "captured_nodes": int(baseline_snapshot["baseline"]["captured_nodes"]),
            "required_nodes": int(baseline_snapshot["baseline"]["required_nodes"]),
            "status_message": baseline_snapshot["status_message"],
        }
        print_progress(
            f"Baseline completed with {baseline_snapshot['baseline']['captured_nodes']}/"
            f"{baseline_snapshot['baseline']['required_nodes']} nodes."
        )

        total_jobs = int(args.cols) * int(args.rows) * int(args.captures_per_cell)
        capture_index = 0
        for capture_round in range(int(args.captures_per_cell)):
            for grid_y in range(int(args.rows)):
                for grid_x in range(int(args.cols)):
                    capture_index += 1
                    sleep_with_updates(
                        float(args.move_seconds),
                        (
                            f"Move to real cell ({grid_x + 1}, {grid_y + 1}) "
                            f"for capture {capture_index}/{total_jobs}, round {capture_round + 1}"
                        ),
                    )
                    pre_snapshot = engine.snapshot()
                    before = snapshot_cell(pre_snapshot, grid_x, grid_y)["capture_count"]
                    post_snapshot: dict[str, Any] | None = None
                    capture_error: Exception | None = None
                    for attempt in range(1, max(1, int(args.capture_retries)) + 1):
                        try:
                            print_progress(
                                f"Starting Learn capture {capture_index}/{total_jobs} "
                                f"for cell ({grid_x + 1}, {grid_y + 1}), "
                                f"attempt {attempt}/{max(1, int(args.capture_retries))}. Hold still."
                            )
                            engine.start_capture(grid_x, grid_y)
                            post_snapshot = wait_for_snapshot(
                                engine,
                                lambda snap, gx=grid_x, gy=grid_y, prev=before: (
                                    not snap["capture"]["active"]
                                    and snapshot_cell(snap, gx, gy)["capture_count"] > prev
                                ),
                                timeout=float(args.capture_seconds) + 5.0,
                            )
                            break
                        except Exception as exc:  # noqa: BLE001
                            capture_error = exc
                            failed_snapshot = engine.snapshot()
                            print_progress(
                                f"Cell ({grid_x + 1}, {grid_y + 1}) attempt {attempt} failed: "
                                f"{failed_snapshot.get('status_message', repr(exc))}"
                            )
                            if attempt >= max(1, int(args.capture_retries)):
                                raise
                            sleep_with_updates(
                                2.0,
                                f"Retrying cell ({grid_x + 1}, {grid_y + 1}) capture",
                            )
                    if post_snapshot is None:
                        raise RuntimeError(
                            f"Cell ({grid_x + 1}, {grid_y + 1}) capture did not complete: "
                            f"{capture_error!r}"
                        )
                    cell_state = snapshot_cell(post_snapshot, grid_x, grid_y)
                    report["captures"].append(
                        {
                            "grid_x": grid_x,
                            "grid_y": grid_y,
                            "round": capture_round + 1,
                            "capture_count": int(cell_state["capture_count"]),
                            "window_sample_count": int(cell_state["window_sample_count"]),
                            "observed_window_count": int(
                                cell_state.get("observed_window_count", 0)
                            ),
                            "generated_window_count": int(
                                cell_state.get("generated_window_count", 0)
                            ),
                            "observed_window_ratio": round(
                                float(cell_state.get("observed_window_ratio", 0.0)),
                                3,
                            ),
                            "total_frames": int(cell_state["total_frames"]),
                        }
                    )
                    print_progress(
                        f"Cell ({grid_x + 1}, {grid_y + 1}) now has "
                        f"{cell_state['window_sample_count']} windows "
                        f"({cell_state.get('observed_window_count', 0)} observed) across "
                        f"{cell_state['capture_count']} captures."
                    )

        print_progress("Training models on the real multi-cell dataset.")
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

        report["stress"] = {"results": []}
        for model_name in [None, *engine.MODEL_ORDER]:
            result = measure_throughput(
                engine,
                model_name=model_name,
                duration_seconds=float(args.stress_duration_seconds),
            )
            report["stress"]["results"].append(result)
            print_progress(
                f"{result['model_name']}: {result['processed_packets_per_second']} pps, "
                f"inference={result['last_inference_duration_ms']} ms, "
                f"ready={result['prediction_ready']}."
            )

        report["success"] = True
        report["finished_at"] = now_iso()
        return_code = 0
    except Exception as exc:  # noqa: BLE001
        report["success"] = False
        report["finished_at"] = now_iso()
        report["error"] = repr(exc)
        print_progress(f"Real guided benchmark failed: {exc!r}")
        return_code = 1
    finally:
        receiver.stop()
        runtime.stop()
        receiver.join(timeout=1.5)
        runtime.join(timeout=1.5)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_progress(f"Wrote benchmark report to {report_path}.")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
