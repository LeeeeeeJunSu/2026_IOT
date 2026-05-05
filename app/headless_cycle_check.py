from __future__ import annotations

import argparse
import json
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
            "Run a headless end-to-end smoke test against live ESP32 CSI traffic: "
            "baseline capture, per-cell learning, model training, and live inference."
        )
    )
    parser.add_argument("--cycles", type=int, default=2, help="How many end-to-end cycles to run.")
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=4.0,
        help="Per-cell Learn capture duration used during the smoke test.",
    )
    parser.add_argument(
        "--baseline-capture-seconds",
        type=float,
        default=3.0,
        help="Baseline capture duration used during the smoke test.",
    )
    parser.add_argument(
        "--baseline-delay-seconds",
        type=float,
        default=0.0,
        help="Baseline start delay used during the smoke test.",
    )
    parser.add_argument(
        "--live-smoothing-seconds",
        type=float,
        default=0.75,
        help="Probability smoothing used while validating live inference.",
    )
    parser.add_argument(
        "--switch-margin",
        type=float,
        default=0.05,
        help="Required margin before switching the displayed best cell.",
    )
    parser.add_argument(
        "--switch-delay-seconds",
        type=float,
        default=0.5,
        help="Minimum delay before switching the displayed best cell.",
    )
    parser.add_argument(
        "--packet-wait-seconds",
        type=float,
        default=10.0,
        help="How long to wait for all enabled nodes to show up on UDP.",
    )
    parser.add_argument(
        "--live-wait-seconds",
        type=float,
        default=8.0,
        help="How long to wait for live inference to become ready after training.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "headless_cycle_report.json",
        help="Where to write the final smoke-test report JSON.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
    poll_seconds: float = 0.25,
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


def override_runtime_config(args: argparse.Namespace, config_path: Path) -> str:
    original_text = config_path.read_text(encoding="utf-8")
    config = load_system_config(config_path)
    config.fingerprinting.capture_seconds = max(1.0, float(args.capture_seconds))
    config.fingerprinting.baseline_capture_seconds = max(
        1.0,
        float(args.baseline_capture_seconds),
    )
    config.fingerprinting.baseline_start_delay_seconds = max(
        0.0,
        float(args.baseline_delay_seconds),
    )
    config.fingerprinting.live_probability_smoothing_seconds = max(
        0.0,
        float(args.live_smoothing_seconds),
    )
    config.fingerprinting.best_cell_switch_margin = max(0.0, float(args.switch_margin))
    config.fingerprinting.best_cell_switch_delay_seconds = max(
        0.0,
        float(args.switch_delay_seconds),
    )
    save_system_config(config_path, config)
    return original_text


def print_progress(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def summarize_cells(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "grid_x": cell["grid_x"],
            "grid_y": cell["grid_y"],
            "capture_count": cell["capture_count"],
            "window_sample_count": cell["window_sample_count"],
            "trained": cell["trained"],
            "probability": round(float(cell["probability"]), 6),
            "is_best": bool(cell["is_best"]),
        }
        for cell in snapshot["cells"]
    ]


def wait_for_udp_ready(engine: FingerprintEngine, timeout: float) -> dict[str, Any]:
    required_nodes = set(engine.required_node_ids)
    print_progress(
        f"Waiting for UDP traffic from nodes {sorted(required_nodes)} for up to {timeout:.1f}s."
    )
    snapshot = wait_for_snapshot(
        engine,
        lambda snap: required_nodes.issubset({node["node_id"] for node in snap["nodes"]}),
        timeout=timeout,
    )
    print_progress(
        "UDP ready: "
        + ", ".join(
            f"node {node['node_id']}={node['packets_received']} packets"
            for node in snapshot["nodes"]
        )
    )
    return snapshot


def wait_for_capture_completion(
    engine: FingerprintEngine,
    *,
    timeout: float,
    grid_x: int,
    grid_y: int,
    previous_capture_count: int,
) -> dict[str, Any]:
    return wait_for_snapshot(
        engine,
        lambda snap: (
            not snap["capture"]["active"]
            and snapshot_cell(snap, grid_x, grid_y)["capture_count"] > previous_capture_count
        ),
        timeout=timeout,
    )


def wait_for_baseline_completion(
    engine: FingerprintEngine,
    *,
    timeout: float,
) -> dict[str, Any]:
    return wait_for_snapshot(
        engine,
        lambda snap: (not snap["capture"]["active"] and snap["baseline"]["ready"]),
        timeout=timeout,
    )


def wait_for_prediction_ready(
    engine: FingerprintEngine,
    *,
    timeout: float,
) -> dict[str, Any]:
    return wait_for_snapshot(
        engine,
        lambda snap: bool(snap["prediction"]["ready"]),
        timeout=timeout,
    )


def run_cycle(
    engine: FingerprintEngine,
    args: argparse.Namespace,
    cycle_index: int,
) -> dict[str, Any]:
    cycle_result: dict[str, Any] = {
        "cycle_index": cycle_index,
        "started_at": now_iso(),
        "captures": [],
        "model_checks": [],
    }
    wait_for_udp_ready(engine, args.packet_wait_seconds)

    print_progress(f"Cycle {cycle_index}: clearing previous baseline/training state.")
    engine.clear_baseline()

    print_progress(f"Cycle {cycle_index}: starting baseline capture.")
    engine.start_baseline_capture(reset_training=True)
    baseline_timeout = (
        float(args.baseline_delay_seconds)
        + float(args.baseline_capture_seconds)
        + 8.0
    )
    baseline_snapshot = wait_for_baseline_completion(engine, timeout=baseline_timeout)
    cycle_result["baseline"] = {
        "captured_nodes": baseline_snapshot["baseline"]["captured_nodes"],
        "required_nodes": baseline_snapshot["baseline"]["required_nodes"],
        "status_message": baseline_snapshot["status_message"],
    }
    print_progress(
        f"Cycle {cycle_index}: baseline ready with "
        f"{baseline_snapshot['baseline']['captured_nodes']}/"
        f"{baseline_snapshot['baseline']['required_nodes']} nodes."
    )

    capture_timeout = float(args.capture_seconds) + 10.0
    for grid_y in range(engine.grid_rows):
        for grid_x in range(engine.grid_cols):
            pre_snapshot = engine.snapshot()
            before = snapshot_cell(pre_snapshot, grid_x, grid_y)["capture_count"]
            print_progress(
                f"Cycle {cycle_index}: capturing cell ({grid_x + 1}, {grid_y + 1})."
            )
            engine.start_capture(grid_x, grid_y)
            post_snapshot = wait_for_capture_completion(
                engine,
                timeout=capture_timeout,
                grid_x=grid_x,
                grid_y=grid_y,
                previous_capture_count=before,
            )
            cell = snapshot_cell(post_snapshot, grid_x, grid_y)
            capture_summary = {
                "grid_x": grid_x,
                "grid_y": grid_y,
                "capture_count": cell["capture_count"],
                "window_sample_count": cell["window_sample_count"],
                "trained": cell["trained"],
                "status_message": post_snapshot["status_message"],
            }
            cycle_result["captures"].append(capture_summary)
            print_progress(
                f"Cycle {cycle_index}: cell ({grid_x + 1}, {grid_y + 1}) "
                f"now has {cell['window_sample_count']} windows across "
                f"{cell['capture_count']} captures."
            )

    print_progress(f"Cycle {cycle_index}: training models.")
    cycle_result["train_status"] = engine.train_models()
    train_snapshot = engine.snapshot()
    cycle_result["training"] = {
        "trained_model_count": train_snapshot["training"]["trained_model_count"],
        "available_models": list(train_snapshot["training"]["available_models"]),
        "dataset_samples": train_snapshot["training"]["dataset_samples"],
        "status_message": train_snapshot["status_message"],
    }
    print_progress(
        f"Cycle {cycle_index}: trained models={train_snapshot['training']['available_models']} "
        f"dataset_samples={train_snapshot['training']['dataset_samples']}."
    )

    for model_name in engine.MODEL_ORDER:
        print_progress(f"Cycle {cycle_index}: validating live inference with {model_name}.")
        engine.set_active_model(model_name)
        prediction_snapshot = wait_for_prediction_ready(
            engine,
            timeout=float(args.live_wait_seconds),
        )
        model_result = {
            "model_name": model_name,
            "best_cell_key": prediction_snapshot["prediction"]["best_cell_key"],
            "best_probability": round(
                float(prediction_snapshot["prediction"]["best_probability"]),
                6,
            ),
            "active_model": prediction_snapshot["prediction"]["active_model"],
            "ready": bool(prediction_snapshot["prediction"]["ready"]),
            "cells": summarize_cells(prediction_snapshot),
        }
        cycle_result["model_checks"].append(model_result)
        print_progress(
            f"Cycle {cycle_index}: {model_name} ready, "
            f"best={model_result['best_cell_key']} "
            f"prob={model_result['best_probability']:.3f}."
        )

    final_snapshot = engine.snapshot()
    cycle_result["finished_at"] = now_iso()
    cycle_result["final_snapshot"] = {
        "packet_count": final_snapshot["metrics"]["packet_count"],
        "prediction": {
            "ready": final_snapshot["prediction"]["ready"],
            "active_model": final_snapshot["prediction"]["active_model"],
            "best_cell_key": final_snapshot["prediction"]["best_cell_key"],
            "best_probability": round(
                float(final_snapshot["prediction"]["best_probability"]),
                6,
            ),
        },
        "baseline": {
            "ready": final_snapshot["baseline"]["ready"],
            "captured_nodes": final_snapshot["baseline"]["captured_nodes"],
        },
        "training": {
            "dataset_samples": final_snapshot["training"]["dataset_samples"],
            "trained_model_count": final_snapshot["training"]["trained_model_count"],
        },
        "nodes": [
            {
                "node_id": node["node_id"],
                "packets_received": node["packets_received"],
                "window_samples": node["window_samples"],
                "source": node["source"],
            }
            for node in final_snapshot["nodes"]
        ],
        "status_message": final_snapshot["status_message"],
    }
    return cycle_result


def main() -> int:
    args = parse_args()
    workspace = Path(__file__).resolve().parent
    config_path = workspace.parent / "Config" / "system_config.json"
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    original_config_text = override_runtime_config(args, config_path)
    engine: FingerprintEngine | None = None
    receiver: UdpReceiverThread | None = None
    runtime: EngineRuntimeThread | None = None
    report: dict[str, Any] = {
        "started_at": now_iso(),
        "workspace": str(workspace),
        "config_path": str(config_path),
        "test_config": {
            "cycles": args.cycles,
            "capture_seconds": args.capture_seconds,
            "baseline_capture_seconds": args.baseline_capture_seconds,
            "baseline_delay_seconds": args.baseline_delay_seconds,
            "live_smoothing_seconds": args.live_smoothing_seconds,
            "switch_margin": args.switch_margin,
            "switch_delay_seconds": args.switch_delay_seconds,
        },
        "cycles": [],
    }

    try:
        engine = FingerprintEngine(workspace)
        receiver = UdpReceiverThread(
            engine,
            engine.system_config.host.listen_host,
            engine.system_config.host.udp_port,
        )
        runtime = EngineRuntimeThread(engine)
        receiver.start()
        runtime.start()

        for cycle_index in range(1, max(1, int(args.cycles)) + 1):
            cycle_result = run_cycle(engine, args, cycle_index)
            report["cycles"].append(cycle_result)

        report["success"] = True
        report["finished_at"] = now_iso()
        print_progress("Smoke test completed successfully.")
        return_code = 0
    except Exception as exc:  # noqa: BLE001
        report["success"] = False
        report["finished_at"] = now_iso()
        report["error"] = repr(exc)
        print_progress(f"Smoke test failed: {exc!r}")
        return_code = 1
    finally:
        if receiver is not None:
            receiver.stop()
            receiver.join(timeout=1.5)
        if runtime is not None:
            runtime.stop()
            runtime.join(timeout=1.5)
        config_path.write_text(original_config_text, encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_progress(f"Wrote smoke-test report to {report_path}.")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
