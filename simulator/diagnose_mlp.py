from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from statistics import mean
from unittest.mock import patch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from Config.config_loader import dump_system_config, load_system_config
from app.core import FingerprintEngine
from simulator.fingerprint import RoomFingerprintLibrary
from simulator.pathing import Cell, build_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulator.diagnose_mlp",
        description="Run an end-to-end simulated training and inference diagnostic.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to system_config.json. Defaults to Config/system_config.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260410,
        help="Deterministic simulator seed.",
    )
    parser.add_argument(
        "--train-order",
        type=str,
        default="row_major",
        help="Cell order for simulated training. Default: row_major.",
    )
    parser.add_argument(
        "--infer-seconds",
        type=float,
        default=4.0,
        help="Inference dwell time per cell after training.",
    )
    parser.add_argument(
        "--frame-burst-size",
        type=int,
        default=3,
        help="Override burst size used during the diagnostic run.",
    )
    return parser


def _ensure_workspace(root: Path, config_path: Path) -> Path:
    app_dir = root / "app"
    data_dir = app_dir / "data"
    config_dir = root / "Config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    config = load_system_config(config_path)
    dumped = dump_system_config(config)
    (config_dir / "system_config.json").write_text(
        json.dumps(dumped, indent=2),
        encoding="utf-8",
    )
    (data_dir / "fingerprints.json").write_text(
        json.dumps({"cells": {}}, indent=2),
        encoding="utf-8",
    )
    return app_dir


def _emit_cell_frames(
    engine: FingerprintEngine,
    library: RoomFingerprintLibrary,
    cell: Cell,
    start_time: float,
    duration_seconds: float,
    tick_hz: float,
    frame_burst_size: int,
    sequence_by_node: dict[int, int],
    frame_index_start: int,
    collect_predictions: bool = False,
) -> tuple[float, int, list[dict[str, object]]]:
    tick_interval = 1.0 / max(1.0, float(tick_hz))
    time_cursor = start_time
    frame_index = frame_index_start
    nodes = [node.node_id for node in engine.system_config.enabled_nodes()]
    predictions: list[dict[str, object]] = []

    while time_cursor < start_time + duration_seconds - 1e-9:
        for burst_index in range(max(1, int(frame_burst_size))):
            burst_time = time_cursor + burst_index * 1e-4
            for node_id in nodes:
                frame = library.build_frame(
                    node_id=node_id,
                    cell=cell,
                    sequence=sequence_by_node[node_id],
                    frame_index=frame_index,
                    burst_index=burst_index,
                )
                with patch("app.core.time.time", return_value=burst_time):
                    engine.process_packet(
                        frame.to_bytes(),
                        f"sim://node-{node_id}",
                    )
                sequence_by_node[node_id] += 1

        if collect_predictions and (time_cursor - start_time + tick_interval) >= engine.window_seconds:
            probe_time = time_cursor + tick_interval
            with patch("app.core.time.time", return_value=probe_time):
                snapshot = engine.snapshot()
            predictions.append(
                {
                    "expected": engine.cell_key(cell.x, cell.y),
                    "predicted": snapshot["prediction"]["best_cell_key"],
                    "probability": float(snapshot["prediction"]["best_probability"]),
                }
            )

        frame_index += 1
        time_cursor += tick_interval

    return time_cursor, frame_index, predictions


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = (
        args.config
        if args.config is not None
        else Path(__file__).resolve().parents[1] / "Config" / "system_config.json"
    )
    config = load_system_config(config_path)
    train_cells = build_path(config.grid.cols, config.grid.rows, args.train_order)

    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = _ensure_workspace(Path(temp_dir), config_path)
        engine = FingerprintEngine(workspace)
        library = RoomFingerprintLibrary(config=config, seed=args.seed)
        sequence_by_node = {
            node.node_id: 0 for node in config.enabled_nodes()
        }
        frame_index = 0
        simulated_time = 1_000.0

        print(
            f"training cells={[(cell.x, cell.y) for cell in train_cells]} "
            f"capture={engine.capture_seconds:.2f}s window={engine.window_seconds:.2f}s "
            f"burst={args.frame_burst_size}"
        )

        for cell in train_cells:
            with patch("app.core.time.time", return_value=simulated_time):
                engine.start_capture(cell.x, cell.y)
            simulated_time, frame_index, _ = _emit_cell_frames(
                engine=engine,
                library=library,
                cell=cell,
                start_time=simulated_time,
                duration_seconds=engine.capture_seconds,
                tick_hz=config.simulation.tick_hz,
                frame_burst_size=args.frame_burst_size,
                sequence_by_node=sequence_by_node,
                frame_index_start=frame_index,
                collect_predictions=False,
            )
            simulated_time += 0.05
            with patch("app.core.time.time", return_value=simulated_time):
                snapshot = engine.snapshot()
            dataset_cell = next(
                item
                for item in snapshot["cells"]
                if item["cell_key"] == engine.cell_key(cell.x, cell.y)
            )
            print(
                f"trained cell=({cell.x},{cell.y}) "
                f"samples={dataset_cell['window_sample_count']} "
                f"frames={dataset_cell['total_frames']}"
            )

        with patch("app.core.time.time", return_value=simulated_time + 0.1):
            engine.train_models()
        with patch("app.core.time.time", return_value=simulated_time + 0.1):
            trained_snapshot = engine.snapshot()
        print(
            f"model_ready={trained_snapshot['training']['model_ready']} "
            f"dataset_samples={trained_snapshot['training']['dataset_samples']}"
        )

        all_predictions: list[dict[str, object]] = []
        for cell in train_cells:
            simulated_time += 0.5
            simulated_time, frame_index, predictions = _emit_cell_frames(
                engine=engine,
                library=library,
                cell=cell,
                start_time=simulated_time,
                duration_seconds=max(args.infer_seconds, engine.window_seconds * 2.0),
                tick_hz=config.simulation.tick_hz,
                frame_burst_size=args.frame_burst_size,
                sequence_by_node=sequence_by_node,
                frame_index_start=frame_index,
                collect_predictions=True,
            )
            matches = [item for item in predictions if item["predicted"] == item["expected"]]
            accuracy = (len(matches) / len(predictions)) if predictions else 0.0
            mean_prob = mean(item["probability"] for item in predictions) if predictions else 0.0
            print(
                f"infer cell=({cell.x},{cell.y}) "
                f"checks={len(predictions)} accuracy={accuracy:.3f} "
                f"mean_best_prob={mean_prob:.3f}"
            )
            all_predictions.extend(predictions)

        overall = (
            sum(1 for item in all_predictions if item["predicted"] == item["expected"])
            / len(all_predictions)
            if all_predictions
            else 0.0
        )
        print(f"overall_accuracy={overall:.3f} checks={len(all_predictions)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
