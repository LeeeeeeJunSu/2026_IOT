from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import FingerprintEngine, ModelMetadata
from app.protocol import ACTIVE_SUBCARRIER_COUNT, FeatureFrame
from app.raw_training import DEFAULT_QUANTILES
from app.storage import save_fingerprint_store, save_pickle_store


MODEL_NAME = "VariableNodeAggregateExtraTrees"


@dataclass(frozen=True)
class GtSession:
    session_id: str
    gt_location: int
    path: Path
    frames_by_node: dict[int, list[FeatureFrame]]


def build_parser() -> argparse.ArgumentParser:
    workspace_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train the live app model from app/raw_data ground-truth JSONL captures."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=workspace_root,
        help="Repository root. Defaults to the current project root.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=workspace_root / "app" / "raw_data",
        help="Directory containing GT raw JSONL captures.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=workspace_root / "app" / "data" / "gt_training_report.json",
        help="Training report output path.",
    )
    parser.add_argument("--trees", type=int, default=320, help="ExtraTrees estimator count.")
    parser.add_argument("--random-state", type=int, default=42, help="ExtraTrees random seed.")
    parser.add_argument("--n-jobs", type=int, default=1, help="ExtraTrees parallelism.")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Chronological tail fraction of each capture session used for test reporting.",
    )
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=27,
        help="Windows skipped between train and test chunks to reduce overlap leakage.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_gt_model(
        workspace_root=args.workspace_root,
        raw_dir=args.raw_dir,
        report_path=args.report_path,
        trees=max(50, int(args.trees)),
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
        test_ratio=float(args.test_ratio),
        purge_gap=max(0, int(args.purge_gap)),
    )
    return 0


def train_gt_model(
    *,
    workspace_root: Path,
    raw_dir: Path,
    report_path: Path,
    trees: int = 320,
    random_state: int = 42,
    n_jobs: int = 1,
    test_ratio: float = 0.20,
    purge_gap: int = 27,
) -> dict[str, Any]:
    workspace_root = Path(workspace_root)
    app_root = workspace_root / "app"
    engine = FingerprintEngine(app_root)
    sessions = load_gt_sessions(raw_dir)
    if not sessions:
        raise RuntimeError(f"No GT sessions found in {raw_dir}. Capture data with app.capture_gt first.")

    required_node_ids = engine.required_node_ids
    baseline_by_node, baseline_counts, baseline_source = build_baseline(
        sessions,
        required_node_ids,
    )
    engine.empty_room_baseline_by_node = baseline_by_node
    engine.empty_room_baseline_counts = baseline_counts

    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    records: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    with engine.lock:
        for session in sessions:
            features = build_session_aggregate_features(engine, session)
            if not features:
                raise RuntimeError(
                    f"No training windows could be generated from {session.path}. "
                    "Check capture duration and active node coverage."
                )
            label = str(session.gt_location)
            first_index = len(X_rows)
            X_rows.extend(features)
            y_rows.extend([label] * len(features))
            records.extend(
                {
                    "session_id": session.session_id,
                    "gt_location": session.gt_location,
                    "window_index": window_index // max(1, len(required_node_ids)),
                }
                for window_index in range(len(features))
            )
            session_summaries.append(
                {
                    "session_id": session.session_id,
                    "gt_location": session.gt_location,
                    "path": str(session.path),
                    "aggregate_windows": len(features),
                    "first_sample_index": first_index,
                    "last_sample_index": len(X_rows) - 1,
                    "packets_by_node": {
                        str(node_id): len(session.frames_by_node.get(node_id, []))
                        for node_id in required_node_ids
                    },
                }
            )

    if not X_rows:
        raise RuntimeError("No training samples were generated.")
    labels = sorted({str(label) for label in y_rows}, key=int)
    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows)
    train_indices, test_indices, split_summary = build_temporal_test_split(
        records,
        test_ratio=max(0.05, min(0.45, float(test_ratio))),
        purge_gap=max(0, int(purge_gap)),
    )

    eval_model = build_model(
        trees=trees,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    eval_model.fit(X[train_indices], y[train_indices])
    train_predictions = eval_model.predict(X[train_indices])
    test_predictions = eval_model.predict(X[test_indices])
    eval_metrics = {
        "split_kind": "chronological_session_tail",
        "test_ratio": max(0.05, min(0.45, float(test_ratio))),
        "purge_gap_windows": max(0, int(purge_gap)),
        "train_sample_count": int(train_indices.size),
        "test_sample_count": int(test_indices.size),
        "train_accuracy": float(accuracy_score(y[train_indices], train_predictions)),
        "train_macro_f1": float(f1_score(y[train_indices], train_predictions, average="macro")),
        "test_accuracy": float(accuracy_score(y[test_indices], test_predictions)),
        "test_macro_f1": float(f1_score(y[test_indices], test_predictions, average="macro")),
        "test_confusion_matrix": confusion_matrix(
            y[test_indices],
            test_predictions,
            labels=labels,
        ).tolist(),
        "split_summary": split_summary,
    }

    model = build_model(
        trees=trees,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X, y)
    predictions = model.predict(X)
    metrics = {
        "train_accuracy": float(accuracy_score(y, predictions)),
        "train_macro_f1": float(f1_score(y, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
    }

    metadata = ModelMetadata(
        model_key=MODEL_NAME,
        trained_at=time.time(),
        window_seconds=engine.window_seconds,
        window_step_seconds=engine.window_step_seconds,
        feature_signature=engine.feature_signature,
        node_ids=required_node_ids,
        input_size=engine.aggregate_feature_size,
        sample_count=len(X_rows),
        class_labels=labels,
        summary=(
            "GT numeric variable-node aggregate classifier, labels 0..6, "
            f"n_estimators={trees}, random_state={random_state}"
        ),
    )
    save_pickle_store(
        engine.model_path,
        {
            "active_model": MODEL_NAME,
            "models": {
                MODEL_NAME: {
                    "pipeline": model,
                    "metadata": {
                        "model_key": metadata.model_key,
                        "trained_at": metadata.trained_at,
                        "window_seconds": metadata.window_seconds,
                        "window_step_seconds": metadata.window_step_seconds,
                        "feature_signature": metadata.feature_signature,
                        "node_ids": metadata.node_ids,
                        "input_size": metadata.input_size,
                        "sample_count": metadata.sample_count,
                        "class_labels": metadata.class_labels,
                        "summary": metadata.summary,
                    },
                }
            },
        },
    )
    save_training_baseline(engine)

    report = {
        "created_at_unix": metadata.trained_at,
        "raw_dir": str(Path(raw_dir).resolve()),
        "model_path": str(engine.model_path.resolve()),
        "fingerprint_path": str(engine.fingerprint_path.resolve()),
        "labels": labels,
        "node_ids": required_node_ids,
        "sample_count": len(X_rows),
        "feature_dim": int(X.shape[1]),
        "feature_generation": "variable_node_aggregate_live_rolling_horizon",
        "live_preprocess_horizon_windows": engine.LIVE_PREPROCESS_HORIZON_WINDOWS,
        "live_preprocess_horizon_samples": max(
            engine.window_sample_count + engine.smoothing_half_window * 2,
            engine.window_sample_count * engine.LIVE_PREPROCESS_HORIZON_WINDOWS,
        ),
        "window_sample_count": engine.window_sample_count,
        "window_step_samples": engine.window_step_samples,
        "effective_packets_per_second": engine.effective_packets_per_second,
        "baseline_counts": {str(key): value for key, value in baseline_counts.items()},
        "baseline_source": baseline_source,
        "sessions": session_summaries,
        "metrics": metrics,
        "holdout_metrics": eval_metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== GT TRAINING ===")
    print(f"labels={','.join(labels)}")
    print(f"samples={len(X_rows)} feature_dim={int(X.shape[1])}")
    print(f"train_accuracy={metrics['train_accuracy']:.4f}")
    print(f"train_macro_f1={metrics['train_macro_f1']:.4f}")
    print()
    print("=== TEMPORAL HOLDOUT ===")
    print(f"train_samples={eval_metrics['train_sample_count']}")
    print(f"test_samples={eval_metrics['test_sample_count']}")
    print(f"test_accuracy={eval_metrics['test_accuracy']:.4f}")
    print(f"test_macro_f1={eval_metrics['test_macro_f1']:.4f}")
    print(f"model_saved_to={engine.model_path.resolve()}")
    print(f"report_saved_to={report_path.resolve()}")
    return report


def build_model(
    *,
    trees: int,
    random_state: int,
    n_jobs: int,
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trees,
        random_state=random_state,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=2,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
    )


def build_temporal_test_split(
    records: list[dict[str, Any]],
    *,
    test_ratio: float,
    purge_gap: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    by_session: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_session[str(record["session_id"])].append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []
    split_summary: list[dict[str, Any]] = []
    for session_id, indices in by_session.items():
        ordered = sorted(indices, key=lambda item: int(records[item]["window_index"]))
        total = len(ordered)
        test_count = max(1, int(round(total * test_ratio)))
        split_at = max(1, total - test_count)
        train_end = max(1, split_at - purge_gap)
        test_start = min(total - 1, split_at + purge_gap)
        local_train = ordered[:train_end]
        local_test = ordered[test_start:]
        if not local_train or not local_test:
            midpoint = total // 2
            local_train = ordered[:max(1, midpoint)]
            local_test = ordered[min(total - 1, midpoint):]
        train_indices.extend(local_train)
        test_indices.extend(local_test)
        split_summary.append(
            {
                "session_id": session_id,
                "gt_location": records[ordered[0]]["gt_location"],
                "total_windows": total,
                "train_windows": len(local_train),
                "test_windows": len(local_test),
                "purge_gap_windows": purge_gap,
            }
        )

    if not train_indices or not test_indices:
        raise RuntimeError("Could not build train/test split from GT sessions.")
    return (
        np.asarray(sorted(train_indices), dtype=int),
        np.asarray(sorted(test_indices), dtype=int),
        split_summary,
    )


def load_gt_sessions(raw_dir: str | Path) -> list[GtSession]:
    sessions: list[GtSession] = []
    for path in sorted(Path(raw_dir).glob("*.jsonl")):
        if path.name == "sessions.jsonl":
            continue
        session_id = path.stem
        gt_location: int | None = None
        frames_by_node: dict[int, list[FeatureFrame]] = defaultdict(list)
        session_ended = False
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "gt_location" in record:
                    gt_location = int(record["gt_location"])
                if record.get("record_type") == "session_end":
                    session_ended = True
                if record.get("record_type") != "packet" or not record.get("valid_adr018"):
                    continue
                frame = feature_frame_from_record(record)
                if frame is not None:
                    frames_by_node[frame.node_id].append(frame)
                    session_id = str(record.get("session_id", session_id))
        if gt_location is None or not session_ended:
            continue
        if gt_location < 0 or gt_location > 6:
            raise RuntimeError(f"{path} has invalid gt_location={gt_location}; expected 0..6.")
        sessions.append(
            GtSession(
                session_id=session_id,
                gt_location=gt_location,
                path=path.resolve(),
                frames_by_node={key: list(value) for key, value in frames_by_node.items()},
            )
        )
    return sessions


def feature_frame_from_record(record: dict[str, Any]) -> FeatureFrame | None:
    vector = record.get("feature_vector")
    if not isinstance(vector, list) or len(vector) < ACTIVE_SUBCARRIER_COUNT:
        return None
    rssi_dbm = float(record.get("rssi_dbm", 0.0))
    noise_floor_dbm = float(record.get("noise_floor_dbm", 0.0))
    return FeatureFrame(
        node_id=int(record.get("node_id", 0)),
        source=str(record.get("source", "")),
        captured_at=float(record.get("captured_at_unix", 0.0)),
        sequence=int(record.get("sequence", 0)),
        n_subcarriers=ACTIVE_SUBCARRIER_COUNT,
        rssi_dbm=rssi_dbm,
        noise_floor_dbm=noise_floor_dbm,
        snr_db=float(record.get("snr_db", rssi_dbm - noise_floor_dbm)),
        amplitude_mean=float(record.get("amplitude_mean", 0.0)),
        amplitude_std=float(record.get("amplitude_std", 0.0)),
        amplitude_rms=float(record.get("amplitude_rms", 0.0)),
        amplitude_p90=float(record.get("amplitude_p90", 0.0)),
        gradient_mean=float(record.get("gradient_mean", 0.0)),
        phase_step_std=float(record.get("phase_step_std", 0.0)),
        feature_vector=[float(value) for value in vector[:ACTIVE_SUBCARRIER_COUNT]],
    )


def build_baseline(
    sessions: list[GtSession],
    required_node_ids: list[int],
) -> tuple[dict[int, list[float]], dict[int, int], str]:
    baseline_sessions = [session for session in sessions if session.gt_location == 0]
    baseline_source = "gt_0"
    if not baseline_sessions:
        baseline_sessions = sessions
        baseline_source = "all_gt_sessions"

    baseline_by_node: dict[int, list[float]] = {}
    baseline_counts: dict[int, int] = {}
    for node_id in required_node_ids:
        vectors = [
            frame.feature_vector
            for session in baseline_sessions
            for frame in session.frames_by_node.get(node_id, [])
        ]
        if not vectors:
            continue
        baseline_by_node[node_id] = np.asarray(vectors, dtype=np.float32).mean(axis=0).tolist()
        baseline_counts[node_id] = len(vectors)
    if not baseline_by_node:
        raise RuntimeError("GT captures do not contain any baseline node data.")
    return baseline_by_node, baseline_counts, baseline_source


def build_session_features(
    engine: FingerprintEngine,
    session: GtSession,
) -> list[list[float]]:
    all_frames = [frame for frames in session.frames_by_node.values() for frame in frames]
    if not all_frames:
        return []
    start_time = min(frame.captured_at for frame in all_frames)
    end_time = max(frame.captured_at for frame in all_frames)
    horizon_sample_count = max(
        engine.window_sample_count + engine.smoothing_half_window * 2,
        engine.window_sample_count * engine.LIVE_PREPROCESS_HORIZON_WINDOWS,
    )
    required_recent_samples = engine.window_sample_count + engine.smoothing_half_window * 2
    (
        aligned_vectors_by_node,
        aligned_scalars_by_node,
        observed_by_node,
        total_slots,
    ) = build_session_rows_for_live_horizons(
        engine,
        session,
        start_time=start_time,
        end_time=end_time,
    )
    if total_slots < horizon_sample_count:
        return []

    if engine.smoothing_half_window <= 0 and engine.scalar_smoothing_half_window <= 0:
        return build_vectorized_live_horizon_features(
            engine,
            aligned_vectors_by_node=aligned_vectors_by_node,
            aligned_scalars_by_node=aligned_scalars_by_node,
            observed_by_node=observed_by_node,
            horizon_sample_count=horizon_sample_count,
            required_recent_samples=required_recent_samples,
        )

    rows: list[list[float]] = []
    for end_slot in range(
        horizon_sample_count,
        total_slots + 1,
        engine.window_step_samples,
    ):
        start_slot = end_slot - horizon_sample_count
        if count_valid_horizon_slots(
            observed_by_node,
            required_node_ids=engine.required_node_ids,
            start_slot=start_slot,
            end_slot=end_slot,
        ) < required_recent_samples:
            continue

        row: list[float] = []
        for node_id in engine.required_node_ids:
            per_node_features = engine._build_window_feature_vectors_from_rows_locked(
                node_id=node_id,
                amplitude_rows=aligned_vectors_by_node[node_id][start_slot:end_slot],
                scalar_rows=aligned_scalars_by_node[node_id][start_slot:end_slot],
            )
            if not per_node_features:
                row = []
                break
            row.extend(per_node_features[-1])
        if len(row) == engine.expected_input_size:
            rows.append(row)
    return rows


def build_session_node_features(
    engine: FingerprintEngine,
    session: GtSession,
) -> list[list[float]]:
    grouped = build_session_node_feature_groups(engine, session)
    return [row for group in grouped for row in group]


def build_session_aggregate_features(
    engine: FingerprintEngine,
    session: GtSession,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for node_rows in build_session_node_feature_groups(engine, session):
        aggregate = engine._aggregate_node_feature_rows(node_rows)
        if aggregate is not None:
            rows.append(aggregate)
    return rows


def build_session_node_feature_groups(
    engine: FingerprintEngine,
    session: GtSession,
) -> list[list[list[float]]]:
    all_frames = [frame for frames in session.frames_by_node.values() for frame in frames]
    if not all_frames:
        return []
    start_time = min(frame.captured_at for frame in all_frames)
    end_time = max(frame.captured_at for frame in all_frames)
    horizon_sample_count = max(
        engine.window_sample_count + engine.smoothing_half_window * 2,
        engine.window_sample_count * engine.LIVE_PREPROCESS_HORIZON_WINDOWS,
    )
    required_recent_samples = engine.window_sample_count + engine.smoothing_half_window * 2
    (
        aligned_vectors_by_node,
        aligned_scalars_by_node,
        observed_by_node,
        total_slots,
    ) = build_session_rows_for_live_horizons(
        engine,
        session,
        start_time=start_time,
        end_time=end_time,
    )
    if total_slots < horizon_sample_count:
        return []

    features_by_node: dict[int, np.ndarray] = {}
    valid_counts_by_node: dict[int, np.ndarray] = {}
    for node_id in engine.required_node_ids:
        features = build_vectorized_node_live_horizon_features(
            engine,
            node_id=node_id,
            amplitude_rows=aligned_vectors_by_node[node_id][:total_slots],
            scalar_rows=aligned_scalars_by_node[node_id][:total_slots],
            horizon_sample_count=horizon_sample_count,
        )
        if features.size == 0:
            continue
        observed = np.asarray(observed_by_node[node_id][:total_slots], dtype=bool)
        cumulative_valid = np.concatenate(([0], np.cumsum(observed.astype(np.int32))))
        valid_counts_by_node[node_id] = (
            cumulative_valid[horizon_sample_count:] - cumulative_valid[:-horizon_sample_count]
        )
        features_by_node[node_id] = features

    groups: list[list[list[float]]] = []
    max_feature_count = max((features.shape[0] for features in features_by_node.values()), default=0)
    for feature_index in range(0, max_feature_count, engine.window_step_samples):
        group: list[list[float]] = []
        for node_id in engine.required_node_ids:
            features = features_by_node.get(node_id)
            horizon_valid_counts = valid_counts_by_node.get(node_id)
            if features is None or horizon_valid_counts is None:
                continue
            if feature_index >= features.shape[0] or feature_index >= len(horizon_valid_counts):
                continue
            if horizon_valid_counts[feature_index] < required_recent_samples:
                continue
            row = features[feature_index]
            if row.shape[0] == engine.per_node_feature_size:
                group.append(row.astype(np.float32, copy=False).tolist())
        if group:
            groups.append(group)
    return groups


def build_vectorized_live_horizon_features(
    engine: FingerprintEngine,
    *,
    aligned_vectors_by_node: dict[int, list[list[float]]],
    aligned_scalars_by_node: dict[int, list[list[float]]],
    observed_by_node: dict[int, list[bool]],
    horizon_sample_count: int,
    required_recent_samples: int,
) -> list[list[float]]:
    required_node_ids = engine.required_node_ids
    if not required_node_ids:
        return []
    total_slots = min(len(aligned_vectors_by_node[node_id]) for node_id in required_node_ids)
    if total_slots < horizon_sample_count:
        return []

    valid_mask = np.ones(total_slots, dtype=bool)
    for node_id in required_node_ids:
        observed = np.asarray(observed_by_node[node_id][:total_slots], dtype=bool)
        valid_mask &= observed
    cumulative_valid = np.concatenate(([0], np.cumsum(valid_mask.astype(np.int32))))
    horizon_valid_counts = (
        cumulative_valid[horizon_sample_count:] - cumulative_valid[:-horizon_sample_count]
    )

    per_node_features: list[np.ndarray] = []
    for node_id in required_node_ids:
        features = build_vectorized_node_live_horizon_features(
            engine,
            node_id=node_id,
            amplitude_rows=aligned_vectors_by_node[node_id][:total_slots],
            scalar_rows=aligned_scalars_by_node[node_id][:total_slots],
            horizon_sample_count=horizon_sample_count,
        )
        if features.size == 0:
            return []
        per_node_features.append(features)

    rows: list[list[float]] = []
    for feature_index in range(0, len(horizon_valid_counts), engine.window_step_samples):
        if horizon_valid_counts[feature_index] < required_recent_samples:
            continue
        row = np.concatenate([features[feature_index] for features in per_node_features])
        if row.shape[0] == engine.expected_input_size:
            rows.append(row.astype(np.float32, copy=False).tolist())
    return rows


def build_vectorized_node_live_horizon_features(
    engine: FingerprintEngine,
    *,
    node_id: int,
    amplitude_rows: list[list[float]],
    scalar_rows: list[list[float]],
    horizon_sample_count: int,
) -> np.ndarray:
    amplitude_array = np.asarray(amplitude_rows, dtype=np.float32)
    scalar_array = np.asarray(scalar_rows, dtype=np.float32)
    if amplitude_array.shape[0] < horizon_sample_count:
        return np.empty((0, engine.per_node_feature_size), dtype=np.float32)
    if scalar_array.shape[0] < horizon_sample_count:
        return np.empty((0, engine.per_node_feature_size), dtype=np.float32)

    amplitude_windows = np.lib.stride_tricks.sliding_window_view(
        amplitude_array,
        window_shape=horizon_sample_count,
        axis=0,
    ).transpose(0, 2, 1)
    scalar_windows = np.lib.stride_tricks.sliding_window_view(
        scalar_array,
        window_shape=horizon_sample_count,
        axis=0,
    ).transpose(0, 2, 1)

    baseline = np.asarray(
        engine._baseline_vector_for_node_locked(node_id, amplitude_rows),
        dtype=np.float32,
    )
    centered = amplitude_windows - baseline.reshape(1, 1, -1)
    scales = np.percentile(np.abs(centered), 95.0, axis=1)
    scales = np.maximum(scales, 1e-6).astype(np.float32, copy=False)
    normalized_amplitudes = centered / scales[:, np.newaxis, :]

    scalar_mean = scalar_windows.mean(axis=1)
    scalar_std = scalar_windows.std(axis=1)
    normalized_scalars = (
        scalar_windows - scalar_mean[:, np.newaxis, :]
    ) / np.maximum(scalar_std[:, np.newaxis, :], 1e-6)

    amplitude_tail = normalized_amplitudes[:, -engine.window_sample_count :, :]
    scalar_tail = normalized_scalars[:, -engine.window_sample_count :, :]
    parts = [
        amplitude_tail.mean(axis=1),
        amplitude_tail.std(axis=1),
        np.percentile(
            amplitude_tail,
            list(DEFAULT_QUANTILES),
            axis=1,
        ).transpose(1, 0, 2).reshape(amplitude_tail.shape[0], -1),
        scalar_tail.mean(axis=1),
        scalar_tail.std(axis=1),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def build_session_rows_for_live_horizons(
    engine: FingerprintEngine,
    session: GtSession,
    *,
    start_time: float,
    end_time: float,
) -> tuple[
    dict[int, list[list[float]]],
    dict[int, list[list[float]]],
    dict[int, list[bool]],
    int,
]:
    required_node_ids = engine.required_node_ids
    bucketed_vectors_by_node: dict[int, list[list[float] | None]] = {}
    bucketed_scalars_by_node: dict[int, list[list[float] | None]] = {}
    total_slots = 0
    for node_id in required_node_ids:
        bucketed_vectors, bucketed_scalars = engine._bucketize_feature_frames_locked(
            session.frames_by_node.get(node_id, []),
            start_time=start_time,
            end_time=end_time,
        )
        bucketed_vectors_by_node[node_id] = bucketed_vectors
        bucketed_scalars_by_node[node_id] = bucketed_scalars
        total_slots = max(total_slots, len(bucketed_vectors))

    aligned_vectors_by_node: dict[int, list[list[float]]] = {}
    aligned_scalars_by_node: dict[int, list[list[float]]] = {}
    observed_by_node: dict[int, list[bool]] = {}
    for node_id in required_node_ids:
        bucketed_vectors = bucketed_vectors_by_node.get(node_id, [])
        bucketed_scalars = bucketed_scalars_by_node.get(node_id, [])
        observed_by_node[node_id] = [
            index < len(bucketed_vectors) and bucketed_vectors[index] is not None
            for index in range(total_slots)
        ]
        aligned_vectors_by_node[node_id] = engine._fill_bucket_gaps_locked(
            node_id,
            bucketed_vectors,
            total_slots=total_slots,
        )
        aligned_scalars_by_node[node_id] = engine._fill_scalar_bucket_gaps_locked(
            bucketed_scalars,
            total_slots=total_slots,
        )
    return aligned_vectors_by_node, aligned_scalars_by_node, observed_by_node, total_slots


def count_valid_horizon_slots(
    observed_by_node: dict[int, list[bool]],
    *,
    required_node_ids: list[int],
    start_slot: int,
    end_slot: int,
) -> int:
    valid_slots = 0
    for index in range(start_slot, end_slot):
        if all(
            index < len(observed_by_node.get(node_id, []))
            and observed_by_node[node_id][index]
            for node_id in required_node_ids
        ):
            valid_slots += 1
    return valid_slots


def save_training_baseline(engine: FingerprintEngine) -> None:
    save_fingerprint_store(
        engine.fingerprint_path,
        {
            "version": engine.STORE_VERSION,
            "window_seconds": engine.window_seconds,
            "window_step_seconds": engine.window_step_seconds,
            "feature_signature": engine.feature_signature,
            "node_ids": engine.required_node_ids,
            "input_size": engine.expected_input_size,
            "baseline": {
                "vectors": {
                    str(node_id): vector
                    for node_id, vector in sorted(engine.empty_room_baseline_by_node.items())
                },
                "counts": {
                    str(node_id): count
                    for node_id, count in sorted(engine.empty_room_baseline_counts.items())
                },
            },
            "empty_room": None,
            "cells": {},
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
