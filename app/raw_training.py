from __future__ import annotations

import json
import math
import pickle
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from Config.config_loader import SystemConfig, load_system_config

from .protocol import ACTIVE_SUBCARRIER_COUNT


SCALAR_KEYS = (
    "rssi_dbm",
    "snr_db",
    "amplitude_mean",
    "amplitude_std",
    "amplitude_rms",
    "amplitude_p90",
    "gradient_mean",
    "phase_step_std",
)
DEFAULT_QUANTILES = (10.0, 50.0, 90.0)


@dataclass(frozen=True)
class PacketSample:
    captured_at: float
    node_id: int
    feature_vector: tuple[float, ...]
    scalars: tuple[float, ...]


@dataclass(frozen=True)
class RawSession:
    session_id: str
    kind: str
    label: str
    grid_x: int | None
    grid_y: int | None
    path: str
    packets: tuple[PacketSample, ...]


@dataclass(frozen=True)
class FeatureConfig:
    effective_packets_per_second: float
    window_size: int
    window_step: int
    amplitude_smoothing_half_window: int
    scalar_smoothing_half_window: int
    include_quantiles: bool = True
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    train_ratio: float = 0.70
    purge_gap_windows: int = 0

    def resolved_purge_gap(self) -> int:
        if self.purge_gap_windows > 0:
            return int(self.purge_gap_windows)
        return max(15, self.window_size * 3)


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int
    random_state: int
    max_features: str | int | float | None = "sqrt"
    min_samples_leaf: int = 1
    min_samples_split: int = 2
    class_weight: str | dict[str, float] | None = "balanced_subsample"
    n_jobs: int = 1


@dataclass
class WindowSample:
    label: str
    session_id: str
    session_path: str
    window_index: int
    features: np.ndarray


@dataclass
class BucketedSession:
    session: RawSession
    bucket_count: int
    node_vectors: dict[int, list[np.ndarray | None]]
    node_scalars: dict[int, list[np.ndarray | None]]


@dataclass
class DatasetSplit:
    train_records: list[WindowSample]
    val_records: list[WindowSample]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    split_summary: list[dict[str, Any]]


@dataclass
class CandidateResult:
    feature_config: FeatureConfig
    feature_dim: int
    train_sample_count: int
    val_sample_count: int
    inner_cv: dict[str, Any]
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]


@dataclass
class SearchResult:
    best_candidate: CandidateResult
    best_model: ExtraTreesClassifier
    best_split: DatasetSplit
    candidate_results: list[CandidateResult]
    label_order: list[str]
    report_path: Path
    model_path: Path


def default_workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_system_config_path(workspace_root: str | Path | None = None) -> Path:
    root = default_workspace_root() if workspace_root is None else Path(workspace_root)
    return root / "Config" / "system_config.json"


def load_training_system_config(workspace_root: str | Path | None = None) -> SystemConfig:
    return load_system_config(default_system_config_path(workspace_root))


def load_default_node_ids(workspace_root: str | Path | None = None) -> list[int]:
    system_config = load_training_system_config(workspace_root)
    node_ids = [int(node.node_id) for node in system_config.enabled_nodes()]
    return node_ids if node_ids else list(range(1, 10))


def display_label(label: str) -> str:
    if label == "empty_room":
        return "Empty Room"
    grid_x_text, grid_y_text = label.split(",", 1)
    return f"Cell ({int(grid_x_text) + 1}, {int(grid_y_text) + 1})"


def ordered_class_labels(labels: Iterable[str]) -> list[str]:
    cell_labels: list[tuple[int, int, str]] = []
    empty_labels: list[str] = []
    for label in labels:
        if label == "empty_room":
            empty_labels.append(label)
            continue
        grid_x_text, grid_y_text = label.split(",", 1)
        cell_labels.append((int(grid_y_text), int(grid_x_text), label))
    ordered = [label for _, _, label in sorted(cell_labels)]
    ordered.extend(sorted(set(empty_labels)))
    return ordered


def _coerce_feature_vector(raw: Sequence[object]) -> tuple[float, ...] | None:
    if not isinstance(raw, Sequence):
        return None
    values = [float(value) for value in raw[:ACTIVE_SUBCARRIER_COUNT]]
    if len(values) < ACTIVE_SUBCARRIER_COUNT:
        return None
    return tuple(values)


def _coerce_scalars(record: dict[str, object]) -> tuple[float, ...]:
    rssi_dbm = float(record.get("rssi_dbm", 0.0))
    noise_floor_dbm = float(record.get("noise_floor_dbm", 0.0))
    return (
        rssi_dbm,
        float(record.get("snr_db", rssi_dbm - noise_floor_dbm)),
        float(record.get("amplitude_mean", 0.0)),
        float(record.get("amplitude_std", 0.0)),
        float(record.get("amplitude_rms", 0.0)),
        float(record.get("amplitude_p90", 0.0)),
        float(record.get("gradient_mean", 0.0)),
        float(record.get("phase_step_std", 0.0)),
    )


def load_raw_sessions(raw_dir: str | Path) -> list[RawSession]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    sessions: list[RawSession] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        if path.name == "sessions.jsonl":
            continue
        packets: list[PacketSample] = []
        session_end: dict[str, object] | None = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record_type = str(record.get("record_type", ""))
                if record_type == "packet" and bool(record.get("valid_adr018", False)):
                    feature_vector = _coerce_feature_vector(record.get("feature_vector", []))
                    if feature_vector is None:
                        continue
                    packets.append(
                        PacketSample(
                            captured_at=float(record.get("captured_at_unix", 0.0)),
                            node_id=int(record.get("node_id", 0)),
                            feature_vector=feature_vector,
                            scalars=_coerce_scalars(record),
                        )
                    )
                elif record_type == "session_end":
                    session_end = record

        if session_end is None:
            continue
        if not packets:
            raise RuntimeError(f"No valid packet rows found in {path}")

        sessions.append(
            RawSession(
                session_id=str(session_end.get("session_id", path.stem)),
                kind=str(session_end.get("kind", "")),
                label=str(session_end.get("label", path.stem)),
                grid_x=(
                    None
                    if session_end.get("grid_x") is None
                    else int(session_end.get("grid_x", 0))
                ),
                grid_y=(
                    None
                    if session_end.get("grid_y") is None
                    else int(session_end.get("grid_y", 0))
                ),
                path=str(path.resolve()),
                packets=tuple(packets),
            )
        )
    if not sessions:
        raise RuntimeError(f"No raw sessions could be loaded from {raw_dir}")
    return sessions


def build_extra_trees_model(model_config: ModelConfig) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(model_config.n_estimators),
        random_state=int(model_config.random_state),
        max_features=model_config.max_features,
        min_samples_leaf=int(model_config.min_samples_leaf),
        min_samples_split=int(model_config.min_samples_split),
        class_weight=model_config.class_weight,
        n_jobs=int(model_config.n_jobs),
    )


class RawWindowDatasetBuilder:
    def __init__(
        self,
        sessions: Sequence[RawSession],
        *,
        node_ids: Sequence[int],
        effective_packets_per_second: float,
    ) -> None:
        self.sessions = list(sessions)
        self.node_ids = [int(node_id) for node_id in node_ids]
        self.effective_packets_per_second = float(effective_packets_per_second)
        self.baseline_by_node, self.baseline_counts = self._build_baseline_by_node()
        self.bucketed_sessions = self._bucketize_sessions()

    @property
    def labels(self) -> list[str]:
        ordered: list[str] = []
        for session in self.sessions:
            label = "empty_room" if session.kind == "empty_room" else f"{session.grid_x},{session.grid_y}"
            ordered.append(label)
        return ordered_class_labels(ordered)

    def _build_baseline_by_node(self) -> tuple[dict[int, np.ndarray], dict[int, int]]:
        baseline_packets = [
            packet
            for session in self.sessions
            if session.kind == "empty_room"
            for packet in session.packets
            if packet.node_id in self.node_ids
        ]
        if not baseline_packets:
            raise RuntimeError("At least one empty-room session is required to build a baseline.")

        baseline_by_node: dict[int, np.ndarray] = {}
        baseline_counts: dict[int, int] = {}
        for node_id in self.node_ids:
            vectors = [
                np.asarray(packet.feature_vector, dtype=np.float32)
                for packet in baseline_packets
                if packet.node_id == node_id
            ]
            if not vectors:
                raise RuntimeError(f"Empty-room data is missing node {node_id}.")
            baseline_by_node[node_id] = np.mean(np.stack(vectors), axis=0)
            baseline_counts[node_id] = len(vectors)
        return baseline_by_node, baseline_counts

    def _bucketize_sessions(self) -> list[BucketedSession]:
        bucketed_sessions: list[BucketedSession] = []
        for session in self.sessions:
            start_time = min(packet.captured_at for packet in session.packets)
            end_time = max(packet.captured_at for packet in session.packets)
            bucket_count = max(
                1,
                int(round((end_time - start_time) * self.effective_packets_per_second)),
            )
            by_node: dict[int, list[list[PacketSample]]] = {
                node_id: [[] for _ in range(bucket_count)] for node_id in self.node_ids
            }
            for packet in session.packets:
                if packet.node_id not in by_node:
                    continue
                bucket_index = int(
                    (packet.captured_at - start_time) * self.effective_packets_per_second
                )
                if bucket_index < 0:
                    continue
                if bucket_index >= bucket_count:
                    if math.isclose(packet.captured_at, end_time, abs_tol=1e-9):
                        bucket_index = bucket_count - 1
                    else:
                        continue
                by_node[packet.node_id][bucket_index].append(packet)

            node_vectors: dict[int, list[np.ndarray | None]] = {}
            node_scalars: dict[int, list[np.ndarray | None]] = {}
            for node_id in self.node_ids:
                vector_rows: list[np.ndarray | None] = []
                scalar_rows: list[np.ndarray | None] = []
                for bucket in by_node[node_id]:
                    if not bucket:
                        vector_rows.append(None)
                        scalar_rows.append(None)
                        continue
                    vector_rows.append(
                        np.mean(
                            np.stack(
                                [
                                    np.asarray(packet.feature_vector, dtype=np.float32)
                                    for packet in bucket
                                ]
                            ),
                            axis=0,
                        )
                    )
                    scalar_rows.append(
                        np.mean(
                            np.stack(
                                [
                                    np.asarray(packet.scalars, dtype=np.float32)
                                    for packet in bucket
                                ]
                            ),
                            axis=0,
                        )
                    )
                node_vectors[node_id] = vector_rows
                node_scalars[node_id] = scalar_rows

            bucketed_sessions.append(
                BucketedSession(
                    session=session,
                    bucket_count=bucket_count,
                    node_vectors=node_vectors,
                    node_scalars=node_scalars,
                )
            )
        return bucketed_sessions

    def build_window_samples(self, feature_config: FeatureConfig) -> list[WindowSample]:
        samples: list[WindowSample] = []
        for bucketed_session in self.bucketed_sessions:
            label = (
                "empty_room"
                if bucketed_session.session.kind == "empty_room"
                else f"{bucketed_session.session.grid_x},{bucketed_session.session.grid_y}"
            )
            prepared_by_node = self._prepare_session_arrays(bucketed_session, feature_config)
            for start in range(
                0,
                bucketed_session.bucket_count - feature_config.window_size + 1,
                feature_config.window_step,
            ):
                feature_parts: list[np.ndarray] = []
                for node_id in self.node_ids:
                    amplitude_rows, scalar_rows = prepared_by_node[node_id]
                    amplitude_window = amplitude_rows[start : start + feature_config.window_size]
                    scalar_window = scalar_rows[start : start + feature_config.window_size]
                    feature_parts.append(
                        self._window_features(
                            amplitude_window,
                            scalar_window,
                            feature_config,
                        )
                    )
                samples.append(
                    WindowSample(
                        label=label,
                        session_id=bucketed_session.session.session_id,
                        session_path=bucketed_session.session.path,
                        window_index=start,
                        features=np.concatenate(feature_parts).astype(np.float32, copy=False),
                    )
                )
        if not samples:
            raise RuntimeError("No window samples could be generated from the raw data.")
        return samples

    def build_split(self, feature_config: FeatureConfig) -> DatasetSplit:
        samples = self.build_window_samples(feature_config)
        grouped: OrderedDict[str, list[WindowSample]] = OrderedDict()
        for sample in samples:
            grouped.setdefault(sample.session_id, []).append(sample)

        train_records: list[WindowSample] = []
        val_records: list[WindowSample] = []
        split_summary: list[dict[str, Any]] = []
        purge_gap = feature_config.resolved_purge_gap()
        for session_id, session_records in grouped.items():
            ordered_records = sorted(session_records, key=lambda item: item.window_index)
            total = len(ordered_records)
            split_index = int(round(total * feature_config.train_ratio))
            train_end = max(1, split_index - purge_gap)
            val_start = min(total - 1, split_index + purge_gap)
            if val_start <= train_end:
                midpoint = total // 2
                train_end = max(1, midpoint - purge_gap)
                val_start = min(total - 1, midpoint + purge_gap)
            local_train = ordered_records[:train_end]
            local_val = ordered_records[val_start:]
            if not local_train or not local_val:
                raise RuntimeError(
                    f"Session {session_id} is too short for a purged train/val split "
                    f"(total_windows={total}, purge_gap={purge_gap})."
                )
            train_records.extend(local_train)
            val_records.extend(local_val)
            split_summary.append(
                {
                    "session_id": session_id,
                    "session_path": local_train[0].session_path,
                    "label": display_label(local_train[0].label),
                    "total_windows": total,
                    "train_windows": len(local_train),
                    "val_windows": len(local_val),
                    "purge_gap_windows": purge_gap,
                }
            )

        X_train = np.stack([record.features for record in train_records]).astype(np.float32, copy=False)
        y_train = np.asarray([record.label for record in train_records])
        X_val = np.stack([record.features for record in val_records]).astype(np.float32, copy=False)
        y_val = np.asarray([record.label for record in val_records])
        return DatasetSplit(
            train_records=train_records,
            val_records=val_records,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            split_summary=split_summary,
        )

    def _prepare_session_arrays(
        self,
        bucketed_session: BucketedSession,
        feature_config: FeatureConfig,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        prepared: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        scalar_default = np.zeros(len(SCALAR_KEYS), dtype=np.float32)
        for node_id in self.node_ids:
            baseline = self.baseline_by_node[node_id].astype(np.float32, copy=False)
            filled_vectors = self._fill_missing_rows(
                bucketed_session.node_vectors[node_id],
                baseline,
            )
            filled_scalars = self._fill_missing_rows(
                bucketed_session.node_scalars[node_id],
                scalar_default,
            )

            amplitude_rows = np.stack(filled_vectors).astype(np.float32, copy=False)
            amplitude_rows = amplitude_rows - baseline
            scales = np.percentile(np.abs(amplitude_rows), 95.0, axis=0)
            scales = np.maximum(scales, 1e-6).astype(np.float32, copy=False)
            amplitude_rows = amplitude_rows / scales
            amplitude_rows = smooth_rows(
                amplitude_rows,
                feature_config.amplitude_smoothing_half_window,
            )

            scalar_rows = np.stack(filled_scalars).astype(np.float32, copy=False)
            scalar_mean = scalar_rows.mean(axis=0)
            scalar_std = scalar_rows.std(axis=0)
            scalar_rows = (scalar_rows - scalar_mean) / np.maximum(scalar_std, 1e-6)
            scalar_rows = smooth_rows(
                scalar_rows,
                feature_config.scalar_smoothing_half_window,
            )
            prepared[node_id] = (amplitude_rows, scalar_rows)
        return prepared

    @staticmethod
    def _fill_missing_rows(
        rows: Sequence[np.ndarray | None],
        default_row: np.ndarray,
    ) -> list[np.ndarray]:
        normalized: list[np.ndarray | None] = [None if row is None else row.copy() for row in rows]
        last_seen: np.ndarray | None = None
        for index, row in enumerate(normalized):
            if row is None:
                if last_seen is not None:
                    normalized[index] = last_seen.copy()
                continue
            last_seen = row

        next_seen: np.ndarray | None = None
        for index in range(len(normalized) - 1, -1, -1):
            row = normalized[index]
            if row is None:
                if next_seen is not None:
                    normalized[index] = next_seen.copy()
                continue
            next_seen = row

        return [
            default_row.copy() if row is None else row.astype(np.float32, copy=False)
            for row in normalized
        ]

    @staticmethod
    def _window_features(
        amplitude_window: np.ndarray,
        scalar_window: np.ndarray,
        feature_config: FeatureConfig,
    ) -> np.ndarray:
        parts: list[np.ndarray] = [
            amplitude_window.mean(axis=0),
            amplitude_window.std(axis=0),
        ]
        if feature_config.include_quantiles:
            quantile_values = np.percentile(
                amplitude_window,
                list(feature_config.quantiles),
                axis=0,
            ).reshape(-1)
            parts.append(quantile_values.astype(np.float32, copy=False))
        parts.append(scalar_window.mean(axis=0))
        parts.append(scalar_window.std(axis=0))
        return np.concatenate(parts).astype(np.float32, copy=False)


def smooth_rows(rows: np.ndarray, half_window: int) -> np.ndarray:
    if half_window <= 0 or rows.shape[0] <= 1:
        return rows.copy()
    denominator = float(half_window * 2 + 1)
    smoothed = np.zeros_like(rows, dtype=np.float32)
    for row_index in range(rows.shape[0]):
        for offset in range(-half_window, half_window + 1):
            smoothed[row_index] += rows[_mirrored_index(row_index + offset, rows.shape[0])]
        smoothed[row_index] /= denominator
    return smoothed


def _mirrored_index(index: int, size: int) -> int:
    if size <= 1:
        return 0
    while index < 0 or index >= size:
        if index < 0:
            index = -index
        elif index >= size:
            index = size - (index - size) - 2
    return index


def build_temporal_cv_folds(
    records: Sequence[WindowSample],
    *,
    n_splits: int,
    purge_gap_windows: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    grouped: OrderedDict[str, list[int]] = OrderedDict()
    for index, record in enumerate(records):
        grouped.setdefault(record.session_id, []).append(index)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(n_splits):
        fold_train: list[int] = []
        fold_val: list[int] = []
        for session_indices in grouped.values():
            total = len(session_indices)
            boundaries = np.linspace(0, total, n_splits + 1, dtype=int)
            val_start = int(boundaries[fold_index])
            val_end = int(boundaries[fold_index + 1])
            if val_end <= val_start:
                continue
            fold_val.extend(session_indices[val_start:val_end])

            left_end = max(0, val_start - purge_gap_windows)
            right_start = min(total, val_end + purge_gap_windows)
            fold_train.extend(session_indices[:left_end])
            fold_train.extend(session_indices[right_start:])
        if fold_train and fold_val:
            folds.append(
                (
                    np.asarray(sorted(fold_train), dtype=int),
                    np.asarray(sorted(fold_val), dtype=int),
                )
            )
    if not folds:
        raise RuntimeError("Could not build any temporal CV folds from the training data.")
    return folds


def evaluate_model_metrics(
    model: ExtraTreesClassifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    label_order: Sequence[str],
) -> dict[str, Any]:
    predictions = model.predict(X)
    return evaluate_predictions(
        y_true=y,
        y_pred=predictions,
        label_order=label_order,
    )


def evaluate_predictions(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_order: Sequence[str],
) -> dict[str, Any]:
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(label_order),
        target_names=[display_label(label) for label in label_order],
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(label_order),
        ).tolist(),
        "classification_report": report,
    }


def evaluate_temporal_cv(
    X: np.ndarray,
    y: np.ndarray,
    records: Sequence[WindowSample],
    *,
    label_order: Sequence[str],
    model_config: ModelConfig,
    n_splits: int,
    purge_gap_windows: int,
) -> dict[str, Any]:
    folds = build_temporal_cv_folds(
        records,
        n_splits=n_splits,
        purge_gap_windows=purge_gap_windows,
    )
    fold_results: list[dict[str, Any]] = []
    for fold_index, (train_indices, val_indices) in enumerate(folds, start=1):
        model = build_extra_trees_model(model_config)
        model.fit(X[train_indices], y[train_indices])
        metrics = evaluate_model_metrics(
            model,
            X[val_indices],
            y[val_indices],
            label_order=label_order,
        )
        fold_results.append(
            {
                "fold": fold_index,
                "train_count": int(train_indices.size),
                "val_count": int(val_indices.size),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return {
        "n_splits": len(fold_results),
        "purge_gap_windows": purge_gap_windows,
        "folds": fold_results,
        "mean_accuracy": float(
            np.mean([fold_result["accuracy"] for fold_result in fold_results])
        ),
        "mean_macro_f1": float(
            np.mean([fold_result["macro_f1"] for fold_result in fold_results])
        ),
    }


def train_search_candidates(
    builder: RawWindowDatasetBuilder,
    *,
    feature_configs: Sequence[FeatureConfig],
    search_model_config: ModelConfig,
    final_model_config: ModelConfig,
    cv_splits: int,
) -> tuple[CandidateResult, ExtraTreesClassifier, DatasetSplit, list[CandidateResult]]:
    label_order = builder.labels
    candidate_results: list[CandidateResult] = []
    selected_model: ExtraTreesClassifier | None = None
    selected_split: DatasetSplit | None = None
    selected_candidate: CandidateResult | None = None

    for feature_config in feature_configs:
        split = builder.build_split(feature_config)
        inner_cv = evaluate_temporal_cv(
            split.X_train,
            split.y_train,
            split.train_records,
            label_order=label_order,
            model_config=search_model_config,
            n_splits=cv_splits,
            purge_gap_windows=max(1, feature_config.window_size),
        )

        final_model = build_extra_trees_model(final_model_config)
        final_model.fit(split.X_train, split.y_train)
        train_metrics = evaluate_model_metrics(
            final_model,
            split.X_train,
            split.y_train,
            label_order=label_order,
        )
        val_metrics = evaluate_model_metrics(
            final_model,
            split.X_val,
            split.y_val,
            label_order=label_order,
        )
        candidate = CandidateResult(
            feature_config=feature_config,
            feature_dim=int(split.X_train.shape[1]),
            train_sample_count=int(split.X_train.shape[0]),
            val_sample_count=int(split.X_val.shape[0]),
            inner_cv=inner_cv,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        candidate_results.append(candidate)

        if selected_candidate is None:
            selected_candidate = candidate
            selected_model = final_model
            selected_split = split
            continue
        current_key = (
            candidate.inner_cv["mean_macro_f1"],
            candidate.val_metrics["macro_f1"],
            candidate.val_metrics["accuracy"],
        )
        selected_key = (
            selected_candidate.inner_cv["mean_macro_f1"],
            selected_candidate.val_metrics["macro_f1"],
            selected_candidate.val_metrics["accuracy"],
        )
        if current_key > selected_key:
            selected_candidate = candidate
            selected_model = final_model
            selected_split = split

    if selected_candidate is None or selected_model is None or selected_split is None:
        raise RuntimeError("No training candidate could be selected.")
    return selected_candidate, selected_model, selected_split, candidate_results


def train_and_save_raw_model(
    *,
    raw_dir: str | Path,
    output_dir: str | Path,
    node_ids: Sequence[int],
    feature_configs: Sequence[FeatureConfig],
    search_model_config: ModelConfig,
    final_model_config: ModelConfig,
    cv_splits: int,
) -> SearchResult:
    builder = RawWindowDatasetBuilder(
        load_raw_sessions(raw_dir),
        node_ids=node_ids,
        effective_packets_per_second=feature_configs[0].effective_packets_per_second,
    )
    best_candidate, best_model, best_split, candidate_results = train_search_candidates(
        builder,
        feature_configs=feature_configs,
        search_model_config=search_model_config,
        final_model_config=final_model_config,
        cv_splits=cv_splits,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "raw_training_report.json"
    model_path = output_dir / "raw_training_model.pkl"

    report_payload = build_report_payload(
        builder=builder,
        best_candidate=best_candidate,
        best_split=best_split,
        candidate_results=candidate_results,
        search_model_config=search_model_config,
        final_model_config=final_model_config,
        raw_dir=Path(raw_dir),
        output_dir=output_dir,
    )
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": best_model,
                "feature_config": asdict(best_candidate.feature_config),
                "model_config": asdict(final_model_config),
                "labels": builder.labels,
                "label_display_names": [display_label(label) for label in builder.labels],
                "node_ids": builder.node_ids,
                "baseline_by_node": {
                    str(node_id): builder.baseline_by_node[node_id].tolist()
                    for node_id in builder.node_ids
                },
                "baseline_counts": {
                    str(node_id): int(builder.baseline_counts[node_id])
                    for node_id in builder.node_ids
                },
                "scalar_keys": list(SCALAR_KEYS),
                "created_at_unix": time.time(),
            },
            handle,
        )

    return SearchResult(
        best_candidate=best_candidate,
        best_model=best_model,
        best_split=best_split,
        candidate_results=candidate_results,
        label_order=builder.labels,
        report_path=report_path,
        model_path=model_path,
    )


def build_report_payload(
    *,
    builder: RawWindowDatasetBuilder,
    best_candidate: CandidateResult,
    best_split: DatasetSplit,
    candidate_results: Sequence[CandidateResult],
    search_model_config: ModelConfig,
    final_model_config: ModelConfig,
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    session_rows: list[dict[str, Any]] = []
    for session in builder.sessions:
        session_rows.append(
            {
                "session_id": session.session_id,
                "kind": session.kind,
                "label": session.label,
                "display_label": (
                    "Empty Room"
                    if session.kind == "empty_room"
                    else f"Cell ({int(session.grid_x) + 1}, {int(session.grid_y) + 1})"
                ),
                "path": session.path,
                "packet_count": len(session.packets),
            }
        )

    return {
        "raw_dir": str(raw_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "label_order": builder.labels,
        "label_display_names": [display_label(label) for label in builder.labels],
        "node_ids": builder.node_ids,
        "baseline_counts": {
            str(node_id): int(builder.baseline_counts[node_id]) for node_id in builder.node_ids
        },
        "sessions": session_rows,
        "search_model_config": asdict(search_model_config),
        "final_model_config": asdict(final_model_config),
        "best_candidate": _candidate_to_jsonable(best_candidate),
        "candidate_results": [_candidate_to_jsonable(candidate) for candidate in candidate_results],
        "split_summary": best_split.split_summary,
        "dataset_shapes": {
            "train": [int(best_split.X_train.shape[0]), int(best_split.X_train.shape[1])],
            "val": [int(best_split.X_val.shape[0]), int(best_split.X_val.shape[1])],
        },
        "created_at_unix": time.time(),
    }


def _candidate_to_jsonable(candidate: CandidateResult) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["feature_config"]["quantiles"] = list(candidate.feature_config.quantiles)
    return payload


def format_candidate_summary(candidate: CandidateResult) -> str:
    feature_config = candidate.feature_config
    return (
        "window="
        f"{feature_config.window_size}, step={feature_config.window_step}, "
        f"amp_smoothing={feature_config.amplitude_smoothing_half_window}, "
        f"scalar_smoothing={feature_config.scalar_smoothing_half_window}, "
        f"quantiles={'on' if feature_config.include_quantiles else 'off'}, "
        f"cv_macro_f1={candidate.inner_cv['mean_macro_f1']:.4f}, "
        f"val_macro_f1={candidate.val_metrics['macro_f1']:.4f}, "
        f"val_acc={candidate.val_metrics['accuracy']:.4f}"
    )
