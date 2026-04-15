from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Config.config_loader import load_system_config, save_system_config

from .protocol import (
    ACTIVE_SUBCARRIER_COUNT,
    FeatureFrame,
    build_feature_frame,
    parse_adr018_frame,
)
from sklearn.ensemble import RandomForestClassifier

from .storage import (
    load_fingerprint_store,
    load_pickle_store,
    remove_store,
    save_fingerprint_store,
    save_pickle_store,
)


@dataclass
class CellDataset:
    cell_key: str
    grid_x: int
    grid_y: int
    captured_at: float
    total_frames: int
    capture_count: int = 1
    node_count: int = 0
    window_sample_count: int = 0
    samples: list[list[float]] = field(default_factory=list)


@dataclass
class LiveNodeState:
    node_id: int
    label: str
    source: str = ""
    last_seen_ts: float = 0.0
    last_sequence: int = 0
    packets_received: int = 0
    window_samples: int = 0
    rssi_dbm: float = 0.0
    noise_floor_dbm: float = 0.0
    snr_db: float = 0.0
    subcarrier_count: int = 0


@dataclass
class CaptureSession:
    cell_key: str
    grid_x: int
    grid_y: int
    started_at: float
    ends_at: float
    frames_by_node: dict[int, list[FeatureFrame]] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    model_key: str
    trained_at: float
    window_seconds: float
    window_step_seconds: float
    node_ids: list[int]
    input_size: int
    sample_count: int
    class_labels: list[str]
    summary: str


class DualStageRandomForestClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 280,
        random_state: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_features: str | int | float | None = "sqrt",
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.column_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            n_jobs=-1,
        )
        self.row_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            n_jobs=-1,
        )
        self.classes_: list[str] = []
        self._class_pairs: list[tuple[str, str]] = []
        self._column_prob_index: dict[str, int] = {}
        self._row_prob_index: dict[str, int] = {}

    @staticmethod
    def _split_cell_label(label: str) -> tuple[str, str]:
        grid_x_text, grid_y_text = str(label).split(",", 1)
        return grid_x_text, grid_y_text

    def fit(self, features: list[list[float]], labels: list[str]) -> DualStageRandomForestClassifier:
        if not features or not labels:
            raise RuntimeError("Training samples are required for dual-stage RandomForest.")
        column_labels: list[str] = []
        row_labels: list[str] = []
        for label in labels:
            grid_x_text, grid_y_text = self._split_cell_label(label)
            column_labels.append(grid_x_text)
            row_labels.append(grid_y_text)

        self.column_classifier.fit(features, column_labels)
        self.row_classifier.fit(features, row_labels)
        self._column_prob_index = {
            str(value): index for index, value in enumerate(self.column_classifier.classes_)
        }
        self._row_prob_index = {
            str(value): index for index, value in enumerate(self.row_classifier.classes_)
        }

        ordered_labels: list[str] = []
        seen: set[str] = set()
        for label in labels:
            normalized = str(label)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_labels.append(normalized)
        self.classes_ = ordered_labels
        self._class_pairs = [self._split_cell_label(label) for label in self.classes_]
        return self

    def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        if not self.classes_:
            raise RuntimeError("Dual-stage RandomForest model is not fitted.")
        column_probabilities = self.column_classifier.predict_proba(features)
        row_probabilities = self.row_classifier.predict_proba(features)

        merged_probabilities: list[list[float]] = []
        for sample_index in range(len(features)):
            row_distribution: list[float] = []
            for grid_x_text, grid_y_text in self._class_pairs:
                column_index = self._column_prob_index.get(grid_x_text)
                row_index = self._row_prob_index.get(grid_y_text)
                if column_index is None or row_index is None:
                    row_distribution.append(0.0)
                    continue
                row_distribution.append(
                    float(column_probabilities[sample_index][column_index])
                    * float(row_probabilities[sample_index][row_index])
                )
            total = sum(row_distribution)
            if total > 0.0:
                row_distribution = [value / total for value in row_distribution]
            merged_probabilities.append(row_distribution)
        return merged_probabilities


class FingerprintEngine:
    STORE_VERSION = 4
    PROBABILITY_SMOOTHING_SECONDS = 0.8
    BEST_CELL_SWITCH_MARGIN = 0.08
    BEST_CELL_SWITCH_DELAY_SECONDS = 0.9
    PREDICTION_STALE_GRACE_SECONDS = 1.25
    MODEL_ORDER = ("RandomForestDualStage", "RandomForestUnified")
    CSI_SMOOTHING_HALF_WINDOW = 20
    LIVE_PREPROCESS_HORIZON_WINDOWS = 6

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.config_path = self.workspace_root.parent / "Config" / "system_config.json"
        self.fingerprint_path = self.workspace_root / "data" / "fingerprints.json"
        self.model_path = self.workspace_root / "data" / "model_bundle.pkl"
        self.comm_log_path = self.workspace_root / "data" / "communication.log"
        self.lock = threading.RLock()
        self.system_config = load_system_config(self.config_path)
        self.node_windows: dict[int, deque[FeatureFrame]] = {}
        self.live_nodes: dict[int, LiveNodeState] = {}
        self.cell_datasets: dict[str, CellDataset] = {}
        self.capture_session: CaptureSession | None = None
        self.packet_count = 0
        self.last_packet_ts: float | None = None
        self.last_probabilities: dict[str, float] = {}
        self.last_prediction_ts: float | None = None
        self.last_best_cell: str | None = None
        self.last_best_probability = 0.0
        self.pending_best_cell: str | None = None
        self.pending_best_since: float | None = None
        self.model_pipelines: dict[str, Any] = {}
        self.model_metadata_by_name: dict[str, ModelMetadata] = {}
        self.active_model_name: str | None = None
        self.empty_room_baseline_by_node: dict[int, list[float]] = {}
        self.empty_room_baseline_counts: dict[int, int] = {}
        self.status_message = (
            "Press Learn on each cell to collect data, then click Train Models."
        )
        self.udp_status = (
            f"Ready for UDP {self.system_config.host.listen_host}:{self.system_config.host.udp_port}"
        )
        self.comm_logs: deque[str] = deque(maxlen=250)
        self.last_invalid_packet_log_ts = 0.0
        self._loaded_store_window_seconds: float | None = None
        self._loaded_store_window_step_seconds: float | None = None
        self._loaded_store_node_ids: list[int] | None = None
        self._loaded_store_input_size: int | None = None
        self._load_datasets()
        self._normalize_training_state_for_config()
        self._load_models()
        if self.model_pipelines:
            self.status_message = (
                f"Loaded {len(self.model_pipelines)} trained models. "
                f"Active model: {self.active_model_name}."
            )
        elif any(dataset.window_sample_count > 0 for dataset in self.cell_datasets.values()):
            self.status_message = (
                "Saved Learn data loaded. Click Train Models to fit "
                "RandomForestDualStage / RandomForestUnified."
            )
        self._log_event_locked(
            "INFO",
            "App started with config "
            f"{self.config_path} and UDP {self.system_config.host.listen_host}:{self.system_config.host.udp_port}",
        )

    @property
    def grid_cols(self) -> int:
        return self.system_config.grid.cols

    @property
    def grid_rows(self) -> int:
        return self.system_config.grid.rows

    @property
    def total_cells(self) -> int:
        return self.grid_cols * self.grid_rows

    @property
    def min_samples(self) -> int:
        return self.system_config.fingerprinting.minimum_samples_per_node

    @property
    def feature_bin_count(self) -> int:
        return self.system_config.fingerprinting.feature_bin_count

    @property
    def capture_seconds(self) -> float:
        return self.system_config.fingerprinting.capture_seconds

    @property
    def window_seconds(self) -> float:
        return self.system_config.fingerprinting.window_seconds

    @property
    def window_step_seconds(self) -> float:
        return self.system_config.fingerprinting.window_step_seconds

    @property
    def keepalive_pings_per_second(self) -> float:
        return self.system_config.host.keepalive_pings_per_second

    @property
    def required_node_ids(self) -> list[int]:
        enabled = [node.node_id for node in self.system_config.enabled_nodes()]
        if enabled:
            return enabled
        return sorted(self.live_nodes)

    @property
    def per_node_feature_size(self) -> int:
        return ACTIVE_SUBCARRIER_COUNT * 2

    @property
    def expected_input_size(self) -> int:
        return len(self.required_node_ids) * self.per_node_feature_size

    def set_udp_status(self, message: str) -> None:
        with self.lock:
            self.udp_status = message
            self._log_event_locked("UDP", message)

    def train_models(self) -> str:
        with self.lock:
            if self.capture_session is not None:
                raise RuntimeError("Wait for the current Learn capture to finish first.")
            if not self._can_train_locked():
                raise RuntimeError(
                    "Every cell needs at least one Learn capture before training all models."
                )
            return self._train_models_locked()

    def set_active_model(self, model_name: str) -> None:
        with self.lock:
            if model_name not in self.model_pipelines:
                raise RuntimeError(f"Model '{model_name}' is not trained yet.")
            if self.active_model_name == model_name:
                return
            self.active_model_name = model_name
            self._clear_prediction_locked()
            self.status_message = f"Active inference model set to {model_name}."
            self._save_models()
            self._log_event_locked("TRAIN", f"Active inference model set to {model_name}")

    def apply_grid_settings(
        self,
        cols: int,
        rows: int,
        capture_seconds: float,
        window_seconds: float,
        window_step_seconds: float,
        keepalive_pings_per_second: float | None = None,
    ) -> None:
        with self.lock:
            cols = max(1, int(cols))
            rows = max(1, int(rows))
            capture_seconds = max(1.0, float(capture_seconds))
            window_seconds = max(0.25, min(float(window_seconds), capture_seconds))
            window_step_seconds = max(
                0.05,
                min(float(window_step_seconds), window_seconds),
            )
            if keepalive_pings_per_second is None:
                keepalive_pings_per_second = self.keepalive_pings_per_second
            keepalive_pings_per_second = max(0.0, float(keepalive_pings_per_second))
            grid_changed = cols != self.grid_cols or rows != self.grid_rows
            window_changed = not math.isclose(window_seconds, self.window_seconds, abs_tol=1e-6)
            window_step_changed = not math.isclose(
                window_step_seconds,
                self.window_step_seconds,
                abs_tol=1e-6,
            )

            self.system_config.grid.cols = cols
            self.system_config.grid.rows = rows
            self.system_config.fingerprinting.capture_seconds = capture_seconds
            self.system_config.fingerprinting.window_seconds = window_seconds
            self.system_config.fingerprinting.window_step_seconds = window_step_seconds
            self.system_config.host.keepalive_pings_per_second = keepalive_pings_per_second
            save_system_config(self.config_path, self.system_config)

            if grid_changed or window_changed or window_step_changed:
                self._reset_training_state_locked()
                reason = []
                if grid_changed:
                    reason.append(f"grid {cols}x{rows}")
                if window_changed:
                    reason.append(f"window {window_seconds:.2f}s")
                if window_step_changed:
                    reason.append(f"window step {window_step_seconds:.2f}s")
                self.status_message = (
                    "Training data and trained models were cleared after changing "
                    + " and ".join(reason)
                    + "."
                )
                self._log_event_locked(
                    "CFG",
                    "Reset saved training data after updating "
                    + " and ".join(reason),
                )
            else:
                self._log_event_locked(
                    "CFG",
                    f"Updated capture_seconds to {capture_seconds:.2f}s, "
                    f"window_seconds to {window_seconds:.2f}s, "
                    f"window_step_seconds to {window_step_seconds:.2f}s, "
                    f"and keepalive ping rate to {keepalive_pings_per_second:.1f}/s",
                )

    def start_capture(self, grid_x: int, grid_y: int) -> None:
        with self.lock:
            now = time.time()
            self._advance_capture_locked(now)
            if self.capture_session is not None:
                active = self.capture_session
                raise RuntimeError(
                    f"Capture already running for cell ({active.grid_x + 1}, {active.grid_y + 1})."
                )
            self.capture_session = CaptureSession(
                cell_key=self.cell_key(grid_x, grid_y),
                grid_x=grid_x,
                grid_y=grid_y,
                started_at=now,
                ends_at=now + self.capture_seconds,
            )
            self.status_message = (
                f"Started capture for cell ({grid_x + 1}, {grid_y + 1}). "
                f"Hold position for {self.capture_seconds:.1f}s. "
                f"Window size: {self.window_seconds:.2f}s. "
                f"Step: {self.window_step_seconds:.2f}s."
            )
            self._log_event_locked(
                "CAPTURE",
                f"Capture started for cell ({grid_x + 1}, {grid_y + 1}) for "
                f"{self.capture_seconds:.1f}s with {self.window_seconds:.2f}s windows "
                f"and {self.window_step_seconds:.2f}s step",
            )

    def clear_cell(self, grid_x: int, grid_y: int) -> None:
        with self.lock:
            cell_key = self.cell_key(grid_x, grid_y)
            if self.cell_datasets.pop(cell_key, None) is not None:
                self._save_datasets()
                self._clear_models_locked()
                self.status_message = (
                    f"Cleared Learn data for cell ({grid_x + 1}, {grid_y + 1}). "
                    "Train Models is now required again."
                )
                self._log_event_locked(
                    "CAPTURE",
                    f"Cleared training data for cell ({grid_x + 1}, {grid_y + 1})",
                )

    def clear_all(self) -> None:
        with self.lock:
            self._reset_training_state_locked()
            self.status_message = (
                "Cleared all saved Learn data and all trained models."
            )
            self._log_event_locked("CAPTURE", "Cleared all saved training data and models")

    def process_packet(self, payload: bytes, source: str) -> bool:
        frame = parse_adr018_frame(payload)
        if frame is None:
            now = time.time()
            with self.lock:
                if now - self.last_invalid_packet_log_ts >= 5.0:
                    self.last_invalid_packet_log_ts = now
                    self._log_event_locked(
                        "WARN",
                        f"Ignored non-ADR018 or malformed packet from {source}",
                    )
            return False

        now = time.time()
        feature = build_feature_frame(frame, source, now, self.feature_bin_count)
        with self.lock:
            self.packet_count += 1
            self.last_packet_ts = now
            self._advance_capture_locked(now)

            node_window = self.node_windows.setdefault(feature.node_id, deque())
            node_window.append(feature)
            self._prune_node_window_locked(feature.node_id, now)
            self._update_empty_room_baseline_locked(feature)

            node_state = self.live_nodes.get(feature.node_id)
            if node_state is None:
                node_state = LiveNodeState(
                    node_id=feature.node_id,
                    label=self._node_label(feature.node_id),
                )
                self.live_nodes[feature.node_id] = node_state
                self._log_event_locked(
                    "UDP",
                    f"First packet from node {feature.node_id} ({node_state.label}) source={source}",
                )

            node_state.label = self._node_label(feature.node_id)
            node_state.source = source
            node_state.last_seen_ts = now
            node_state.last_sequence = feature.sequence
            node_state.packets_received += 1
            node_state.window_samples = self._count_recent_frames(node_window, now)
            node_state.rssi_dbm = feature.rssi_dbm
            node_state.noise_floor_dbm = feature.noise_floor_dbm
            node_state.snr_db = feature.snr_db
            node_state.subcarrier_count = feature.n_subcarriers
            if node_state.packets_received == 1 or node_state.packets_received % 25 == 0:
                self._log_event_locked(
                    "UDP",
                    f"Node {feature.node_id} seq={feature.sequence} "
                    f"packets={node_state.packets_received} "
                    f"rssi={feature.rssi_dbm:.1f} snr={feature.snr_db:.1f} source={source}",
                )

            if self.capture_session is not None and now <= self.capture_session.ends_at:
                self.capture_session.frames_by_node.setdefault(feature.node_id, []).append(
                    feature
                )

            self._advance_capture_locked(now)
            self._recompute_probabilities_locked(now)
        return True

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            now = time.time()
            self._advance_capture_locked(now)
            self._prune_all_live_windows_locked(now)
            self._recompute_probabilities_locked(now)

            capture_payload: dict[str, object] = {
                "active": False,
                "grid_x": None,
                "grid_y": None,
                "remaining_seconds": 0.0,
                "progress": [],
            }
            if self.capture_session is not None:
                capture_payload = {
                    "active": True,
                    "grid_x": self.capture_session.grid_x,
                    "grid_y": self.capture_session.grid_y,
                    "remaining_seconds": max(0.0, self.capture_session.ends_at - now),
                    "progress": [
                        {
                            "node_id": node_id,
                            "sample_count": len(frames),
                        }
                        for node_id, frames in sorted(
                            self.capture_session.frames_by_node.items()
                        )
                    ],
                }

            cells = []
            trained_cells = 0
            dataset_samples = 0
            for grid_y in range(self.grid_rows):
                for grid_x in range(self.grid_cols):
                    cell_key = self.cell_key(grid_x, grid_y)
                    dataset = self.cell_datasets.get(cell_key)
                    trained = dataset is not None and dataset.window_sample_count > 0
                    if trained:
                        trained_cells += 1
                        dataset_samples += dataset.window_sample_count
                    cells.append(
                        {
                            "cell_key": cell_key,
                            "grid_x": grid_x,
                            "grid_y": grid_y,
                            "trained": trained,
                            "node_count": dataset.node_count if dataset else 0,
                            "total_frames": dataset.total_frames if dataset else 0,
                            "capture_count": dataset.capture_count if dataset else 0,
                            "window_sample_count": dataset.window_sample_count
                            if dataset
                            else 0,
                            "probability": self.last_probabilities.get(cell_key, 0.0),
                            "is_best": self.last_best_cell == cell_key,
                            "is_capturing": self.capture_session is not None
                            and self.capture_session.cell_key == cell_key,
                        }
                    )

            nodes = []
            for node_id, node in sorted(self.live_nodes.items()):
                nodes.append(
                    {
                        "node_id": node_id,
                        "label": node.label,
                        "source": node.source,
                        "age_ms": max(0.0, (now - node.last_seen_ts) * 1000.0)
                        if node.last_seen_ts
                        else None,
                        "packets_received": node.packets_received,
                        "window_samples": node.window_samples,
                        "rssi_dbm": node.rssi_dbm,
                        "snr_db": node.snr_db,
                        "subcarrier_count": node.subcarrier_count,
                    }
                )

            available_models = self._available_model_names_locked()
            active_model_ready = self.active_model_name in self.model_pipelines
            live_prediction_ready = active_model_ready and bool(self.last_probabilities)

            return {
                "host": {
                    "listen_host": self.system_config.host.listen_host,
                    "target_ip": self.system_config.host.target_ip,
                    "udp_port": self.system_config.host.udp_port,
                    "keepalive_pings_per_second": self.system_config.host.keepalive_pings_per_second,
                    "config_path": str(self.config_path),
                },
                "grid": {
                    "cols": self.grid_cols,
                    "rows": self.grid_rows,
                    "capture_seconds": self.capture_seconds,
                    "window_seconds": self.window_seconds,
                    "window_step_seconds": self.window_step_seconds,
                },
                "training": {
                    "trained_cells": trained_cells,
                    "total_cells": self.total_cells,
                    "dataset_samples": dataset_samples,
                    "required_nodes": len(self.required_node_ids),
                    "model_ready": active_model_ready,
                    "can_train": self._can_train_locked(),
                    "active_model": self.active_model_name,
                    "available_models": available_models,
                    "trained_model_count": len(available_models),
                    "ready_for_inference": live_prediction_ready,
                },
                "metrics": {
                    "packet_count": self.packet_count,
                    "active_nodes": sum(
                        1
                        for node in self.live_nodes.values()
                        if now - node.last_seen_ts <= 3.0
                    ),
                    "last_packet_age_ms": None
                    if self.last_packet_ts is None
                    else max(0.0, (now - self.last_packet_ts) * 1000.0),
                },
                "capture": capture_payload,
                "prediction": {
                    "ready": live_prediction_ready,
                    "model_ready": active_model_ready,
                    "active_model": self.active_model_name,
                    "available_models": available_models,
                    "best_cell_key": self.last_best_cell,
                    "best_probability": self.last_best_probability,
                },
                "cells": cells,
                "nodes": nodes,
                "udp_status": self.udp_status,
                "status_message": self.status_message,
                "comm_logs": list(self.comm_logs),
                "comm_log_path": str(self.comm_log_path),
            }

    def _advance_capture_locked(self, now: float) -> None:
        if self.capture_session is None or now < self.capture_session.ends_at:
            return
        session = self.capture_session
        self.capture_session = None

        (
            samples,
            total_window_slots,
            valid_window_count,
            total_frames,
            observed_node_count,
        ) = self._build_dataset_samples_locked(session)

        if not samples:
            self.status_message = (
                f"Capture failed for cell ({session.grid_x + 1}, {session.grid_y + 1}): "
                f"no complete {self.window_seconds:.2f}s windows at "
                f"{self.window_step_seconds:.2f}s steps with all "
                f"{len(self.required_node_ids)} required nodes."
            )
            self._log_event_locked(
                "CAPTURE",
                f"Capture failed for cell ({session.grid_x + 1}, {session.grid_y + 1}); "
                f"valid windows=0/{total_window_slots} observed nodes={observed_node_count}/"
                f"{len(self.required_node_ids)}",
            )
            return

        previous = self.cell_datasets.get(session.cell_key)
        merged_samples = samples
        merged_total_frames = total_frames
        merged_capture_count = 1
        if previous is not None:
            merged_samples = previous.samples + samples
            merged_total_frames = previous.total_frames + total_frames
            merged_capture_count = previous.capture_count + 1
        self.cell_datasets[session.cell_key] = CellDataset(
            cell_key=session.cell_key,
            grid_x=session.grid_x,
            grid_y=session.grid_y,
            captured_at=now,
            total_frames=merged_total_frames,
            capture_count=merged_capture_count,
            node_count=len(self.required_node_ids),
            window_sample_count=len(merged_samples),
            samples=merged_samples,
        )
        self._save_datasets()
        self._clear_models_locked()

        message = (
            f"Capture completed for cell ({session.grid_x + 1}, {session.grid_y + 1}): "
            f"{valid_window_count}/{total_window_slots} windows kept, "
            f"{total_frames} frames across {observed_node_count}/"
            f"{len(self.required_node_ids)} required nodes. "
            f"Cell now has {len(merged_samples)} total windows across "
            f"{merged_capture_count} captures. Click Train Models when ready."
        )
        self.status_message = message
        self._log_event_locked(
            "CAPTURE",
            f"Capture completed for cell ({session.grid_x + 1}, {session.grid_y + 1}) "
            f"with {valid_window_count}/{total_window_slots} valid windows and "
            f"{total_frames} frames; accumulated windows={len(merged_samples)} "
            f"captures={merged_capture_count}",
        )

    def _build_dataset_samples_locked(
        self, session: CaptureSession
    ) -> tuple[list[list[float]], int, int, int, int]:
        required_node_ids = self.required_node_ids
        duration = session.ends_at - session.started_at
        window_starts = self._window_start_offsets(duration)
        total_window_slots = len(window_starts)
        if total_window_slots <= 0 or not required_node_ids:
            return [], total_window_slots, 0, 0, 0

        total_frames = 0
        observed_node_ids: set[int] = set()
        features_by_node: dict[int, list[list[float] | None]] = {}

        for node_id in required_node_ids:
            frames = session.frames_by_node.get(node_id, [])
            total_frames += len(frames)
            if frames:
                observed_node_ids.add(node_id)
            features_by_node[node_id] = self._build_window_feature_vectors(
                frames,
                node_id,
                session.started_at,
                window_starts,
            )

        samples: list[list[float]] = []
        valid_window_count = 0
        for index, _window_start in enumerate(window_starts):
            sample: list[float] = []
            complete = True
            for node_id in required_node_ids:
                feature_vector = features_by_node[node_id][index]
                if feature_vector is None:
                    complete = False
                    break
                sample.extend(feature_vector)
            if complete:
                samples.append(sample)
                valid_window_count += 1

        return (
            samples,
            total_window_slots,
            valid_window_count,
            total_frames,
            len(observed_node_ids),
        )

    def _window_start_offsets(self, duration: float) -> list[float]:
        if duration + 1e-9 < self.window_seconds:
            return []
        slot_count = (
            int(
                math.floor(
                    (duration - self.window_seconds) / self.window_step_seconds + 1e-9
                )
            )
            + 1
        )
        return [
            index * self.window_step_seconds for index in range(max(0, slot_count))
        ]

    def _build_window_feature_vectors(
        self,
        frames: list[FeatureFrame],
        node_id: int,
        session_started_at: float,
        window_starts: list[float],
    ) -> list[list[float] | None]:
        if not window_starts:
            return []

        usable_frames: list[tuple[float, list[float]]] = []
        max_window_end = window_starts[-1] + self.window_seconds
        for frame in frames:
            offset = frame.captured_at - session_started_at
            if offset < 0.0 or offset >= max_window_end:
                continue
            if len(frame.feature_vector) != ACTIVE_SUBCARRIER_COUNT:
                continue
            usable_frames.append((offset, frame.feature_vector))

        if not usable_frames:
            return [None for _ in window_starts]

        usable_frames.sort(key=lambda item: item[0])
        timestamps = [offset for offset, _ in usable_frames]
        raw_vectors = [vector for _, vector in usable_frames]
        preprocessed = self._preprocess_node_vectors_locked(node_id, raw_vectors)

        features: list[list[float] | None] = []
        left = 0
        right = 0
        for window_start in window_starts:
            window_end = window_start + self.window_seconds
            while left < len(timestamps) and timestamps[left] < window_start:
                left += 1
            if right < left:
                right = left
            while right < len(timestamps) and timestamps[right] < window_end:
                right += 1
            if right <= left:
                features.append(None)
                continue
            window_vectors = preprocessed[left:right]
            means = self._vector_mean(window_vectors)
            stds = self._vector_std(window_vectors, means)
            features.append([*means, *stds])
        return features

    def _preprocess_node_vectors_locked(
        self,
        node_id: int,
        vectors: list[list[float]],
    ) -> list[list[float]]:
        if not vectors:
            return []
        feature_size = min(len(vector) for vector in vectors)
        if feature_size <= 0:
            return [[] for _ in vectors]
        clipped_vectors = [vector[:feature_size] for vector in vectors]
        baseline = self._baseline_vector_for_node_locked(node_id, clipped_vectors)

        centered: list[list[float]] = []
        for vector in clipped_vectors:
            centered.append(
                [vector[index] - baseline[index] for index in range(feature_size)]
            )

        scales: list[float] = []
        for index in range(feature_size):
            max_abs = max(abs(vector[index]) for vector in centered)
            scales.append(max(max_abs, 1e-9))

        normalized = [
            [vector[index] / scales[index] for index in range(feature_size)]
            for vector in centered
        ]
        return self._smooth_vectors(normalized, self.CSI_SMOOTHING_HALF_WINDOW)

    def _baseline_vector_for_node_locked(
        self,
        node_id: int,
        vectors: list[list[float]],
    ) -> list[float]:
        cached = self.empty_room_baseline_by_node.get(node_id)
        if cached is not None and len(cached) == len(vectors[0]):
            return cached
        return self._vector_mean(vectors)

    def _update_empty_room_baseline_locked(self, frame: FeatureFrame) -> None:
        if self.capture_session is not None:
            return
        if self.cell_datasets:
            return
        vector = frame.feature_vector
        if len(vector) != ACTIVE_SUBCARRIER_COUNT:
            return
        baseline = self.empty_room_baseline_by_node.get(frame.node_id)
        count = self.empty_room_baseline_counts.get(frame.node_id, 0)
        if baseline is None or len(baseline) != len(vector):
            self.empty_room_baseline_by_node[frame.node_id] = list(vector)
            self.empty_room_baseline_counts[frame.node_id] = 1
            return

        new_count = min(5000, count + 1)
        weight = 1.0 / float(new_count)
        for index, value in enumerate(vector):
            baseline[index] += (value - baseline[index]) * weight
        self.empty_room_baseline_counts[frame.node_id] = new_count

    @classmethod
    def _smooth_vectors(
        cls,
        vectors: list[list[float]],
        half_window: int,
    ) -> list[list[float]]:
        if not vectors:
            return []
        if half_window <= 0 or len(vectors) == 1:
            return [vector.copy() for vector in vectors]

        frame_count = len(vectors)
        feature_size = min(len(vector) for vector in vectors)
        denominator = float(half_window * 2 + 1)
        smoothed = [[0.0 for _ in range(feature_size)] for _ in range(frame_count)]
        for frame_index in range(frame_count):
            for offset in range(-half_window, half_window + 1):
                mirrored_index = cls._mirrored_index(frame_index + offset, frame_count)
                source = vectors[mirrored_index]
                for feature_index in range(feature_size):
                    smoothed[frame_index][feature_index] += source[feature_index]
            for feature_index in range(feature_size):
                smoothed[frame_index][feature_index] /= denominator
        return smoothed

    @staticmethod
    def _mirrored_index(index: int, size: int) -> int:
        if size <= 1:
            return 0
        while index < 0 or index >= size:
            if index < 0:
                index = -index
                continue
            index = 2 * size - index - 2
        return index

    def _recompute_probabilities_locked(self, now: float) -> None:
        active_pipeline = self._active_model_pipeline_locked()
        if active_pipeline is None:
            self._clear_prediction_locked()
            return

        sample = self._build_live_sample_locked(now)
        if sample is None:
            stale_after = max(
                self.PREDICTION_STALE_GRACE_SECONDS,
                self.window_seconds * 1.5,
            )
            if (
                self.last_prediction_ts is not None
                and now - self.last_prediction_ts <= stale_after
                and self.last_probabilities
            ):
                return
            self._clear_prediction_locked()
            return

        raw_probabilities = {
            cell_key: 0.0 for cell_key in self._ordered_cell_keys()
        }
        probabilities = active_pipeline.predict_proba([sample])[0]
        classes = [str(label) for label in active_pipeline.classes_]
        for cell_key, probability in zip(classes, probabilities):
            raw_probabilities[cell_key] = float(probability)

        self.last_probabilities = self._smooth_probabilities_locked(
            raw_probabilities,
            now,
        )
        candidate_best, candidate_probability = max(
            self.last_probabilities.items(),
            key=lambda item: item[1],
        )
        self._update_best_cell_locked(
            candidate_best,
            candidate_probability,
            now,
        )
        self.last_prediction_ts = now

    def _active_model_pipeline_locked(self) -> Any | None:
        if self.active_model_name is None:
            return None
        return self.model_pipelines.get(self.active_model_name)

    def _smooth_probabilities_locked(
        self,
        raw_probabilities: dict[str, float],
        now: float,
    ) -> dict[str, float]:
        if not self.last_probabilities or self.last_prediction_ts is None:
            return dict(raw_probabilities)

        smoothing_seconds = max(
            self.PROBABILITY_SMOOTHING_SECONDS,
            self.window_seconds * 0.75,
        )
        delta_seconds = max(0.0, now - self.last_prediction_ts)
        if delta_seconds <= 0.0 or delta_seconds >= smoothing_seconds * 3.0:
            return dict(raw_probabilities)

        alpha = 1.0 - math.exp(-delta_seconds / smoothing_seconds)
        alpha = max(0.18, min(0.85, alpha))
        smoothed = {}
        for cell_key in self._ordered_cell_keys():
            previous = self.last_probabilities.get(cell_key, 0.0)
            current = raw_probabilities.get(cell_key, 0.0)
            smoothed[cell_key] = previous + (current - previous) * alpha

        total = sum(smoothed.values())
        if total > 0.0:
            return {
                cell_key: value / total for cell_key, value in smoothed.items()
            }
        return smoothed

    def _update_best_cell_locked(
        self,
        candidate_best: str,
        candidate_probability: float,
        now: float,
    ) -> None:
        previous_best = self.last_best_cell
        if previous_best is None:
            self.last_best_cell = candidate_best
            self.last_best_probability = candidate_probability
            self.pending_best_cell = None
            self.pending_best_since = None
            return

        current_probability = self.last_probabilities.get(previous_best, 0.0)
        if candidate_best == previous_best:
            self.last_best_probability = current_probability
            self.pending_best_cell = None
            self.pending_best_since = None
            return

        if self.pending_best_cell != candidate_best:
            self.pending_best_cell = candidate_best
            self.pending_best_since = now

        pending_age = 0.0
        if self.pending_best_since is not None:
            pending_age = max(0.0, now - self.pending_best_since)
        switch_delay = max(
            self.BEST_CELL_SWITCH_DELAY_SECONDS,
            self.window_seconds * 0.75,
        )
        if (
            candidate_probability
            >= current_probability + self.BEST_CELL_SWITCH_MARGIN
            or pending_age >= switch_delay
        ):
            self.last_best_cell = candidate_best
            self.last_best_probability = candidate_probability
            self.pending_best_cell = None
            self.pending_best_since = None
            best_x, best_y = [int(value) + 1 for value in self.last_best_cell.split(",")]
            self._log_event_locked(
                "PREDICT",
                f"Best cell changed to ({best_x}, {best_y}) "
                f"probability={self.last_best_probability * 100.0:.1f}%",
            )
            return

        self.last_best_probability = current_probability

    def _build_live_sample_locked(self, now: float) -> list[float] | None:
        required_node_ids = self.required_node_ids
        if not required_node_ids:
            return None

        live_horizon_seconds = max(
            self.window_seconds * self.LIVE_PREPROCESS_HORIZON_WINDOWS,
            self.window_seconds + self.window_step_seconds * 2.0,
            2.0,
        )
        live_start = now - live_horizon_seconds
        target_window_start = live_horizon_seconds - self.window_seconds
        sample: list[float] = []
        for node_id in required_node_ids:
            node_window = self.node_windows.get(node_id)
            if not node_window:
                return None
            recent_frames = [
                frame
                for frame in node_window
                if frame.captured_at >= live_start
            ]
            if not recent_frames:
                return None
            per_node_features = self._build_window_feature_vectors(
                recent_frames,
                node_id,
                live_start,
                [target_window_start],
            )
            if not per_node_features or per_node_features[0] is None:
                return None
            sample.extend(per_node_features[0])

        if len(sample) != self.expected_input_size:
            return None
        return sample

    def _can_train_locked(self) -> bool:
        ordered_cell_keys = self._ordered_cell_keys()
        if len(self.cell_datasets) != self.total_cells or self.expected_input_size <= 0:
            return False
        return all(
            cell_key in self.cell_datasets and bool(self.cell_datasets[cell_key].samples)
            for cell_key in ordered_cell_keys
        )

    def _available_model_names_locked(self) -> list[str]:
        return [
            model_name for model_name in self.MODEL_ORDER if model_name in self.model_pipelines
        ]

    def _build_training_matrix_locked(self) -> tuple[list[list[float]], list[str]]:
        if not self._can_train_locked():
            raise RuntimeError(
                "Every cell needs at least one Learn capture before training all models."
            )

        features: list[list[float]] = []
        labels: list[str] = []
        for cell_key in self._ordered_cell_keys():
            dataset = self.cell_datasets[cell_key]
            for sample in dataset.samples:
                if len(sample) != self.expected_input_size:
                    raise RuntimeError(
                        f"Training data mismatch for cell ({dataset.grid_x + 1}, "
                        f"{dataset.grid_y + 1}); expected input size "
                        f"{self.expected_input_size}, got {len(sample)}."
                    )
                features.append(sample)
                labels.append(cell_key)
        return features, labels

    def _build_model_pipeline_locked(
        self, model_name: str
    ) -> tuple[Any, str]:
        if model_name == "RandomForestDualStage":
            model = DualStageRandomForestClassifier(
                n_estimators=280,
                random_state=200,
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features="sqrt",
            )
            return (
                model,
                "hierarchical RF (column + row), n_estimators=280, random_state=200",
            )
        if model_name == "RandomForestUnified":
            model = RandomForestClassifier(
                n_estimators=320,
                random_state=200,
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features="sqrt",
                n_jobs=-1,
            )
            return (
                model,
                "unified RF, n_estimators=320, random_state=200",
            )
        raise RuntimeError(f"Unsupported model '{model_name}'.")

    def _train_models_locked(self) -> str:
        features, labels = self._build_training_matrix_locked()

        self._clear_prediction_locked()
        self.model_pipelines = {}
        self.model_metadata_by_name = {}
        remove_store(self.model_path)

        trained_models: list[str] = []
        sample_count = len(features)
        class_labels = self._ordered_cell_keys()
        for model_name in self.MODEL_ORDER:
            pipeline, summary = self._build_model_pipeline_locked(model_name)
            pipeline.fit(features, labels)
            self.model_pipelines[model_name] = pipeline
            self.model_metadata_by_name[model_name] = ModelMetadata(
                model_key=model_name,
                trained_at=time.time(),
                window_seconds=self.window_seconds,
                window_step_seconds=self.window_step_seconds,
                node_ids=self.required_node_ids,
                input_size=self.expected_input_size,
                sample_count=sample_count,
                class_labels=class_labels,
                summary=summary,
            )
            trained_models.append(model_name)
            self._log_event_locked(
                "TRAIN",
                f"Trained {model_name} on {sample_count} samples across "
                f"{len(class_labels)} cells ({summary})",
            )

        self.active_model_name = (
            self.active_model_name
            if self.active_model_name in self.model_pipelines
            else trained_models[0]
        )
        self._save_models()
        self.status_message = (
            f"Trained {len(trained_models)} models on {sample_count} samples. "
            f"Active model: {self.active_model_name}."
        )
        return self.status_message

    def _load_datasets(self) -> None:
        raw = load_fingerprint_store(self.fingerprint_path)
        self._loaded_store_window_seconds = (
            float(raw["window_seconds"]) if "window_seconds" in raw else None
        )
        self._loaded_store_window_step_seconds = (
            float(raw["window_step_seconds"])
            if "window_step_seconds" in raw
            else self._loaded_store_window_seconds
        )
        self._loaded_store_node_ids = (
            [int(value) for value in raw.get("node_ids", [])]
            if "node_ids" in raw
            else None
        )
        self._loaded_store_input_size = (
            int(raw["input_size"]) if "input_size" in raw else None
        )

        for cell_key, payload in raw.get("cells", {}).items():
            samples_raw = payload.get("samples")
            if not isinstance(samples_raw, list):
                continue
            samples = [
                [float(value) for value in sample]
                for sample in samples_raw
                if isinstance(sample, list)
            ]
            self.cell_datasets[cell_key] = CellDataset(
                cell_key=cell_key,
                grid_x=int(payload.get("grid_x", 0)),
                grid_y=int(payload.get("grid_y", 0)),
                captured_at=float(payload.get("captured_at", 0.0)),
                total_frames=int(payload.get("total_frames", 0)),
                capture_count=max(1, int(payload.get("capture_count", 1))),
                node_count=int(payload.get("node_count", len(self.required_node_ids))),
                window_sample_count=int(payload.get("window_sample_count", len(samples))),
                samples=samples,
            )

    def _save_datasets(self) -> None:
        payload = {
            "version": self.STORE_VERSION,
            "window_seconds": self.window_seconds,
            "window_step_seconds": self.window_step_seconds,
            "node_ids": self.required_node_ids,
            "input_size": self.expected_input_size,
            "cells": {
                cell_key: {
                    "grid_x": cell.grid_x,
                    "grid_y": cell.grid_y,
                    "captured_at": cell.captured_at,
                    "total_frames": cell.total_frames,
                    "capture_count": cell.capture_count,
                    "node_count": cell.node_count,
                    "window_sample_count": cell.window_sample_count,
                    "samples": cell.samples,
                }
                for cell_key, cell in sorted(self.cell_datasets.items())
            },
        }
        save_fingerprint_store(self.fingerprint_path, payload)

    def _load_models(self) -> None:
        payload = load_pickle_store(self.model_path)
        if not isinstance(payload, dict):
            return
        loaded_models: dict[str, Any] = {}
        loaded_metadata: dict[str, ModelMetadata] = {}

        if "models" in payload:
            models_raw = payload.get("models")
            if not isinstance(models_raw, dict):
                remove_store(self.model_path)
                return
            for model_name, model_payload in models_raw.items():
                if not isinstance(model_payload, dict):
                    continue
                if model_name not in self.MODEL_ORDER:
                    continue
                pipeline = model_payload.get("pipeline")
                metadata_raw = model_payload.get("metadata")
                if pipeline is None or not isinstance(metadata_raw, dict):
                    continue
                metadata = self._metadata_from_payload(model_name, metadata_raw)
                if not self._model_metadata_matches_config(metadata):
                    continue
                loaded_models[model_name] = pipeline
                loaded_metadata[model_name] = metadata
            active_model_name = str(payload.get("active_model", "")) or None
        else:
            pipeline = payload.get("pipeline")
            metadata_raw = payload.get("metadata")
            if pipeline is None or not isinstance(metadata_raw, dict):
                remove_store(self.model_path)
                return
            fallback_model_name = str(
                metadata_raw.get("model_key", "RandomForestUnified")
            )
            if fallback_model_name not in self.MODEL_ORDER:
                remove_store(self.model_path)
                return
            metadata = self._metadata_from_payload(fallback_model_name, metadata_raw)
            if self._model_metadata_matches_config(metadata):
                loaded_models[fallback_model_name] = pipeline
                loaded_metadata[fallback_model_name] = metadata
            active_model_name = fallback_model_name

        if not loaded_models:
            remove_store(self.model_path)
            return

        self.model_pipelines = loaded_models
        self.model_metadata_by_name = loaded_metadata
        self.active_model_name = (
            active_model_name
            if active_model_name in self.model_pipelines
            else self._available_model_names_locked()[0]
        )

    def _metadata_from_payload(
        self, model_name: str, metadata_raw: dict[str, object]
    ) -> ModelMetadata:
        return ModelMetadata(
            model_key=str(metadata_raw.get("model_key", model_name)),
            trained_at=float(metadata_raw.get("trained_at", 0.0)),
            window_seconds=float(metadata_raw.get("window_seconds", 0.0)),
            window_step_seconds=float(
                metadata_raw.get(
                    "window_step_seconds",
                    metadata_raw.get("window_seconds", 0.0),
                )
            ),
            node_ids=[int(value) for value in metadata_raw.get("node_ids", [])],
            input_size=int(metadata_raw.get("input_size", 0)),
            sample_count=int(metadata_raw.get("sample_count", 0)),
            class_labels=[str(value) for value in metadata_raw.get("class_labels", [])],
            summary=str(metadata_raw.get("summary", "")),
        )

    def _save_models(self) -> None:
        if not self.model_pipelines:
            remove_store(self.model_path)
            return
        payload = {
            "active_model": self.active_model_name,
            "models": {
                model_name: {
                    "pipeline": pipeline,
                    "metadata": {
                        "model_key": metadata.model_key,
                        "trained_at": metadata.trained_at,
                        "window_seconds": metadata.window_seconds,
                        "window_step_seconds": metadata.window_step_seconds,
                        "node_ids": metadata.node_ids,
                        "input_size": metadata.input_size,
                        "sample_count": metadata.sample_count,
                        "class_labels": metadata.class_labels,
                        "summary": metadata.summary,
                    },
                }
                for model_name, pipeline in self.model_pipelines.items()
                if (metadata := self.model_metadata_by_name.get(model_name)) is not None
            },
        }
        save_pickle_store(self.model_path, payload)

    def _normalize_training_state_for_config(self) -> None:
        valid_cell_keys = set(self._ordered_cell_keys())
        config_matches = (
            self._loaded_store_window_seconds is None
            or math.isclose(
                self._loaded_store_window_seconds,
                self.window_seconds,
                abs_tol=1e-6,
            )
        )
        if self._loaded_store_window_step_seconds is not None:
            config_matches = config_matches and math.isclose(
                self._loaded_store_window_step_seconds,
                self.window_step_seconds,
                abs_tol=1e-6,
            )
        if self._loaded_store_node_ids is not None:
            config_matches = config_matches and self._loaded_store_node_ids == self.required_node_ids
        if self._loaded_store_input_size is not None:
            config_matches = config_matches and (
                self._loaded_store_input_size == self.expected_input_size
            )

        if not config_matches:
            if self.cell_datasets:
                self._log_event_locked(
                    "CFG",
                    "Cleared saved training data because the config no longer matches "
                    "the stored windowed dataset",
                )
            self.cell_datasets.clear()
            self._save_datasets()
            remove_store(self.model_path)
            return

        normalized = {
            key: value for key, value in self.cell_datasets.items() if key in valid_cell_keys
        }
        if len(normalized) != len(self.cell_datasets):
            self.cell_datasets = normalized
            self._save_datasets()

    def _model_metadata_matches_config(self, metadata: ModelMetadata) -> bool:
        if not math.isclose(metadata.window_seconds, self.window_seconds, abs_tol=1e-6):
            return False
        if not math.isclose(
            metadata.window_step_seconds,
            self.window_step_seconds,
            abs_tol=1e-6,
        ):
            return False
        if metadata.node_ids != self.required_node_ids:
            return False
        if metadata.input_size != self.expected_input_size:
            return False
        if metadata.class_labels != self._ordered_cell_keys():
            return False
        return True

    def _reset_training_state_locked(self) -> None:
        self.capture_session = None
        self.cell_datasets.clear()
        self.node_windows.clear()
        self._save_datasets()
        self._clear_models_locked()

    def _clear_models_locked(self) -> None:
        self.model_pipelines = {}
        self.model_metadata_by_name = {}
        self.active_model_name = None
        self._clear_prediction_locked()
        remove_store(self.model_path)

    def _clear_prediction_locked(self) -> None:
        self.last_probabilities = {}
        self.last_prediction_ts = None
        self.last_best_cell = None
        self.last_best_probability = 0.0
        self.pending_best_cell = None
        self.pending_best_since = None

    def _prune_all_live_windows_locked(self, now: float) -> None:
        for node_id in list(self.node_windows):
            self._prune_node_window_locked(node_id, now)

    def _prune_node_window_locked(self, node_id: int, now: float) -> None:
        window = self.node_windows.get(node_id)
        if window is None:
            return
        horizon = max(self.capture_seconds + self.window_seconds, self.window_seconds * 3.0, 5.0)
        cutoff = now - horizon
        while window and window[0].captured_at < cutoff:
            window.popleft()
        if not window:
            self.node_windows.pop(node_id, None)

    def _count_recent_frames(self, window: deque[FeatureFrame], now: float) -> int:
        window_start = now - self.window_seconds
        return sum(1 for frame in window if frame.captured_at >= window_start)

    @staticmethod
    def _vector_mean(vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        usable = min(len(vector) for vector in vectors)
        return [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(usable)
        ]

    @staticmethod
    def _vector_std(vectors: list[list[float]], means: list[float]) -> list[float]:
        if not vectors or not means:
            return []
        count = len(vectors)
        return [
            math.sqrt(
                sum((vector[index] - means[index]) ** 2 for vector in vectors) / count
            )
            for index in range(len(means))
        ]

    def _ordered_cell_keys(self) -> list[str]:
        return [
            self.cell_key(grid_x, grid_y)
            for grid_y in range(self.grid_rows)
            for grid_x in range(self.grid_cols)
        ]

    def _node_label(self, node_id: int) -> str:
        for node in self.system_config.nodes:
            if node.node_id == node_id:
                return node.label
        return f"ESP {node_id}"

    @staticmethod
    def cell_key(grid_x: int, grid_y: int) -> str:
        return f"{grid_x},{grid_y}"

    def _log_event_locked(self, level: str, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{timestamp}] [{level}] {message}"
        self.comm_logs.append(line)
        self.comm_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.comm_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
