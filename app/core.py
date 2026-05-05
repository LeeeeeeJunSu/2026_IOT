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
    observed_window_count: int = 0
    generated_window_count: int = 0
    total_window_slots: int = 0
    samples: list[list[float]] = field(default_factory=list)


@dataclass
class LiveNodeState:
    node_id: int
    label: str
    source: str = ""
    last_seen_ts: float = 0.0
    last_status_log_ts: float = 0.0
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
    max_ends_at: float = 0.0
    extension_count: int = 0
    frames_by_node: dict[int, list[FeatureFrame]] = field(default_factory=dict)


@dataclass
class BaselineCaptureSession:
    ready_at: float
    started_at: float
    ends_at: float
    frames_by_node: dict[int, list[FeatureFrame]] = field(default_factory=dict)


@dataclass
class DatasetBuildResult:
    samples: list[list[float]] = field(default_factory=list)
    total_frames: int = 0
    observed_node_count: int = 0
    total_resampled_slots: int = 0
    valid_resampled_slots: int = 0
    total_window_slots: int = 0
    observed_window_count: int = 0
    generated_window_count: int = 0


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
    STORE_VERSION = 8
    MODEL_ORDER = ("RandomForestDualStage", "RandomForestUnified")
    LIVE_PREPROCESS_HORIZON_WINDOWS = 6
    INFERENCE_MIN_INTERVAL_SECONDS = 0.10
    RUNTIME_MAX_TICK_SECONDS = 0.05
    NODE_TELEMETRY_REFRESH_SECONDS = 0.25
    UDP_STATUS_LOG_INTERVAL_SECONDS = 2.0

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
        self.baseline_capture_session: BaselineCaptureSession | None = None
        self.packet_count = 0
        self.last_packet_ts: float | None = None
        self.last_probabilities: dict[str, float] = {}
        self.last_prediction_ts: float | None = None
        self.last_best_cell: str | None = None
        self.last_best_probability = 0.0
        self.pending_best_cell: str | None = None
        self.pending_best_since: float | None = None
        self.last_inference_duration_ms = 0.0
        self.last_inference_completed_ts: float | None = None
        self.inference_cycle_count = 0
        self.model_pipelines: dict[str, Any] = {}
        self.model_metadata_by_name: dict[str, ModelMetadata] = {}
        self.active_model_name: str | None = None
        self.empty_room_baseline_by_node: dict[int, list[float]] = {}
        self.empty_room_baseline_counts: dict[int, int] = {}
        self.status_message = (
            "Capture an empty-room baseline, then press Learn on each cell and click Train Models."
        )
        self.udp_status = (
            f"Ready for UDP {self.system_config.host.listen_host}:{self.system_config.host.udp_port}"
        )
        self.comm_logs: deque[str] = deque(maxlen=250)
        self.last_invalid_packet_log_ts = 0.0
        self._loaded_store_version: int | None = None
        self._loaded_store_window_seconds: float | None = None
        self._loaded_store_window_step_seconds: float | None = None
        self._loaded_store_node_ids: list[int] | None = None
        self._loaded_store_input_size: int | None = None
        self._inference_dirty = True
        self._next_inference_ts = 0.0
        self._last_node_telemetry_refresh_ts = 0.0
        self._load_datasets()
        self._normalize_training_state_for_config()
        self._load_models()
        if self.model_pipelines:
            self.status_message = (
                f"Loaded {len(self.model_pipelines)} trained models. "
                f"Active model: {self.active_model_name}."
            )
        elif self._baseline_ready_locked():
            self.status_message = (
                "Empty-room baseline loaded. Press Learn on each cell to collect data."
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
        return float(self.window_sample_count) / self.effective_packets_per_second

    @property
    def window_step_seconds(self) -> float:
        return float(self.window_step_samples) / self.effective_packets_per_second

    @property
    def effective_packets_per_second(self) -> float:
        return self.system_config.fingerprinting.effective_packets_per_second

    @property
    def window_sample_count(self) -> int:
        return self.system_config.fingerprinting.window_sample_count

    @property
    def window_step_samples(self) -> int:
        return self.system_config.fingerprinting.window_step_samples

    @property
    def baseline_capture_seconds(self) -> float:
        return self.system_config.fingerprinting.baseline_capture_seconds

    @property
    def baseline_start_delay_seconds(self) -> float:
        return self.system_config.fingerprinting.baseline_start_delay_seconds

    @property
    def baseline_required_for_training(self) -> bool:
        return self.system_config.fingerprinting.baseline_required_for_training

    @property
    def smoothing_half_window(self) -> int:
        return self.system_config.fingerprinting.smoothing_half_window

    @property
    def capture_auto_extend_seconds(self) -> float:
        return self.system_config.fingerprinting.capture_auto_extend_seconds

    @property
    def capture_extend_step_seconds(self) -> float:
        return self.system_config.fingerprinting.capture_extend_step_seconds

    @property
    def minimum_observed_windows(self) -> int:
        return self.system_config.fingerprinting.minimum_observed_windows

    @property
    def minimum_observed_window_ratio(self) -> float:
        return self.system_config.fingerprinting.minimum_observed_window_ratio

    @property
    def probability_smoothing_seconds(self) -> float:
        return self.system_config.fingerprinting.live_probability_smoothing_seconds

    @property
    def best_cell_switch_margin(self) -> float:
        return self.system_config.fingerprinting.best_cell_switch_margin

    @property
    def best_cell_switch_delay_seconds(self) -> float:
        return self.system_config.fingerprinting.best_cell_switch_delay_seconds

    @property
    def prediction_stale_grace_seconds(self) -> float:
        return self.system_config.fingerprinting.prediction_stale_grace_seconds

    @property
    def inference_interval_seconds(self) -> float:
        return max(self.window_step_seconds, self.INFERENCE_MIN_INTERVAL_SECONDS)

    @property
    def runtime_tick_seconds(self) -> float:
        return max(
            0.02,
            min(self.RUNTIME_MAX_TICK_SECONDS, self.inference_interval_seconds * 0.5),
        )

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

    def run_runtime_tick(self) -> None:
        with self.lock:
            self._run_runtime_tick_locked(time.time())

    def train_models(self) -> str:
        with self.lock:
            if self.capture_session is not None or self.baseline_capture_session is not None:
                raise RuntimeError("Wait for the current capture to finish first.")
            if self.baseline_required_for_training and not self._baseline_ready_locked():
                raise RuntimeError(
                    "Capture an empty-room baseline for every enabled ESP32 before training."
                )
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
            self._schedule_inference_now_locked()
            self.status_message = f"Active inference model set to {model_name}."
            self._save_models()
            self._log_event_locked("TRAIN", f"Active inference model set to {model_name}")

    def start_baseline_capture(self, *, reset_training: bool = False) -> None:
        with self.lock:
            now = time.time()
            self._advance_capture_locked(now)
            if self.capture_session is not None or self.baseline_capture_session is not None:
                raise RuntimeError("Another capture is already running.")
            if self.cell_datasets and not reset_training:
                raise RuntimeError(
                    "Re-capturing the baseline requires clearing the learned cell datasets first."
                )
            if reset_training:
                self._reset_training_state_locked(clear_baseline=False)
            ready_at = now + self.baseline_start_delay_seconds
            self.baseline_capture_session = BaselineCaptureSession(
                ready_at=ready_at,
                started_at=ready_at,
                ends_at=ready_at + self.baseline_capture_seconds,
            )
            if self.baseline_start_delay_seconds > 0.0:
                self.status_message = (
                    f"Baseline capture will start in {self.baseline_start_delay_seconds:.1f}s. "
                    f"Leave the room, then keep it empty for {self.baseline_capture_seconds:.1f}s."
                )
            else:
                self.status_message = (
                    "Started empty-room baseline capture. Leave the room empty for "
                    f"{self.baseline_capture_seconds:.1f}s."
                )
            self._log_event_locked(
                "BASELINE",
                f"Baseline capture armed with {self.baseline_start_delay_seconds:.1f}s delay "
                f"and {self.baseline_capture_seconds:.1f}s capture",
            )

    def clear_baseline(self) -> None:
        with self.lock:
            self.baseline_capture_session = None
            self.empty_room_baseline_by_node.clear()
            self.empty_room_baseline_counts.clear()
            self._save_datasets()
            if self.cell_datasets:
                self._reset_training_state_locked(clear_baseline=False)
                self.status_message = (
                    "Cleared the empty-room baseline and learned cell datasets. "
                    "Capture a new baseline before learning again."
                )
            else:
                self._clear_models_locked()
                self.status_message = (
                    "Cleared the empty-room baseline. Capture a new baseline before learning."
                )
            self._log_event_locked("BASELINE", "Cleared empty-room baseline")

    def apply_grid_settings(
        self,
        cols: int,
        rows: int,
        capture_seconds: float,
        window_seconds: float,
        window_step_seconds: float,
        baseline_start_delay_seconds: float | None = None,
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
            grid_changed = cols != self.grid_cols or rows != self.grid_rows
            window_changed = not math.isclose(window_seconds, self.window_seconds, abs_tol=1e-6)
            window_step_changed = not math.isclose(
                window_step_seconds,
                self.window_step_seconds,
                abs_tol=1e-6,
            )
            window_sample_count = max(
                1,
                int(round(window_seconds * self.effective_packets_per_second)),
            )
            window_step_samples = max(
                1,
                min(
                    int(round(window_step_seconds * self.effective_packets_per_second)),
                    window_sample_count,
                ),
            )
            if baseline_start_delay_seconds is None:
                baseline_start_delay_seconds = self.baseline_start_delay_seconds
            baseline_start_delay_seconds = max(0.0, float(baseline_start_delay_seconds))

            self.system_config.grid.cols = cols
            self.system_config.grid.rows = rows
            self.system_config.fingerprinting.capture_seconds = capture_seconds
            self.system_config.fingerprinting.window_sample_count = window_sample_count
            self.system_config.fingerprinting.window_step_samples = window_step_samples
            self.system_config.fingerprinting.window_seconds = (
                float(window_sample_count) / self.effective_packets_per_second
            )
            self.system_config.fingerprinting.window_step_seconds = (
                float(window_step_samples) / self.effective_packets_per_second
            )
            self.system_config.fingerprinting.baseline_start_delay_seconds = (
                baseline_start_delay_seconds
            )
            save_system_config(self.config_path, self.system_config)
            self._schedule_inference_now_locked()

            if grid_changed or window_changed or window_step_changed:
                self._reset_training_state_locked(clear_baseline=False)
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
                    f"window_seconds to {self.system_config.fingerprinting.window_seconds:.2f}s "
                    f"({window_sample_count} samples), "
                    f"window_step_seconds to {self.system_config.fingerprinting.window_step_seconds:.2f}s "
                    f"(stride {window_step_samples}), "
                    f"baseline_start_delay_seconds to {baseline_start_delay_seconds:.2f}s",
                )

    def start_capture(self, grid_x: int, grid_y: int) -> None:
        with self.lock:
            now = time.time()
            self._advance_capture_locked(now)
            if self.baseline_capture_session is not None:
                raise RuntimeError("Wait for the current baseline capture to finish first.")
            if self.baseline_required_for_training and not self._baseline_ready_locked():
                raise RuntimeError(
                    "Capture an empty-room baseline before collecting cell fingerprints."
                )
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
                max_ends_at=now + self.capture_seconds + self.capture_auto_extend_seconds,
            )
            self.status_message = (
                f"Started capture for cell ({grid_x + 1}, {grid_y + 1}). "
                f"Hold position for {self.capture_seconds:.1f}s. "
                f"Window size: {self.window_sample_count} samples (~{self.window_seconds:.2f}s). "
                f"Step: {self.window_step_samples} sample (~{self.window_step_seconds:.2f}s)."
            )
            self._log_event_locked(
                "CAPTURE",
                f"Capture started for cell ({grid_x + 1}, {grid_y + 1}) for "
                f"{self.capture_seconds:.1f}s with {self.window_sample_count}-sample windows "
                f"and stride {self.window_step_samples}",
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
            self._reset_training_state_locked(clear_baseline=False)
            self.status_message = (
                "Cleared all saved Learn data and all trained models. The baseline was kept."
            )
            self._log_event_locked("CAPTURE", "Cleared all saved training data and models")

    def process_packet(self, payload: bytes, source: str) -> bool:
        frame = parse_adr018_frame(payload)
        if frame is None:
            return self.process_receiver_event(
                {
                    "type": "packet",
                    "valid": False,
                    "source": source,
                    "received_at": time.time(),
                }
            )
        now = time.time()
        feature = build_feature_frame(frame, source, now, self.feature_bin_count)
        return self.process_receiver_event(
            {
                "type": "packet",
                "valid": True,
                "source": source,
                "received_at": now,
                "feature": feature,
            }
        )

    def process_receiver_event(self, event: dict[str, object]) -> bool:
        now = float(event.get("received_at", time.time()))
        if not event.get("valid"):
            source = str(event.get("source", "unknown"))
            with self.lock:
                if now - self.last_invalid_packet_log_ts >= 5.0:
                    self.last_invalid_packet_log_ts = now
                    self._log_event_locked(
                        "WARN",
                        f"Ignored non-ADR018 or malformed packet from {source}",
                    )
            return False

        feature = event.get("feature")
        if not isinstance(feature, FeatureFrame):
            return False
        with self.lock:
            self.packet_count += 1
            self.last_packet_ts = now

            node_window = self.node_windows.setdefault(feature.node_id, deque())
            node_window.append(feature)
            self._prune_node_window_locked(feature.node_id, now)

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
            node_state.rssi_dbm = feature.rssi_dbm
            node_state.noise_floor_dbm = feature.noise_floor_dbm
            node_state.snr_db = feature.snr_db
            node_state.subcarrier_count = feature.n_subcarriers
            if (
                node_state.packets_received == 1
                or now - node_state.last_status_log_ts >= self.UDP_STATUS_LOG_INTERVAL_SECONDS
            ):
                node_state.last_status_log_ts = now
                self._log_event_locked(
                    "UDP",
                    f"Node {feature.node_id} seq={feature.sequence} "
                    f"packets={node_state.packets_received} "
                    f"rssi={feature.rssi_dbm:.1f} snr={feature.snr_db:.1f} source={source}",
                )

            if (
                self.baseline_capture_session is not None
                and self.baseline_capture_session.started_at <= now <= self.baseline_capture_session.ends_at
            ):
                self.baseline_capture_session.frames_by_node.setdefault(feature.node_id, []).append(
                    feature
                )

            if self.capture_session is not None and now <= self.capture_session.ends_at:
                self.capture_session.frames_by_node.setdefault(feature.node_id, []).append(
                    feature
                )

            self._mark_inference_dirty_locked()
        return True

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            now = time.time()
            self._advance_capture_locked(now)
            self._prune_all_live_windows_locked(now)
            self._refresh_live_node_telemetry_locked(now)

            capture_payload: dict[str, object] = {
                "active": False,
                "kind": None,
                "label": "",
                "started": False,
                "grid_x": None,
                "grid_y": None,
                "remaining_seconds": 0.0,
                "delay_remaining_seconds": 0.0,
                "progress": [],
            }
            if self.baseline_capture_session is not None:
                baseline_started = now >= self.baseline_capture_session.started_at
                capture_payload = {
                    "active": True,
                    "kind": "baseline",
                    "label": (
                        "Empty-room baseline"
                        if baseline_started
                        else "Empty-room baseline (waiting)"
                    ),
                    "started": baseline_started,
                    "grid_x": None,
                    "grid_y": None,
                    "remaining_seconds": max(
                        0.0,
                        (
                            self.baseline_capture_session.ends_at - now
                            if baseline_started
                            else self.baseline_capture_seconds
                        ),
                    ),
                    "delay_remaining_seconds": max(
                        0.0,
                        self.baseline_capture_session.started_at - now,
                    ),
                    "progress": [
                        {
                            "node_id": node_id,
                            "sample_count": len(frames),
                        }
                        for node_id, frames in sorted(
                            self.baseline_capture_session.frames_by_node.items()
                        )
                    ],
                }
            elif self.capture_session is not None:
                capture_payload = {
                    "active": True,
                    "kind": "cell",
                    "label": f"Cell ({self.capture_session.grid_x + 1}, {self.capture_session.grid_y + 1})",
                    "started": True,
                    "grid_x": self.capture_session.grid_x,
                    "grid_y": self.capture_session.grid_y,
                    "remaining_seconds": max(0.0, self.capture_session.ends_at - now),
                    "delay_remaining_seconds": 0.0,
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
                            "observed_window_count": dataset.observed_window_count
                            if dataset
                            else 0,
                            "generated_window_count": dataset.generated_window_count
                            if dataset
                            else 0,
                            "observed_window_ratio": (
                                float(dataset.observed_window_count)
                                / max(1, dataset.total_window_slots)
                            )
                            if dataset
                            else 0.0,
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
            baseline_ready = self._baseline_ready_locked()

            return {
                "host": {
                    "listen_host": self.system_config.host.listen_host,
                    "target_ip": self.system_config.host.target_ip,
                    "udp_port": self.system_config.host.udp_port,
                    "node_ports": [
                        {
                            "node_id": node.node_id,
                            "port": self.system_config.node_target_port(node),
                        }
                        for node in self.system_config.enabled_nodes()
                    ],
                    "config_path": str(self.config_path),
                },
                "grid": {
                    "cols": self.grid_cols,
                    "rows": self.grid_rows,
                    "capture_seconds": self.capture_seconds,
                    "window_seconds": self.window_seconds,
                    "window_step_seconds": self.window_step_seconds,
                    "effective_packets_per_second": self.effective_packets_per_second,
                    "window_sample_count": self.window_sample_count,
                    "window_step_samples": self.window_step_samples,
                },
                "training": {
                    "trained_cells": trained_cells,
                    "total_cells": self.total_cells,
                    "dataset_samples": dataset_samples,
                    "required_nodes": len(self.required_node_ids),
                    "baseline_ready": baseline_ready,
                    "model_ready": active_model_ready,
                    "can_train": self._can_train_locked(),
                    "active_model": self.active_model_name,
                    "available_models": available_models,
                    "trained_model_count": len(available_models),
                    "ready_for_inference": live_prediction_ready,
                },
                "baseline": {
                    "ready": baseline_ready,
                    "required": self.baseline_required_for_training,
                    "capture_seconds": self.baseline_capture_seconds,
                    "start_delay_seconds": self.baseline_start_delay_seconds,
                    "captured_nodes": len(
                        [
                            node_id
                            for node_id in self.required_node_ids
                            if node_id in self.empty_room_baseline_by_node
                        ]
                    ),
                    "required_nodes": len(self.required_node_ids),
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
                    "inference_interval_seconds": self.inference_interval_seconds,
                    "last_inference_duration_ms": self.last_inference_duration_ms,
                    "last_inference_age_ms": None
                    if self.last_inference_completed_ts is None
                    else max(0.0, (now - self.last_inference_completed_ts) * 1000.0),
                    "inference_cycle_count": self.inference_cycle_count,
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

    def _run_runtime_tick_locked(self, now: float) -> None:
        self._advance_capture_locked(now)
        self._prune_all_live_windows_locked(now)
        self._refresh_live_node_telemetry_locked(now)
        self._maybe_recompute_probabilities_locked(now)

    def _advance_capture_locked(self, now: float) -> None:
        if self.baseline_capture_session is not None:
            if (
                self.baseline_capture_session.ready_at > 0.0
                and now >= self.baseline_capture_session.ready_at
            ):
                self.baseline_capture_session.ready_at = 0.0
                self.status_message = (
                    "Baseline capture started. Keep the room empty for "
                    f"{self.baseline_capture_seconds:.1f}s."
                )
                self._log_event_locked(
                    "BASELINE",
                    f"Baseline capture started for {self.baseline_capture_seconds:.1f}s",
                )
            if now >= self.baseline_capture_session.ends_at:
                session = self.baseline_capture_session
                self.baseline_capture_session = None
                self._finalize_baseline_capture_locked(session, now)

        if self.capture_session is None or now < self.capture_session.ends_at:
            return
        session = self.capture_session
        build_result = self._build_dataset_samples_locked(session)
        if self._capture_should_extend_locked(session, build_result):
            self._extend_capture_session_locked(session, build_result)
            return
        self.capture_session = None

        if not build_result.samples:
            self.status_message = (
                f"Capture failed for cell ({session.grid_x + 1}, {session.grid_y + 1}): "
                f"no complete {self.window_sample_count}-sample windows at "
                f"stride {self.window_step_samples} with all "
                f"{len(self.required_node_ids)} required nodes."
            )
            self._log_event_locked(
                "CAPTURE",
                f"Capture failed for cell ({session.grid_x + 1}, {session.grid_y + 1}); "
                f"observed windows={build_result.observed_window_count}/"
                f"{build_result.total_window_slots} observed nodes={build_result.observed_node_count}/"
                f"{len(self.required_node_ids)}",
            )
            return

        required_observed_windows = self._required_observed_window_count_locked(
            build_result.total_window_slots
        )
        if self._capture_quality_is_insufficient_locked(build_result):
            self.status_message = (
                f"Capture failed for cell ({session.grid_x + 1}, {session.grid_y + 1}): "
                f"only {build_result.observed_window_count}/{build_result.total_window_slots} "
                f"fully observed windows; need at least {required_observed_windows}. "
                "Try the cell again or improve signal coverage."
            )
            self._log_event_locked(
                "CAPTURE",
                f"Capture quality too low for cell ({session.grid_x + 1}, {session.grid_y + 1}); "
                f"observed windows={build_result.observed_window_count}/"
                f"{build_result.total_window_slots} required={required_observed_windows} "
                f"generated={build_result.generated_window_count} frames={build_result.total_frames}",
            )
            return

        previous = self.cell_datasets.get(session.cell_key)
        merged_samples = build_result.samples
        merged_total_frames = build_result.total_frames
        merged_capture_count = 1
        merged_observed_window_count = build_result.observed_window_count
        merged_generated_window_count = build_result.generated_window_count
        merged_total_window_slots = build_result.total_window_slots
        if previous is not None:
            merged_samples = previous.samples + build_result.samples
            merged_total_frames = previous.total_frames + build_result.total_frames
            merged_capture_count = previous.capture_count + 1
            merged_observed_window_count = (
                previous.observed_window_count + build_result.observed_window_count
            )
            merged_generated_window_count = (
                previous.generated_window_count + build_result.generated_window_count
            )
            merged_total_window_slots = (
                previous.total_window_slots + build_result.total_window_slots
            )
        self.cell_datasets[session.cell_key] = CellDataset(
            cell_key=session.cell_key,
            grid_x=session.grid_x,
            grid_y=session.grid_y,
            captured_at=now,
            total_frames=merged_total_frames,
            capture_count=merged_capture_count,
            node_count=len(self.required_node_ids),
            window_sample_count=len(merged_samples),
            observed_window_count=merged_observed_window_count,
            generated_window_count=merged_generated_window_count,
            total_window_slots=merged_total_window_slots,
            samples=merged_samples,
        )
        self._save_datasets()
        self._clear_models_locked()

        message = (
            f"Capture completed for cell ({session.grid_x + 1}, {session.grid_y + 1}): "
            f"{build_result.observed_window_count}/{build_result.total_window_slots} observed "
            f"windows, {build_result.generated_window_count} generated windows kept, "
            f"{build_result.total_frames} frames across {build_result.observed_node_count}/"
            f"{len(self.required_node_ids)} required nodes. "
            f"Cell now has {len(merged_samples)} total windows across "
            f"{merged_capture_count} captures. Click Train Models when ready."
        )
        self.status_message = message
        self._log_event_locked(
            "CAPTURE",
            f"Capture completed for cell ({session.grid_x + 1}, {session.grid_y + 1}) "
            f"with {build_result.observed_window_count}/{build_result.total_window_slots} "
            f"observed windows, generated={build_result.generated_window_count} "
            f"frames={build_result.total_frames}; accumulated windows={len(merged_samples)} "
            f"captures={merged_capture_count}",
        )

    def _finalize_baseline_capture_locked(
        self,
        session: BaselineCaptureSession,
        captured_at: float,
    ) -> None:
        required_node_ids = self.required_node_ids
        captured_nodes = 0
        total_frames = 0
        updated_baselines: dict[int, list[float]] = {}
        updated_counts: dict[int, int] = {}

        for node_id in required_node_ids:
            frames = session.frames_by_node.get(node_id, [])
            total_frames += len(frames)
            vectors = self._resample_node_vectors_locked(
                frames,
                start_time=session.started_at,
                end_time=session.ends_at,
            )
            if not vectors:
                continue
            captured_nodes += 1
            updated_baselines[node_id] = self._vector_mean(vectors)
            updated_counts[node_id] = len(vectors)

        if captured_nodes < len(required_node_ids):
            self.status_message = (
                "Baseline capture failed: missing CSI frames from one or more enabled ESP32 nodes."
            )
            self._log_event_locked(
                "BASELINE",
                f"Baseline capture failed; captured_nodes={captured_nodes}/{len(required_node_ids)} "
                f"frames={total_frames}",
            )
            return

        self.empty_room_baseline_by_node.update(updated_baselines)
        self.empty_room_baseline_counts.update(updated_counts)
        self._save_datasets()
        self.status_message = (
            f"Baseline capture completed: {captured_nodes}/{len(required_node_ids)} nodes, "
            f"{total_frames} total frames. Press Learn on each cell."
        )
        self._log_event_locked(
            "BASELINE",
            f"Baseline capture completed with {captured_nodes}/{len(required_node_ids)} nodes "
            f"and {total_frames} total frames",
        )

    def _build_dataset_samples_locked(
        self, session: CaptureSession
    ) -> DatasetBuildResult:
        required_node_ids = self.required_node_ids
        if not required_node_ids:
            return DatasetBuildResult()

        total_frames = 0
        observed_node_ids: set[int] = set()
        for node_id in required_node_ids:
            frames = session.frames_by_node.get(node_id, [])
            total_frames += len(frames)
            if frames:
                observed_node_ids.add(node_id)

        result = DatasetBuildResult(
            total_frames=total_frames,
            observed_node_count=len(observed_node_ids),
        )

        (
            aligned_vectors_by_node,
            total_resampled_slots,
            valid_resampled_slots,
        ) = self._build_aligned_node_vectors_locked(
            session.frames_by_node,
            start_time=session.started_at,
            end_time=session.ends_at,
        )
        result.total_resampled_slots = total_resampled_slots
        result.valid_resampled_slots = valid_resampled_slots
        if total_resampled_slots <= 0:
            return result

        features_by_node: dict[int, list[list[float]]] = {}
        window_counts: list[int] = []
        for node_id in required_node_ids:
            features_by_node[node_id] = self._build_window_feature_vectors_from_vectors(
                aligned_vectors_by_node.get(node_id, []),
                node_id,
            )
            window_counts.append(len(features_by_node[node_id]))

        result.total_window_slots = self._count_window_slots(total_resampled_slots)
        result.observed_window_count = self._count_window_slots(valid_resampled_slots)
        result.generated_window_count = min(window_counts) if window_counts else 0
        if result.generated_window_count <= 0:
            return result

        samples: list[list[float]] = []
        for index in range(result.generated_window_count):
            sample: list[float] = []
            for node_id in required_node_ids:
                sample.extend(features_by_node[node_id][index])
            samples.append(sample)

        result.samples = samples
        return result

    def _required_observed_window_count_locked(self, total_window_slots: int) -> int:
        if total_window_slots <= 0:
            return 0
        ratio_target = int(
            math.ceil(total_window_slots * self.minimum_observed_window_ratio)
        )
        return min(
            total_window_slots,
            max(self.minimum_observed_windows, ratio_target),
        )

    def _capture_quality_is_insufficient_locked(
        self,
        build_result: DatasetBuildResult,
    ) -> bool:
        if build_result.observed_node_count < len(self.required_node_ids):
            return True
        required_observed_windows = self._required_observed_window_count_locked(
            build_result.total_window_slots
        )
        return build_result.observed_window_count < required_observed_windows

    def _capture_should_extend_locked(
        self,
        session: CaptureSession,
        build_result: DatasetBuildResult,
    ) -> bool:
        if not self._capture_quality_is_insufficient_locked(build_result):
            return False
        return session.max_ends_at > session.ends_at + 1e-6

    def _extend_capture_session_locked(
        self,
        session: CaptureSession,
        build_result: DatasetBuildResult,
    ) -> None:
        remaining_extension = max(0.0, session.max_ends_at - session.ends_at)
        if remaining_extension <= 0.0:
            return
        extension_seconds = min(self.capture_extend_step_seconds, remaining_extension)
        session.ends_at += extension_seconds
        session.extension_count += 1
        required_observed_windows = self._required_observed_window_count_locked(
            build_result.total_window_slots
        )
        self.status_message = (
            f"Extending capture for cell ({session.grid_x + 1}, {session.grid_y + 1}) by "
            f"{extension_seconds:.1f}s because only "
            f"{build_result.observed_window_count}/{build_result.total_window_slots} "
            f"observed windows are available (need {required_observed_windows})."
        )
        self._log_event_locked(
            "CAPTURE",
            f"Extended capture for cell ({session.grid_x + 1}, {session.grid_y + 1}) by "
            f"{extension_seconds:.1f}s; observed windows="
            f"{build_result.observed_window_count}/{build_result.total_window_slots} "
            f"required={required_observed_windows} extensions={session.extension_count}",
        )

    def _build_window_feature_vectors(
        self,
        frames: list[FeatureFrame],
        node_id: int,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[list[float]]:
        vectors = self._resample_node_vectors_locked(
            frames,
            start_time=start_time,
            end_time=end_time,
        )
        return self._build_window_feature_vectors_from_vectors(vectors, node_id)

    def _build_window_feature_vectors_from_vectors(
        self,
        vectors: list[list[float]],
        node_id: int,
    ) -> list[list[float]]:
        if len(vectors) < self.window_sample_count:
            return []

        preprocessed = self._preprocess_node_vectors_locked(node_id, vectors)
        features: list[list[float]] = []
        for start in range(
            0,
            len(preprocessed) - self.window_sample_count + 1,
            self.window_step_samples,
        ):
            window_vectors = preprocessed[start : start + self.window_sample_count]
            means = self._vector_mean(window_vectors)
            stds = self._vector_std(window_vectors, means)
            features.append([*means, *stds])
        return features

    def _resample_node_vectors_locked(
        self,
        frames: list[FeatureFrame],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[list[float]]:
        if not frames:
            return []
        sorted_frames = sorted(frames, key=lambda item: item.captured_at)
        if start_time is None:
            start_time = sorted_frames[0].captured_at
        if end_time is None:
            end_time = sorted_frames[-1].captured_at
        bucketed = self._bucketize_frames_locked(
            sorted_frames,
            start_time=start_time,
            end_time=end_time,
        )
        return [vector for vector in bucketed if vector is not None]

    def _build_aligned_node_vectors_locked(
        self,
        frames_by_node: dict[int, list[FeatureFrame]],
        *,
        start_time: float,
        end_time: float,
    ) -> tuple[dict[int, list[list[float]]], int, int]:
        required_node_ids = self.required_node_ids
        aligned_by_node = {
            node_id: [] for node_id in required_node_ids
        }
        if not required_node_ids:
            return aligned_by_node, 0, 0

        bucketed_by_node: dict[int, list[list[float] | None]] = {}
        total_slots = 0
        for node_id in required_node_ids:
            bucketed = self._bucketize_frames_locked(
                frames_by_node.get(node_id, []),
                start_time=start_time,
                end_time=end_time,
            )
            bucketed_by_node[node_id] = bucketed
            total_slots = max(total_slots, len(bucketed))

        if total_slots <= 0:
            return aligned_by_node, 0, 0

        valid_slots = 0
        for index in range(total_slots):
            if any(
                index >= len(bucketed_by_node[node_id])
                or bucketed_by_node[node_id][index] is None
                for node_id in required_node_ids
            ):
                continue
            valid_slots += 1

        filled_by_node: dict[int, list[list[float]]] = {}
        for node_id in required_node_ids:
            filled_by_node[node_id] = self._fill_bucket_gaps_locked(
                node_id,
                bucketed_by_node.get(node_id, []),
                total_slots=total_slots,
            )

        for index in range(total_slots):
            for node_id in required_node_ids:
                aligned_by_node[node_id].append(filled_by_node[node_id][index])
        return aligned_by_node, total_slots, valid_slots

    def _fill_bucket_gaps_locked(
        self,
        node_id: int,
        bucketed: list[list[float] | None],
        *,
        total_slots: int,
    ) -> list[list[float]]:
        normalized: list[list[float] | None] = list(bucketed[:total_slots])
        if len(normalized) < total_slots:
            normalized.extend([None] * (total_slots - len(normalized)))

        observed = [vector for vector in normalized if vector is not None]
        default_vector = self._default_imputation_vector_locked(node_id, observed)

        last_seen: list[float] | None = None
        for index, vector in enumerate(normalized):
            if vector is not None:
                last_seen = vector
                continue
            if last_seen is not None:
                normalized[index] = list(last_seen)

        next_seen: list[float] | None = None
        for index in range(total_slots - 1, -1, -1):
            vector = normalized[index]
            if vector is not None:
                next_seen = vector
                continue
            if next_seen is not None:
                normalized[index] = list(next_seen)

        filled: list[list[float]] = []
        for vector in normalized:
            filled.append(list(vector) if vector is not None else list(default_vector))
        return filled

    def _default_imputation_vector_locked(
        self,
        node_id: int,
        observed_vectors: list[list[float]],
    ) -> list[float]:
        if observed_vectors:
            return self._vector_mean(observed_vectors)

        baseline = self.empty_room_baseline_by_node.get(node_id)
        if baseline:
            return list(baseline)

        recent_window = self.node_windows.get(node_id)
        if recent_window:
            for frame in reversed(recent_window):
                if len(frame.feature_vector) == ACTIVE_SUBCARRIER_COUNT:
                    return list(frame.feature_vector)

        return [0.0] * ACTIVE_SUBCARRIER_COUNT

    def _bucketize_frames_locked(
        self,
        frames: list[FeatureFrame],
        *,
        start_time: float,
        end_time: float,
    ) -> list[list[float] | None]:
        if end_time <= start_time:
            return []
        bucket_count = max(
            1,
            int(round((end_time - start_time) * self.effective_packets_per_second)),
        )
        buckets: list[list[list[float]]] = [[] for _ in range(bucket_count)]
        for frame in frames:
            if len(frame.feature_vector) != ACTIVE_SUBCARRIER_COUNT:
                continue
            if frame.captured_at < start_time or frame.captured_at > end_time:
                continue
            position = (frame.captured_at - start_time) * self.effective_packets_per_second
            bucket_index = int(position)
            if bucket_index < 0:
                continue
            if bucket_index >= bucket_count:
                if math.isclose(frame.captured_at, end_time, abs_tol=1e-9):
                    bucket_index = bucket_count - 1
                else:
                    continue
            buckets[bucket_index].append(frame.feature_vector)

        aggregated: list[list[float] | None] = []
        for bucket in buckets:
            aggregated.append(self._vector_mean(bucket) if bucket else None)
        return aggregated

    def _count_window_slots(self, sample_count: int) -> int:
        if sample_count < self.window_sample_count:
            return 0
        return 1 + (sample_count - self.window_sample_count) // self.window_step_samples

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
        return self._smooth_vectors(normalized, self.smoothing_half_window)

    def _baseline_vector_for_node_locked(
        self,
        node_id: int,
        vectors: list[list[float]],
    ) -> list[float]:
        cached = self.empty_room_baseline_by_node.get(node_id)
        if cached is not None and len(cached) == len(vectors[0]):
            return cached
        return self._vector_mean(vectors)

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
                self.prediction_stale_grace_seconds,
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

        if self.probability_smoothing_seconds <= 0.0:
            return dict(raw_probabilities)

        smoothing_seconds = max(
            self.probability_smoothing_seconds,
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
        if self.best_cell_switch_delay_seconds <= 0.0:
            switch_delay = 0.0
        else:
            switch_delay = max(
                self.best_cell_switch_delay_seconds,
                self.window_seconds * 0.75,
            )
        if (
            candidate_probability
            >= current_probability + self.best_cell_switch_margin
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

    def _mark_inference_dirty_locked(self) -> None:
        self._inference_dirty = True

    def _schedule_inference_now_locked(self) -> None:
        self._inference_dirty = True
        self._next_inference_ts = 0.0

    def _maybe_recompute_probabilities_locked(self, now: float) -> None:
        if now < self._next_inference_ts:
            return
        if not self._inference_dirty and not self.last_probabilities:
            self._next_inference_ts = now + self.inference_interval_seconds
            return
        started_at = time.perf_counter()
        self._recompute_probabilities_locked(now)
        self.last_inference_duration_ms = (time.perf_counter() - started_at) * 1000.0
        self.last_inference_completed_ts = now
        self.inference_cycle_count += 1
        self._inference_dirty = False
        self._next_inference_ts = now + self.inference_interval_seconds

    def _build_live_sample_locked(self, now: float) -> list[float] | None:
        required_node_ids = self.required_node_ids
        if not required_node_ids:
            return None
        if self.baseline_required_for_training and not self._baseline_ready_locked():
            return None

        required_recent_samples = self.window_sample_count + self.smoothing_half_window * 2
        horizon_sample_count = max(
            self.window_sample_count + self.smoothing_half_window * 2,
            self.window_sample_count * self.LIVE_PREPROCESS_HORIZON_WINDOWS,
        )
        horizon_seconds = (
            float(horizon_sample_count) / self.effective_packets_per_second
        )
        start_time = now - horizon_seconds
        aligned_frames_by_node = {
            node_id: list(self.node_windows.get(node_id, []))
            for node_id in required_node_ids
        }
        (
            aligned_vectors_by_node,
            _total_resampled_slots,
            valid_resampled_slots,
        ) = self._build_aligned_node_vectors_locked(
            aligned_frames_by_node,
            start_time=start_time,
            end_time=now,
        )
        if valid_resampled_slots < required_recent_samples:
            return None

        sample: list[float] = []
        for node_id in required_node_ids:
            aligned_vectors = aligned_vectors_by_node.get(node_id, [])
            per_node_features = self._build_window_feature_vectors_from_vectors(
                aligned_vectors,
                node_id,
            )
            if not per_node_features:
                return None
            sample.extend(per_node_features[-1])

        if len(sample) != self.expected_input_size:
            return None
        return sample

    def _baseline_ready_locked(self) -> bool:
        required_node_ids = self.required_node_ids
        if not required_node_ids:
            return False
        return all(
            node_id in self.empty_room_baseline_by_node
            and len(self.empty_room_baseline_by_node[node_id]) == ACTIVE_SUBCARRIER_COUNT
            for node_id in required_node_ids
        )

    def _can_train_locked(self) -> bool:
        ordered_cell_keys = self._ordered_cell_keys()
        if (
            self.baseline_required_for_training
            and not self._baseline_ready_locked()
        ):
            return False
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
        self._schedule_inference_now_locked()
        self._save_models()
        self.status_message = (
            f"Trained {len(trained_models)} models on {sample_count} samples. "
            f"Active model: {self.active_model_name}."
        )
        return self.status_message

    def _load_datasets(self) -> None:
        raw = load_fingerprint_store(self.fingerprint_path)
        self._loaded_store_version = (
            int(raw["version"]) if "version" in raw else None
        )
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
        baseline_raw = raw.get("baseline", {})
        if isinstance(baseline_raw, dict):
            vectors_raw = baseline_raw.get("vectors", {})
            counts_raw = baseline_raw.get("counts", {})
            if isinstance(vectors_raw, dict):
                self.empty_room_baseline_by_node = {
                    int(node_id): [float(value) for value in vector]
                    for node_id, vector in vectors_raw.items()
                    if isinstance(vector, list)
                }
            if isinstance(counts_raw, dict):
                self.empty_room_baseline_counts = {
                    int(node_id): int(count)
                    for node_id, count in counts_raw.items()
                }

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
                observed_window_count=int(
                    payload.get("observed_window_count", len(samples))
                ),
                generated_window_count=int(
                    payload.get("generated_window_count", len(samples))
                ),
                total_window_slots=int(payload.get("total_window_slots", len(samples))),
                samples=samples,
            )

    def _save_datasets(self) -> None:
        payload = {
            "version": self.STORE_VERSION,
            "window_seconds": self.window_seconds,
            "window_step_seconds": self.window_step_seconds,
            "node_ids": self.required_node_ids,
            "input_size": self.expected_input_size,
            "baseline": {
                "vectors": {
                    str(node_id): vector
                    for node_id, vector in sorted(self.empty_room_baseline_by_node.items())
                },
                "counts": {
                    str(node_id): count
                    for node_id, count in sorted(self.empty_room_baseline_counts.items())
                },
            },
            "cells": {
                cell_key: {
                    "grid_x": cell.grid_x,
                    "grid_y": cell.grid_y,
                    "captured_at": cell.captured_at,
                    "total_frames": cell.total_frames,
                    "capture_count": cell.capture_count,
                    "node_count": cell.node_count,
                    "window_sample_count": cell.window_sample_count,
                    "observed_window_count": cell.observed_window_count,
                    "generated_window_count": cell.generated_window_count,
                    "total_window_slots": cell.total_window_slots,
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
        mismatch_reasons: list[str] = []
        config_matches = True
        if (
            self._loaded_store_version is not None
            and self._loaded_store_version != self.STORE_VERSION
        ):
            config_matches = False
            mismatch_reasons.append(
                f"stored dataset version {self._loaded_store_version} != {self.STORE_VERSION}"
            )
        if self._loaded_store_window_seconds is not None and not math.isclose(
            self._loaded_store_window_seconds,
            self.window_seconds,
            abs_tol=1e-6,
        ):
            config_matches = False
            mismatch_reasons.append("window seconds changed")
        if self._loaded_store_window_step_seconds is not None:
            step_matches = math.isclose(
                self._loaded_store_window_step_seconds,
                self.window_step_seconds,
                abs_tol=1e-6,
            )
            config_matches = config_matches and step_matches
            if not step_matches:
                mismatch_reasons.append("window step changed")
        if self._loaded_store_node_ids is not None:
            node_matches = self._loaded_store_node_ids == self.required_node_ids
            config_matches = config_matches and node_matches
            if not node_matches:
                mismatch_reasons.append("enabled node set changed")
        if self._loaded_store_input_size is not None:
            input_matches = self._loaded_store_input_size == self.expected_input_size
            config_matches = config_matches and input_matches
            if not input_matches:
                mismatch_reasons.append("input size changed")

        if not config_matches:
            if self.cell_datasets:
                self._log_event_locked(
                    "CFG",
                    "Cleared saved training data because "
                    + ", ".join(mismatch_reasons or ["the stored dataset no longer matches the app"]),
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

        valid_node_ids = set(self.required_node_ids)
        filtered_baselines = {
            node_id: vector
            for node_id, vector in self.empty_room_baseline_by_node.items()
            if node_id in valid_node_ids and len(vector) == ACTIVE_SUBCARRIER_COUNT
        }
        filtered_counts = {
            node_id: count
            for node_id, count in self.empty_room_baseline_counts.items()
            if node_id in filtered_baselines
        }
        if (
            filtered_baselines != self.empty_room_baseline_by_node
            or filtered_counts != self.empty_room_baseline_counts
        ):
            self.empty_room_baseline_by_node = filtered_baselines
            self.empty_room_baseline_counts = filtered_counts
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

    def _reset_training_state_locked(self, *, clear_baseline: bool) -> None:
        self.baseline_capture_session = None
        self.capture_session = None
        self.cell_datasets.clear()
        self.node_windows.clear()
        if clear_baseline:
            self.empty_room_baseline_by_node.clear()
            self.empty_room_baseline_counts.clear()
        self._save_datasets()
        self._clear_models_locked()

    def _clear_models_locked(self) -> None:
        self.model_pipelines = {}
        self.model_metadata_by_name = {}
        self.active_model_name = None
        self._clear_prediction_locked()
        self._schedule_inference_now_locked()
        remove_store(self.model_path)

    def _clear_prediction_locked(self) -> None:
        self.last_probabilities = {}
        self.last_prediction_ts = None
        self.last_best_cell = None
        self.last_best_probability = 0.0
        self.pending_best_cell = None
        self.pending_best_since = None

    def _refresh_live_node_telemetry_locked(self, now: float) -> None:
        if (
            self._last_node_telemetry_refresh_ts > 0.0
            and now - self._last_node_telemetry_refresh_ts < self.NODE_TELEMETRY_REFRESH_SECONDS
        ):
            return
        for node_id, node_state in self.live_nodes.items():
            node_state.window_samples = self._count_recent_frames(
                self.node_windows.get(node_id),
                now,
            )
        self._last_node_telemetry_refresh_ts = now

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

    def _count_recent_frames(self, window: deque[FeatureFrame] | None, now: float) -> int:
        if not window:
            return 0
        window_start = now - self.window_seconds
        bucketed = self._bucketize_frames_locked(
            list(window),
            start_time=window_start,
            end_time=now,
        )
        return sum(1 for vector in bucketed if vector is not None)

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
