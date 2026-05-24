from __future__ import annotations

import base64
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from Config.config_loader import load_system_config, save_system_config

from .protocol import FeatureFrame, ParsedFrame, build_feature_frame, parse_adr018_frame


@dataclass
class RawNodeState:
    node_id: int
    label: str
    source: str = ""
    last_seen_ts: float = 0.0
    last_status_log_ts: float = 0.0
    last_sequence: int = 0
    packets_received: int = 0
    rssi_dbm: float = 0.0
    noise_floor_dbm: float = 0.0
    snr_db: float = 0.0
    subcarrier_count: int = 0


@dataclass
class RawCaptureSession:
    session_id: str
    kind: str
    label: str
    grid_x: int | None
    grid_y: int | None
    armed_at: float
    started_at: float
    ends_at: float | None
    path: Path
    handle: TextIO
    start_written: bool = False
    valid_packets: int = 0
    invalid_packets: int = 0
    packets_by_node: dict[int, int] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)
    metadata: dict[str, object] = field(default_factory=dict)


class RawCaptureEngine:
    RUNTIME_TICK_SECONDS = 0.10
    UDP_STATUS_LOG_INTERVAL_SECONDS = 2.0

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.config_path = self.workspace_root.parent / "Config" / "system_config.json"
        self.raw_data_dir = self.workspace_root / "raw_data"
        self.comm_log_path = self.raw_data_dir / "raw_capture_communication.log"
        self.session_index_path = self.raw_data_dir / "sessions.jsonl"
        self.lock = threading.RLock()
        self.system_config = load_system_config(self.config_path)
        self.live_nodes: dict[int, RawNodeState] = {}
        self.capture_session: RawCaptureSession | None = None
        self.packet_count = 0
        self.saved_packet_count = 0
        self.last_packet_ts: float | None = None
        self.last_invalid_packet_log_ts = 0.0
        self.udp_status = (
            f"Ready for UDP {self.system_config.host.listen_host}:{self.system_config.host.udp_port}"
        )
        self.status_message = (
            "Press Empty Room or a cell Learn button to save raw ESP packets to JSONL."
        )
        self.comm_logs: deque[str] = deque(maxlen=250)

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
    def capture_seconds(self) -> float:
        return self.system_config.fingerprinting.capture_seconds

    @property
    def empty_room_delay_seconds(self) -> float:
        return self.system_config.fingerprinting.baseline_start_delay_seconds

    @property
    def runtime_tick_seconds(self) -> float:
        return self.RUNTIME_TICK_SECONDS

    def set_udp_status(self, message: str) -> None:
        with self.lock:
            self.udp_status = message
            self._log_event_locked("UDP", message)

    def run_runtime_tick(self) -> None:
        with self.lock:
            self._advance_capture_locked(time.time())

    def apply_grid_settings(
        self,
        cols: int,
        rows: int,
        capture_seconds: float,
        empty_room_delay_seconds: float | None = None,
    ) -> None:
        with self.lock:
            if self.capture_session is not None:
                raise RuntimeError("Stop the active raw capture before changing settings.")
            cols = max(1, int(cols))
            rows = max(1, int(rows))
            capture_seconds = max(0.0, float(capture_seconds))
            if empty_room_delay_seconds is None:
                empty_room_delay_seconds = self.empty_room_delay_seconds
            empty_room_delay_seconds = max(0.0, float(empty_room_delay_seconds))
            self.system_config.grid.cols = cols
            self.system_config.grid.rows = rows
            self.system_config.fingerprinting.capture_seconds = capture_seconds
            self.system_config.fingerprinting.baseline_start_delay_seconds = (
                empty_room_delay_seconds
            )
            save_system_config(self.config_path, self.system_config)
            duration_text = "manual stop" if capture_seconds <= 0 else f"{capture_seconds:.1f}s"
            self.status_message = (
                f"Raw capture config saved: grid {cols}x{rows}, duration {duration_text}, "
                f"empty-room delay {empty_room_delay_seconds:.1f}s."
            )
            self._log_event_locked(
                "CONFIG",
                f"Updated raw capture config grid={cols}x{rows} duration={duration_text} "
                f"empty_room_delay={empty_room_delay_seconds:.1f}s",
            )

    def start_empty_room_capture(self) -> None:
        self._start_capture(kind="empty_room", grid_x=None, grid_y=None)

    def start_cell_capture(self, grid_x: int, grid_y: int) -> None:
        self._start_capture(kind="cell", grid_x=grid_x, grid_y=grid_y)

    def start_ground_truth_capture(
        self,
        gt_location: int | str,
        *,
        duration_seconds: float | None = None,
        start_delay_seconds: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        gt_location = int(gt_location)
        if gt_location < 0 or gt_location > 6:
            raise RuntimeError("Ground-truth location must be an integer from 0 to 6.")
        capture_metadata = {"gt_location": gt_location}
        if metadata:
            capture_metadata.update(metadata)
        self._start_capture(
            kind="ground_truth",
            grid_x=None,
            grid_y=None,
            label=f"GT {gt_location}",
            file_label=f"gt_{gt_location}",
            duration_seconds=duration_seconds,
            start_delay_seconds=start_delay_seconds,
            metadata=capture_metadata,
        )

    def stop_capture(self) -> None:
        with self.lock:
            if self.capture_session is None:
                return
            self._finalize_capture_locked(self.capture_session, time.time(), "stopped")
            self.capture_session = None

    def clear_raw_data(self) -> int:
        with self.lock:
            if self.capture_session is not None:
                raise RuntimeError("Stop the active raw capture before clearing raw data.")
            deleted_count = 0
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            for path in self.raw_data_dir.iterdir():
                if not path.is_file():
                    continue
                path.unlink()
                deleted_count += 1
            self.saved_packet_count = 0
            self.comm_logs.clear()
            self.status_message = f"Cleared raw data files: {deleted_count} deleted."
            self._log_event_locked("RAW", f"Cleared raw data files count={deleted_count}")
            return deleted_count

    def process_packet(self, payload: bytes, source: str) -> bool:
        now = time.time()
        frame = parse_adr018_frame(payload)
        feature = (
            build_feature_frame(
                frame,
                source,
                now,
                self.system_config.fingerprinting.feature_bin_count,
            )
            if frame is not None
            else None
        )
        return self.process_receiver_event(
            {
                "type": "packet",
                "valid": frame is not None,
                "source": source,
                "received_at": now,
                "payload": payload,
                "frame": frame,
                "feature": feature,
            }
        )

    def process_receiver_event(self, event: dict[str, object]) -> bool:
        now = float(event.get("received_at", time.time()))
        source = str(event.get("source", "unknown"))
        payload = event.get("payload")
        frame = event.get("frame")
        feature = event.get("feature")
        valid = bool(event.get("valid"))
        with self.lock:
            self.packet_count += 1
            self.last_packet_ts = now
            session = self.capture_session
            if session is not None and isinstance(payload, bytes):
                self._write_packet_locked(session, payload, source, now, frame)

            if not valid:
                if now - self.last_invalid_packet_log_ts >= 5.0:
                    self.last_invalid_packet_log_ts = now
                    self._log_event_locked(
                        "WARN",
                        f"Ignored non-ADR018 or malformed packet from {source}",
                    )
                return False

            if not isinstance(feature, FeatureFrame):
                if isinstance(frame, ParsedFrame):
                    feature = build_feature_frame(
                        frame,
                        source,
                        now,
                        self.system_config.fingerprinting.feature_bin_count,
                    )
                else:
                    return False
            node_state = self.live_nodes.get(feature.node_id)
            if node_state is None:
                node_state = RawNodeState(
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
        return frame is not None

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            now = time.time()
            self._advance_capture_locked(now)
            capture_payload: dict[str, object] = {
                "active": False,
                "kind": None,
                "label": "",
                "grid_x": None,
                "grid_y": None,
                "started_at": None,
                "started_at_iso": "",
                "remaining_seconds": None,
                "elapsed_seconds": 0.0,
                "path": "",
                "valid_packets": 0,
                "invalid_packets": 0,
                "saved_packets": 0,
                "packets_by_node": [],
            }
            if self.capture_session is not None:
                session = self.capture_session
                waiting_seconds = max(0.0, session.started_at - now)
                remaining = (
                    None
                    if session.ends_at is None
                    else max(0.0, session.ends_at - max(now, session.started_at))
                )
                capture_payload = {
                    "active": True,
                    "kind": session.kind,
                    "label": session.label,
                    "pending": waiting_seconds > 0.0,
                    "delay_remaining_seconds": waiting_seconds,
                    "grid_x": session.grid_x,
                    "grid_y": session.grid_y,
                    "started_at": session.started_at,
                    "started_at_iso": self._format_iso(session.started_at),
                    "remaining_seconds": remaining,
                    "elapsed_seconds": max(0.0, now - session.started_at),
                    "path": str(session.path),
                    "valid_packets": session.valid_packets,
                    "invalid_packets": session.invalid_packets,
                    "saved_packets": session.valid_packets + session.invalid_packets,
                    "packets_by_node": [
                        {"node_id": node_id, "count": count}
                        for node_id, count in sorted(session.packets_by_node.items())
                    ],
                }

            cells = []
            for grid_y in range(self.grid_rows):
                for grid_x in range(self.grid_cols):
                    cell_key = self.cell_key(grid_x, grid_y)
                    is_capturing = (
                        self.capture_session is not None
                        and self.capture_session.kind == "cell"
                        and self.capture_session.grid_x == grid_x
                        and self.capture_session.grid_y == grid_y
                    )
                    cells.append(
                        {
                            "cell_key": cell_key,
                            "grid_x": grid_x,
                            "grid_y": grid_y,
                            "is_capturing": is_capturing,
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
                        "rssi_dbm": node.rssi_dbm,
                        "snr_db": node.snr_db,
                        "subcarrier_count": node.subcarrier_count,
                    }
                )

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
                    "empty_room_delay_seconds": self.empty_room_delay_seconds,
                },
                "metrics": {
                    "packet_count": self.packet_count,
                    "saved_packet_count": self.saved_packet_count,
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
                "cells": cells,
                "nodes": nodes,
                "udp_status": self.udp_status,
                "status_message": self.status_message,
                "comm_logs": list(self.comm_logs),
                "comm_log_path": str(self.comm_log_path),
                "raw_data_dir": str(self.raw_data_dir),
                "session_index_path": str(self.session_index_path),
            }

    def _start_capture(
        self,
        *,
        kind: str,
        grid_x: int | None,
        grid_y: int | None,
        label: str | None = None,
        file_label: str | None = None,
        duration_seconds: float | None = None,
        start_delay_seconds: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        with self.lock:
            now = time.time()
            self._advance_capture_locked(now)
            if self.capture_session is not None:
                raise RuntimeError("A raw capture is already running. Stop it first.")
            if kind == "cell":
                if grid_x is None or grid_y is None:
                    raise RuntimeError("Cell raw capture requires grid coordinates.")
                label = label or f"Cell ({grid_x + 1}, {grid_y + 1})"
                file_label = file_label or f"cell_x{grid_x + 1}_y{grid_y + 1}"
            else:
                label = label or ("Empty Room" if kind == "empty_room" else kind)
                file_label = file_label or ("empty_room" if kind == "empty_room" else kind)
            session_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            filename = f"{session_id}_{file_label}.jsonl"
            path = self.raw_data_dir / filename
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            handle = path.open("w", encoding="utf-8", newline="\n")
            duration = (
                self.capture_seconds
                if duration_seconds is None
                else max(0.0, float(duration_seconds))
            )
            delay = (
                self.empty_room_delay_seconds
                if start_delay_seconds is None and kind == "empty_room"
                else max(0.0, float(start_delay_seconds or 0.0))
            )
            starts_at = now + delay
            ends_at = None if duration <= 0 else starts_at + duration
            session = RawCaptureSession(
                session_id=session_id,
                kind=kind,
                label=label,
                grid_x=grid_x,
                grid_y=grid_y,
                armed_at=now,
                started_at=starts_at,
                ends_at=ends_at,
                path=path,
                handle=handle,
                metadata=dict(metadata or {}),
            )
            self.capture_session = session
            self._write_jsonl_locked(
                handle,
                self._session_header_payload(
                    session,
                    now,
                    delay,
                    record_type="session_armed" if delay > 0.0 else "session_start",
                ),
            )
            session.start_written = delay <= 0.0
            duration_text = "until Stop" if ends_at is None else f"for {duration:.1f}s"
            if delay > 0.0:
                self.status_message = (
                    f"Empty Room raw capture will start in {delay:.1f}s, then record "
                    f"{duration_text}. Saving to {path}."
                )
                self._log_event_locked(
                    "RAW",
                    f"Armed {label} raw capture delay={delay:.1f}s -> {path}",
                )
            else:
                self.status_message = (
                    f"Started raw capture for {label} {duration_text}. Saving to {path}."
                )
                self._log_event_locked("RAW", f"Started {label} raw capture -> {path}")

    def _write_packet_locked(
        self,
        session: RawCaptureSession,
        payload: bytes,
        source: str,
        captured_at: float,
        frame: object,
    ) -> None:
        if captured_at < session.started_at:
            return
        if not session.start_written:
            self._write_session_start_locked(session)
        elapsed_ms = (captured_at - session.started_at) * 1000.0
        record: dict[str, object] = {
            "record_type": "packet",
            "schema_version": 1,
            "session_id": session.session_id,
            "kind": session.kind,
            "label": session.label,
            "grid_x": session.grid_x,
            "grid_y": session.grid_y,
            "started_at_unix": session.started_at,
            "started_at_iso": self._format_iso(session.started_at),
            "captured_at_unix": captured_at,
            "captured_at_iso": self._format_iso(captured_at),
            "elapsed_ms": elapsed_ms,
            "source": source,
            "raw_len": len(payload),
            "raw_b64": base64.b64encode(payload).decode("ascii"),
            "valid_adr018": frame is not None,
        }
        record.update(session.metadata)
        if frame is None:
            session.invalid_packets += 1
        else:
            feature = build_feature_frame(
                frame,
                source,
                captured_at,
                self.system_config.fingerprinting.feature_bin_count,
            )
            session.valid_packets += 1
            session.packets_by_node[frame.node_id] = (
                session.packets_by_node.get(frame.node_id, 0) + 1
            )
            record.update(
                {
                    "node_id": frame.node_id,
                    "n_antennas": frame.n_antennas,
                    "n_subcarriers": frame.n_subcarriers,
                    "freq_mhz": frame.freq_mhz,
                    "sequence": frame.sequence,
                    "rssi_dbm": frame.rssi_dbm,
                    "noise_floor_dbm": frame.noise_floor_dbm,
                    "snr_db": feature.snr_db,
                    "amplitudes": frame.amplitudes,
                    "phases": frame.phases,
                    "active_subcarrier_count": feature.n_subcarriers,
                    "feature_vector": feature.feature_vector,
                    "amplitude_mean": feature.amplitude_mean,
                    "amplitude_std": feature.amplitude_std,
                    "amplitude_rms": feature.amplitude_rms,
                    "amplitude_p90": feature.amplitude_p90,
                    "gradient_mean": feature.gradient_mean,
                    "phase_step_std": feature.phase_step_std,
                }
            )
        session.sources.add(source)
        self.saved_packet_count += 1
        self._write_jsonl_locked(session.handle, record)

    def _advance_capture_locked(self, now: float) -> None:
        if self.capture_session is None:
            return
        if now >= self.capture_session.started_at and not self.capture_session.start_written:
            self._write_session_start_locked(self.capture_session)
        if self.capture_session.ends_at is None or now < self.capture_session.ends_at:
            return
        self._finalize_capture_locked(self.capture_session, now, "completed")
        self.capture_session = None

    def _write_session_start_locked(self, session: RawCaptureSession) -> None:
        delay = max(0.0, session.started_at - session.armed_at)
        self._write_jsonl_locked(
            session.handle,
            self._session_header_payload(
                session,
                session.armed_at,
                delay,
                record_type="session_start",
            ),
        )
        session.start_written = True
        self.status_message = f"Started raw capture for {session.label}. Saving to {session.path}."
        self._log_event_locked("RAW", f"Started {session.label} raw capture -> {session.path}")

    def _finalize_capture_locked(
        self,
        session: RawCaptureSession,
        ended_at: float,
        reason: str,
    ) -> None:
        packet_total = session.valid_packets + session.invalid_packets
        summary = {
            "record_type": "session_end",
            "schema_version": 1,
            "session_id": session.session_id,
            "kind": session.kind,
            "label": session.label,
            "grid_x": session.grid_x,
            "grid_y": session.grid_y,
            "armed_at_unix": session.armed_at,
            "armed_at_iso": self._format_iso(session.armed_at),
            "started_at_unix": session.started_at,
            "started_at_iso": self._format_iso(session.started_at),
            "ended_at_unix": ended_at,
            "ended_at_iso": self._format_iso(ended_at),
            "duration_seconds": max(0.0, ended_at - session.started_at),
            "start_delay_seconds": max(0.0, session.started_at - session.armed_at),
            "end_reason": reason,
            "path": str(session.path),
            "valid_packets": session.valid_packets,
            "invalid_packets": session.invalid_packets,
            "total_packets": packet_total,
            "packets_by_node": {
                str(node_id): count
                for node_id, count in sorted(session.packets_by_node.items())
            },
            "sources": sorted(session.sources),
        }
        summary.update(session.metadata)
        self._write_jsonl_locked(session.handle, summary)
        session.handle.close()
        self._append_session_index_locked(summary)
        self.status_message = (
            f"Raw capture {reason} for {session.label}: {packet_total} packets "
            f"({session.valid_packets} valid) saved to {session.path}."
        )
        self._log_event_locked(
            "RAW",
            f"Finished {session.label} raw capture reason={reason} packets={packet_total} "
            f"valid={session.valid_packets} path={session.path}",
        )

    def _append_session_index_locked(self, summary: dict[str, object]) -> None:
        self.session_index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.session_index_path.open("a", encoding="utf-8", newline="\n") as handle:
            self._write_jsonl_locked(handle, summary)

    def _session_header_payload(
        self,
        session: RawCaptureSession,
        written_at: float,
        delay: float,
        *,
        record_type: str,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "record_type": record_type,
            "schema_version": 1,
            "session_id": session.session_id,
            "kind": session.kind,
            "label": session.label,
            "grid_x": session.grid_x,
            "grid_y": session.grid_y,
            "armed_at_unix": session.armed_at,
            "armed_at_iso": self._format_iso(session.armed_at),
            "started_at_unix": session.started_at,
            "started_at_iso": self._format_iso(session.started_at),
            "start_delay_seconds": delay,
            "duration_seconds": None
            if session.ends_at is None
            else max(0.0, session.ends_at - session.started_at),
            "listen_host": self.system_config.host.listen_host,
            "udp_port": self.system_config.host.udp_port,
            "enabled_node_ids": [
                node.node_id for node in self.system_config.enabled_nodes()
            ],
        }
        if written_at != session.armed_at:
            payload["written_at_unix"] = written_at
            payload["written_at_iso"] = self._format_iso(written_at)
        payload.update(session.metadata)
        return payload

    @staticmethod
    def cell_key(grid_x: int, grid_y: int) -> str:
        return f"{grid_x},{grid_y}"

    def _node_label(self, node_id: int) -> str:
        for node in self.system_config.nodes:
            if node.node_id == node_id:
                return node.label
        return f"ESP {node_id}"

    def _log_event_locked(self, level: str, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{timestamp}] [{level}] {message}"
        self.comm_logs.append(line)
        self.comm_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.comm_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    @staticmethod
    def _write_jsonl_locked(handle: TextIO, payload: dict[str, object]) -> None:
        handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
        handle.flush()

    @staticmethod
    def _format_iso(timestamp: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp)) + (
            f".{int((timestamp % 1.0) * 1000):03d}"
        )
