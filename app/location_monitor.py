from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from .core import EMPTY_ROOM_CLASS_KEY


@dataclass(frozen=True)
class LocationMonitorConfig:
    min_confidence: float = 0.45
    stable_seconds: float = 1.25
    empty_room_stable_seconds: float = 3.0
    stale_packet_ms: float = 1500.0
    min_active_node_ratio: float = 0.7


class StableLocationMonitor:
    def __init__(self, config: LocationMonitorConfig | None = None) -> None:
        self.config = config or LocationMonitorConfig()
        self.stable_label_key: str | None = None
        self.stable_label_display = ""
        self.stable_probability = 0.0
        self.stable_since: float | None = None
        self.candidate_label_key: str | None = None
        self.candidate_since: float | None = None
        self.last_update_ts: float | None = None

    def update(
        self,
        snapshot: dict[str, Any],
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        self.last_update_ts = now

        prediction = _as_dict(snapshot.get("prediction"))
        training = _as_dict(snapshot.get("training"))
        metrics = _as_dict(snapshot.get("metrics"))
        baseline = _as_dict(snapshot.get("baseline"))

        raw_label_key = _optional_str(prediction.get("best_label_key"))
        raw_label_display = _optional_str(prediction.get("best_label_display")) or ""
        raw_probability = _float(prediction.get("best_probability"))
        model_ready = bool(prediction.get("model_ready"))
        prediction_ready = bool(prediction.get("ready"))
        active_nodes = int(_float(metrics.get("active_nodes")))
        required_nodes = int(
            _float(training.get("required_nodes"))
            or _float(baseline.get("required_nodes"))
            or active_nodes
        )
        last_packet_age_ms = metrics.get("last_packet_age_ms")
        packet_fresh = (
            isinstance(last_packet_age_ms, (int, float))
            and float(last_packet_age_ms) <= self.config.stale_packet_ms
        )
        if last_packet_age_ms is None:
            packet_fresh = False

        min_active_nodes = (
            1
            if required_nodes <= 0
            else max(1, math.ceil(required_nodes * self.config.min_active_node_ratio))
        )
        enough_nodes = active_nodes >= min_active_nodes
        confidence_ok = raw_probability >= self.config.min_confidence
        raw_usable = (
            model_ready
            and prediction_ready
            and bool(raw_label_key)
            and confidence_ok
            and packet_fresh
            and enough_nodes
        )

        if raw_usable:
            self._observe_candidate(
                raw_label_key=raw_label_key,
                raw_label_display=raw_label_display,
                raw_probability=raw_probability,
                now=now,
            )
        else:
            self.candidate_label_key = None
            self.candidate_since = None

        status = self._status(
            model_ready=model_ready,
            prediction_ready=prediction_ready,
            packet_fresh=packet_fresh,
            enough_nodes=enough_nodes,
            confidence_ok=confidence_ok,
        )
        stable_age = 0.0
        if self.stable_since is not None:
            stable_age = max(0.0, now - self.stable_since)

        return {
            "status": status,
            "updated_at_unix": now,
            "raw": {
                "label_key": raw_label_key,
                "label_display": raw_label_display,
                "probability": raw_probability,
                "usable": raw_usable,
            },
            "stable": {
                "label_key": self.stable_label_key,
                "label_display": self.stable_label_display,
                "probability": self.stable_probability,
                "is_empty_room": self.stable_label_key == EMPTY_ROOM_CLASS_KEY,
                "age_seconds": stable_age,
            },
            "candidate": {
                "label_key": self.candidate_label_key,
                "age_seconds": 0.0
                if self.candidate_since is None
                else max(0.0, now - self.candidate_since),
            },
            "health": {
                "model_ready": model_ready,
                "prediction_ready": prediction_ready,
                "packet_fresh": packet_fresh,
                "active_nodes": active_nodes,
                "required_nodes": required_nodes,
                "min_active_nodes": min_active_nodes,
                "last_packet_age_ms": last_packet_age_ms,
                "confidence_ok": confidence_ok,
            },
            "grid": snapshot.get("grid", {}),
            "cells": snapshot.get("cells", []),
            "empty_room": snapshot.get("empty_room", {}),
            "nodes": snapshot.get("nodes", []),
            "metrics": metrics,
            "udp_status": snapshot.get("udp_status", ""),
            "status_message": snapshot.get("status_message", ""),
        }

    def _observe_candidate(
        self,
        *,
        raw_label_key: str | None,
        raw_label_display: str,
        raw_probability: float,
        now: float,
    ) -> None:
        if raw_label_key != self.candidate_label_key:
            self.candidate_label_key = raw_label_key
            self.candidate_since = now

        hold_seconds = (
            self.config.empty_room_stable_seconds
            if raw_label_key == EMPTY_ROOM_CLASS_KEY
            else self.config.stable_seconds
        )
        candidate_age = (
            0.0
            if self.candidate_since is None
            else max(0.0, now - self.candidate_since)
        )
        if candidate_age < hold_seconds:
            return

        if self.stable_label_key != raw_label_key:
            self.stable_since = now
        self.stable_label_key = raw_label_key
        self.stable_label_display = raw_label_display
        self.stable_probability = raw_probability

    def _status(
        self,
        *,
        model_ready: bool,
        prediction_ready: bool,
        packet_fresh: bool,
        enough_nodes: bool,
        confidence_ok: bool,
    ) -> str:
        if not model_ready:
            return "no_model"
        if not packet_fresh:
            return "no_signal"
        if not enough_nodes:
            return "sensor_degraded"
        if not prediction_ready:
            return "warming_up"
        if not confidence_ok:
            return "low_confidence"
        if self.stable_label_key == EMPTY_ROOM_CLASS_KEY:
            return "empty_room"
        if self.stable_label_key:
            return "occupied"
        return "stabilizing"


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0
