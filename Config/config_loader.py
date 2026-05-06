from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HostConfig:
    listen_host: str = "0.0.0.0"
    target_ip: str = "127.0.0.1"
    udp_port: int = 5005
    stimulus_enabled: bool = True
    stimulus_broadcast_ip: str = ""
    stimulus_port: int = 40000
    stimulus_interval_ms: int = 20


@dataclass
class GridConfig:
    cols: int = 3
    rows: int = 3


@dataclass
class FingerprintingConfig:
    capture_seconds: float = 4.0
    window_seconds: float = 0.9
    window_step_seconds: float = 0.1
    effective_packets_per_second: float = 10.0
    window_sample_count: int = 9
    window_step_samples: int = 1
    minimum_samples_per_node: int = 6
    feature_bin_count: int = 12
    baseline_capture_seconds: float = 5.0
    baseline_start_delay_seconds: float = 8.0
    baseline_required_for_training: bool = True
    smoothing_half_window: int = 0
    capture_auto_extend_seconds: float = 6.0
    capture_extend_step_seconds: float = 2.0
    minimum_observed_windows: int = 8
    minimum_observed_window_ratio: float = 0.2
    live_probability_smoothing_seconds: float = 0.0
    best_cell_switch_margin: float = 0.0
    best_cell_switch_delay_seconds: float = 0.0
    prediction_stale_grace_seconds: float = 1.25


@dataclass
class StartCellConfig:
    x: int = 0
    y: int = 0


@dataclass
class SimulationConfig:
    tick_hz: float = 20.0
    frame_burst_size: int = 1
    movement_interval_seconds: float = 5.0
    path_mode: str = "snake"
    amplitude_noise: float = 0.18
    phase_jitter: float = 0.05
    start_cell: StartCellConfig = field(default_factory=StartCellConfig)


@dataclass
class NodeConfig:
    node_id: int
    label: str
    enabled: bool = True
    com_port: str = ""
    target_port: int | None = None
    csi_send_interval_ms: int = 20
    wifi_ssid: str = ""
    wifi_password: str = ""
    wifi_channel: int = 6
    source_kind: str = "firmware_or_simulator"


@dataclass
class SystemConfig:
    host: HostConfig = field(default_factory=HostConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    fingerprinting: FingerprintingConfig = field(default_factory=FingerprintingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    nodes: list[NodeConfig] = field(default_factory=list)

    def enabled_nodes(self) -> list[NodeConfig]:
        return [node for node in self.nodes if node.enabled]

    def node_target_port(self, node: NodeConfig) -> int:
        return int(node.target_port if node.target_port is not None else self.host.udp_port)


def default_config_path(base_dir: str | Path | None = None) -> Path:
    if base_dir is None:
        return Path(__file__).resolve().parent / "system_config.json"
    return Path(base_dir).resolve() / "Config" / "system_config.json"


def load_system_config(path: str | Path | None = None) -> SystemConfig:
    config_path = default_config_path() if path is None else Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    host_raw = raw.get("host", {})
    host = HostConfig(
        listen_host=str(host_raw.get("listen_host", "0.0.0.0")),
        target_ip=str(host_raw.get("target_ip", "127.0.0.1")),
        udp_port=int(host_raw.get("udp_port", 5005)),
        stimulus_enabled=bool(host_raw.get("stimulus_enabled", True)),
        stimulus_broadcast_ip=str(host_raw.get("stimulus_broadcast_ip", "")),
        stimulus_port=max(1, min(65535, int(host_raw.get("stimulus_port", 40000)))),
        stimulus_interval_ms=max(
            5,
            min(1000, int(host_raw.get("stimulus_interval_ms", 20))),
        ),
    )
    grid = GridConfig(
        cols=max(1, int(raw.get("grid", {}).get("cols", 3))),
        rows=max(1, int(raw.get("grid", {}).get("rows", 3))),
    )
    fingerprinting_raw = raw.get("fingerprinting", {})
    capture_seconds = max(
        1.0,
        float(fingerprinting_raw.get("capture_seconds", 4.0)),
    )
    effective_packets_per_second = max(
        1.0,
        float(fingerprinting_raw.get("effective_packets_per_second", 10.0)),
    )
    window_seconds = max(
        0.25,
        min(
            float(fingerprinting_raw.get("window_seconds", 0.5)),
            capture_seconds,
        ),
    )
    window_step_seconds = max(
        0.05,
        min(
            float(
                fingerprinting_raw.get(
                    "window_step_seconds",
                    0.1,
                )
            ),
            window_seconds,
        ),
    )
    window_sample_count = max(
        1,
        int(
            fingerprinting_raw.get(
                "window_sample_count",
                max(1, int(round(window_seconds * effective_packets_per_second))),
            )
        ),
    )
    window_step_samples = max(
        1,
        min(
            int(
                fingerprinting_raw.get(
                    "window_step_samples",
                    max(1, int(round(window_step_seconds * effective_packets_per_second))),
                )
            ),
            window_sample_count,
        ),
    )
    window_seconds = min(
        capture_seconds,
        float(window_sample_count) / effective_packets_per_second,
    )
    window_step_seconds = min(
        window_seconds,
        float(window_step_samples) / effective_packets_per_second,
    )
    fingerprinting = FingerprintingConfig(
        capture_seconds=capture_seconds,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        effective_packets_per_second=effective_packets_per_second,
        window_sample_count=window_sample_count,
        window_step_samples=window_step_samples,
        minimum_samples_per_node=max(
            1,
            int(fingerprinting_raw.get("minimum_samples_per_node", 6)),
        ),
        feature_bin_count=max(
            4,
            int(fingerprinting_raw.get("feature_bin_count", 12)),
        ),
        baseline_capture_seconds=max(
            1.0,
            float(fingerprinting_raw.get("baseline_capture_seconds", 5.0)),
        ),
        baseline_start_delay_seconds=max(
            0.0,
            float(fingerprinting_raw.get("baseline_start_delay_seconds", 8.0)),
        ),
        baseline_required_for_training=bool(
            fingerprinting_raw.get("baseline_required_for_training", True)
        ),
        smoothing_half_window=max(
            0,
            int(fingerprinting_raw.get("smoothing_half_window", 20)),
        ),
        capture_auto_extend_seconds=max(
            0.0,
            float(fingerprinting_raw.get("capture_auto_extend_seconds", 6.0)),
        ),
        capture_extend_step_seconds=max(
            0.25,
            float(fingerprinting_raw.get("capture_extend_step_seconds", 2.0)),
        ),
        minimum_observed_windows=max(
            1,
            int(fingerprinting_raw.get("minimum_observed_windows", 8)),
        ),
        minimum_observed_window_ratio=max(
            0.0,
            min(
                1.0,
                float(fingerprinting_raw.get("minimum_observed_window_ratio", 0.2)),
            ),
        ),
        live_probability_smoothing_seconds=max(
            0.0,
            float(
                fingerprinting_raw.get(
                    "live_probability_smoothing_seconds",
                    0.75,
                )
            ),
        ),
        best_cell_switch_margin=max(
            0.0,
            float(fingerprinting_raw.get("best_cell_switch_margin", 0.05)),
        ),
        best_cell_switch_delay_seconds=max(
            0.0,
            float(fingerprinting_raw.get("best_cell_switch_delay_seconds", 0.5)),
        ),
        prediction_stale_grace_seconds=max(
            0.0,
            float(fingerprinting_raw.get("prediction_stale_grace_seconds", 1.25)),
        ),
    )
    simulation_raw = raw.get("simulation", {})
    simulation = SimulationConfig(
        tick_hz=max(1.0, float(simulation_raw.get("tick_hz", 20.0))),
        frame_burst_size=max(1, int(simulation_raw.get("frame_burst_size", 1))),
        movement_interval_seconds=max(
            0.5,
            float(simulation_raw.get("movement_interval_seconds", 5.0)),
        ),
        path_mode=str(simulation_raw.get("path_mode", "snake")),
        amplitude_noise=max(0.0, float(simulation_raw.get("amplitude_noise", 0.18))),
        phase_jitter=max(0.0, float(simulation_raw.get("phase_jitter", 0.05))),
        start_cell=StartCellConfig(**simulation_raw.get("start_cell", {})),
    )
    nodes = [
        NodeConfig(
            node_id=int(item["node_id"]),
            label=str(item.get("label", f"ESP {item['node_id']}")),
            enabled=bool(item.get("enabled", True)),
            com_port=str(item.get("com_port", "")),
            target_port=(
                max(1, min(65535, int(item["target_port"])))
                if item.get("target_port") is not None
                else None
            ),
            csi_send_interval_ms=max(
                0,
                min(1000, int(item.get("csi_send_interval_ms", 20))),
            ),
            wifi_ssid=str(item.get("wifi_ssid", "")),
            wifi_password=str(item.get("wifi_password", "")),
            wifi_channel=max(1, min(13, int(item.get("wifi_channel", 6)))),
            source_kind=str(item.get("source_kind", "firmware_or_simulator")),
        )
        for item in raw.get("nodes", [])
    ]
    return SystemConfig(
        host=host,
        grid=grid,
        fingerprinting=fingerprinting,
        simulation=simulation,
        nodes=nodes,
    )


def dump_system_config(config: SystemConfig) -> dict[str, Any]:
    return {
        "host": {
            "listen_host": config.host.listen_host,
            "target_ip": config.host.target_ip,
            "udp_port": config.host.udp_port,
            "stimulus_enabled": config.host.stimulus_enabled,
            "stimulus_broadcast_ip": config.host.stimulus_broadcast_ip,
            "stimulus_port": config.host.stimulus_port,
            "stimulus_interval_ms": config.host.stimulus_interval_ms,
        },
        "grid": {
            "cols": config.grid.cols,
            "rows": config.grid.rows,
        },
        "fingerprinting": {
            "capture_seconds": config.fingerprinting.capture_seconds,
            "window_seconds": config.fingerprinting.window_seconds,
            "window_step_seconds": config.fingerprinting.window_step_seconds,
            "effective_packets_per_second": (
                config.fingerprinting.effective_packets_per_second
            ),
            "window_sample_count": config.fingerprinting.window_sample_count,
            "window_step_samples": config.fingerprinting.window_step_samples,
            "minimum_samples_per_node": config.fingerprinting.minimum_samples_per_node,
            "feature_bin_count": config.fingerprinting.feature_bin_count,
            "baseline_capture_seconds": config.fingerprinting.baseline_capture_seconds,
            "baseline_start_delay_seconds": (
                config.fingerprinting.baseline_start_delay_seconds
            ),
            "baseline_required_for_training": (
                config.fingerprinting.baseline_required_for_training
            ),
            "smoothing_half_window": config.fingerprinting.smoothing_half_window,
            "capture_auto_extend_seconds": (
                config.fingerprinting.capture_auto_extend_seconds
            ),
            "capture_extend_step_seconds": (
                config.fingerprinting.capture_extend_step_seconds
            ),
            "minimum_observed_windows": (
                config.fingerprinting.minimum_observed_windows
            ),
            "minimum_observed_window_ratio": (
                config.fingerprinting.minimum_observed_window_ratio
            ),
            "live_probability_smoothing_seconds": (
                config.fingerprinting.live_probability_smoothing_seconds
            ),
            "best_cell_switch_margin": config.fingerprinting.best_cell_switch_margin,
            "best_cell_switch_delay_seconds": (
                config.fingerprinting.best_cell_switch_delay_seconds
            ),
            "prediction_stale_grace_seconds": (
                config.fingerprinting.prediction_stale_grace_seconds
            ),
        },
        "simulation": {
            "tick_hz": config.simulation.tick_hz,
            "frame_burst_size": config.simulation.frame_burst_size,
            "movement_interval_seconds": config.simulation.movement_interval_seconds,
            "path_mode": config.simulation.path_mode,
            "amplitude_noise": config.simulation.amplitude_noise,
            "phase_jitter": config.simulation.phase_jitter,
            "start_cell": {
                "x": config.simulation.start_cell.x,
                "y": config.simulation.start_cell.y,
            },
        },
        "nodes": [
            {
                "node_id": node.node_id,
                "label": node.label,
                "enabled": node.enabled,
                "com_port": node.com_port,
                "target_port": node.target_port,
                "csi_send_interval_ms": node.csi_send_interval_ms,
                "wifi_ssid": node.wifi_ssid,
                "wifi_password": node.wifi_password,
                "wifi_channel": node.wifi_channel,
                "source_kind": node.source_kind,
            }
            for node in config.nodes
        ],
    }


def save_system_config(path: str | Path | None, config: SystemConfig) -> None:
    config_path = default_config_path() if path is None else Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(dump_system_config(config), indent=2),
        encoding="utf-8",
    )
