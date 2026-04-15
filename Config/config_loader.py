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
    keepalive_pings_per_second: float = 0.0


@dataclass
class GridConfig:
    cols: int = 3
    rows: int = 3


@dataclass
class FingerprintingConfig:
    capture_seconds: float = 4.0
    window_seconds: float = 0.5
    window_step_seconds: float = 0.05
    minimum_samples_per_node: int = 6
    feature_bin_count: int = 12


@dataclass
class StartCellConfig:
    x: int = 0
    y: int = 0


@dataclass
class SimulationConfig:
    tick_hz: float = 12.0
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
        keepalive_pings_per_second=max(
            0.0,
            float(host_raw.get("keepalive_pings_per_second", 0.0)),
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
                    fingerprinting_raw.get("window_seconds", 0.5),
                )
            ),
            window_seconds,
        ),
    )
    fingerprinting = FingerprintingConfig(
        capture_seconds=capture_seconds,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        minimum_samples_per_node=max(
            1,
            int(fingerprinting_raw.get("minimum_samples_per_node", 6)),
        ),
        feature_bin_count=max(
            4,
            int(fingerprinting_raw.get("feature_bin_count", 12)),
        ),
    )
    simulation_raw = raw.get("simulation", {})
    simulation = SimulationConfig(
        tick_hz=max(1.0, float(simulation_raw.get("tick_hz", 12.0))),
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
            "keepalive_pings_per_second": config.host.keepalive_pings_per_second,
        },
        "grid": {
            "cols": config.grid.cols,
            "rows": config.grid.rows,
        },
        "fingerprinting": {
            "capture_seconds": config.fingerprinting.capture_seconds,
            "window_seconds": config.fingerprinting.window_seconds,
            "window_step_seconds": config.fingerprinting.window_step_seconds,
            "minimum_samples_per_node": config.fingerprinting.minimum_samples_per_node,
            "feature_bin_count": config.fingerprinting.feature_bin_count,
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
