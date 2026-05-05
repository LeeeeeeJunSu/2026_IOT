from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Config.config_loader import NodeConfig, load_system_config, save_system_config


DEFAULT_CONFIG_PATH = REPO_ROOT / "Config" / "system_config.json"
DEFAULT_PROVISION_DIR = (
    REPO_ROOT / "firmware" / "esp32-csi-fingerprint-node" / "build" / "provisioning"
)
NVS_PARTITION_OFFSET = "0x9000"
NVS_PARTITION_SIZE = "0x6000"


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_system_config(config_path)

    node_id = args.node_id or prompt_int("ESP node id")
    com_port = args.com_port or prompt_text("COM port")
    host_ip = args.host_ip or prompt_text(
        "Host target IP",
        default=config.host.target_ip,
    )

    node = find_or_create_node(config.nodes, node_id)
    target_port = args.target_port
    if target_port is None:
        target_port = node.target_port if node.target_port is not None else 5000 + node_id
    target_port = clamp_port(target_port)

    wifi_ssid = args.wifi_ssid if args.wifi_ssid is not None else node.wifi_ssid
    wifi_password = (
        args.wifi_password if args.wifi_password is not None else node.wifi_password
    )
    if not wifi_ssid:
        wifi_ssid = prompt_text("Wi-Fi SSID")
    wifi_channel = args.wifi_channel if args.wifi_channel is not None else node.wifi_channel
    wifi_channel = max(1, min(13, int(wifi_channel)))
    send_interval_ms = (
        args.send_interval_ms
        if args.send_interval_ms is not None
        else node.csi_send_interval_ms
    )
    send_interval_ms = max(0, min(1000, int(send_interval_ms)))

    config.host.target_ip = host_ip
    node.node_id = node_id
    node.label = node.label or f"ESP {node_id}"
    node.enabled = True
    node.com_port = com_port
    node.target_port = target_port
    node.wifi_ssid = wifi_ssid
    node.wifi_password = wifi_password
    node.wifi_channel = wifi_channel
    node.csi_send_interval_ms = send_interval_ms
    save_system_config(config_path, config)

    provision_dir = Path(args.provision_dir).resolve()
    provision_dir.mkdir(parents=True, exist_ok=True)
    csv_path = provision_dir / f"csi_cfg_node{node_id}.csv"
    bin_path = provision_dir / f"csi_cfg_node{node_id}.bin"
    write_nvs_csv(
        csv_path,
        wifi_ssid=wifi_ssid,
        wifi_password=wifi_password,
        target_ip=host_ip,
        target_port=target_port,
        node_id=node_id,
        wifi_channel=wifi_channel,
        send_interval_ms=send_interval_ms,
    )

    generator = find_nvs_partition_generator()
    run(
        [
            sys.executable,
            str(generator),
            "generate",
            "--version",
            "2",
            str(csv_path),
            str(bin_path),
            NVS_PARTITION_SIZE,
        ]
    )

    if not args.no_flash:
        run(
            [
                sys.executable,
                "-m",
                "esptool",
                "--chip",
                "esp32s3",
                "--port",
                com_port,
                "--baud",
                str(args.baud),
                "write-flash",
                "--flash-mode",
                "dio",
                "--flash-freq",
                "80m",
                "--flash-size",
                "4MB",
                NVS_PARTITION_OFFSET,
                str(bin_path),
            ]
        )

    print(
        "Provisioned config "
        f"node={node_id} com={com_port} target={host_ip}:{target_port} "
        f"wifi_channel={wifi_channel} send_interval_ms={send_interval_ms} "
        f"csv={csv_path} bin={bin_path}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update System Config and provision one ESP32 CSI node from it.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--host-ip", help="Host PC IP that ESP nodes send UDP packets to.")
    parser.add_argument("--node-id", type=int, help="ESP node id to provision.")
    parser.add_argument("--com-port", help="Serial COM port, for example COM5.")
    parser.add_argument(
        "--target-port",
        type=int,
        help="Per-node UDP target port. Defaults to existing node target_port or 5000 + node id.",
    )
    parser.add_argument("--wifi-ssid")
    parser.add_argument("--wifi-password")
    parser.add_argument("--wifi-channel", type=int)
    parser.add_argument(
        "--send-interval-ms",
        type=int,
        help="Minimum milliseconds between UDP CSI frames. Default: node config, normally 20.",
    )
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--provision-dir", default=str(DEFAULT_PROVISION_DIR))
    parser.add_argument("--no-flash", action="store_true")
    return parser.parse_args()


def find_or_create_node(nodes: list[NodeConfig], node_id: int) -> NodeConfig:
    for node in nodes:
        if node.node_id == node_id:
            return node
    node = NodeConfig(node_id=node_id, label=f"ESP {node_id}", target_port=5000 + node_id)
    nodes.append(node)
    nodes.sort(key=lambda item: item.node_id)
    return node


def write_nvs_csv(
    path: Path,
    *,
    wifi_ssid: str,
    wifi_password: str,
    target_ip: str,
    target_port: int,
    node_id: int,
    wifi_channel: int,
    send_interval_ms: int,
) -> None:
    lines = [
        "key,type,encoding,value",
        "csi_cfg,namespace,,",
        f"ssid,data,string,{wifi_ssid}",
        f"password,data,string,{wifi_password}",
        f"target_ip,data,string,{target_ip}",
        f"target_port,data,u16,{target_port}",
        f"send_int_ms,data,u16,{send_interval_ms}",
        f"node_id,data,u8,{node_id}",
        f"wifi_channel,data,u8,{wifi_channel}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def find_nvs_partition_generator() -> Path:
    spec = importlib.util.find_spec("esp_idf_nvs_partition_gen.nvs_partition_gen")
    if spec is not None and spec.origin:
        return Path(spec.origin)

    idf_path = Path(str(Path.home()))
    env_idf = Path(os.environ.get("IDF_PATH", ""))
    candidates = [
        env_idf / "components" / "nvs_flash" / "nvs_partition_generator" / "nvs_partition_gen.py",
        idf_path / "esp" / "esp-idf" / "components" / "nvs_flash" / "nvs_partition_generator" / "nvs_partition_gen.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Could not find nvs_partition_gen.py. Install esp-idf-nvs-partition-gen "
        "or run inside an ESP-IDF environment."
    )


def prompt_text(label: str, *, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    if value:
        return value
    if default:
        return default
    raise ValueError(f"{label} is required.")


def prompt_int(label: str) -> int:
    return int(prompt_text(label))


def clamp_port(port: int) -> int:
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"Invalid UDP port: {port}")
    return int(port)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
