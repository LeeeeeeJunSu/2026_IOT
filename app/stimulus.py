from __future__ import annotations

import socket
import threading

from Config.config_loader import SystemConfig


class UdpStimulusBroadcaster(threading.Thread):
    def __init__(
        self,
        broadcast_ip: str,
        port: int,
        interval_ms: int,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="wifi-csi-stimulus-broadcaster", daemon=True)
        self.broadcast_ip = broadcast_ip
        self.port = port
        self.interval_seconds = max(0.005, float(interval_ms) / 1000.0)
        self.enabled = enabled
        self.stop_event = threading.Event()
        self.sock: socket.socket | None = None

    def run(self) -> None:
        if not self.enabled:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock = sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        payload = b"csi-stimulus"
        while not self.stop_event.is_set():
            try:
                sock.sendto(payload, (self.broadcast_ip, self.port))
            except OSError:
                self.stop_event.wait(1.0)
                continue
            self.stop_event.wait(self.interval_seconds)

    def stop(self) -> None:
        self.stop_event.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass


def build_stimulus_for_config(config: SystemConfig) -> UdpStimulusBroadcaster:
    broadcast_ip = config.host.stimulus_broadcast_ip.strip()
    if not broadcast_ip:
        broadcast_ip = _default_broadcast_ip(config.host.target_ip)
    return UdpStimulusBroadcaster(
        broadcast_ip=broadcast_ip,
        port=config.host.stimulus_port,
        interval_ms=config.host.stimulus_interval_ms,
        enabled=config.host.stimulus_enabled,
    )


def _default_broadcast_ip(target_ip: str) -> str:
    parts = target_ip.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        return ".".join([parts[0], parts[1], parts[2], "255"])
    return "255.255.255.255"
