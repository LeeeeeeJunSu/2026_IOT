from __future__ import annotations

import socket
import threading

from .core import FingerprintEngine


class UdpReceiverThread(threading.Thread):
    def __init__(self, engine: FingerprintEngine, host: str, port: int) -> None:
        super().__init__(name="fingerprint-udp-receiver", daemon=True)
        self.engine = engine
        self.host = host
        self.port = port
        self.stop_event = threading.Event()
        self.sock: socket.socket | None = None

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock = sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        sock.settimeout(1.0)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            self.engine.set_udp_status(f"UDP bind failed: {exc}")
            return

        self.engine.set_udp_status(f"Listening on UDP {self.host}:{self.port}")
        while not self.stop_event.is_set():
            try:
                payload, source = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            self.engine.process_packet(payload, f"{source[0]}:{source[1]}")

    def stop(self) -> None:
        self.stop_event.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass

