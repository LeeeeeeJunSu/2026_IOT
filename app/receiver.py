from __future__ import annotations

import multiprocessing as mp
import queue
import socket
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from Config.config_loader import SystemConfig

from .protocol import build_feature_frame, parse_adr018_frame


class PacketEngine(Protocol):
    def set_udp_status(self, message: str) -> None:
        ...

    def process_packet(self, payload: bytes, source: str) -> bool:
        ...

    def process_receiver_event(self, event: dict[str, object]) -> bool:
        ...


class UdpReceiverThread(threading.Thread):
    def __init__(
        self,
        engine: PacketEngine,
        host: str,
        port: int,
        *,
        label: str | None = None,
        expected_node_ids: Iterable[int] = (),
    ) -> None:
        receiver_label = label or f"UDP {port}"
        super().__init__(name=f"fingerprint-udp-receiver-{port}", daemon=True)
        self.engine = engine
        self.host = host
        self.port = port
        self.label = receiver_label
        self.expected_node_ids = tuple(sorted(set(int(node_id) for node_id in expected_node_ids)))
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

        self.engine.set_udp_status(f"Listening on {self.label} UDP {self.host}:{self.port}")
        while not self.stop_event.is_set():
            try:
                payload, source = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            self.engine.process_packet(payload, f"{source[0]}:{source[1]} -> {self.port}")

    def stop(self) -> None:
        self.stop_event.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass


@dataclass(frozen=True)
class ReceiverSpec:
    host: str
    port: int
    label: str
    expected_node_ids: tuple[int, ...]
    forward_interval_ms: int


class UdpReceiverGroup:
    def __init__(self, receivers: list[object]) -> None:
        self.receivers = receivers

    def start(self) -> None:
        for receiver in self.receivers:
            receiver.start()

    def stop(self) -> None:
        for receiver in self.receivers:
            receiver.stop()

    def join(self, timeout: float | None = None) -> None:
        for receiver in self.receivers:
            receiver.join(timeout=timeout)


class MultiprocessUdpReceiverGroup:
    def __init__(
        self,
        engine: PacketEngine,
        specs: list[ReceiverSpec],
        *,
        feature_bin_count: int,
        include_payload: bool = False,
    ) -> None:
        self.engine = engine
        self.specs = specs
        self.feature_bin_count = feature_bin_count
        self.include_payload = include_payload
        self.ctx = mp.get_context("spawn")
        self.event_queue: mp.Queue = self.ctx.Queue(maxsize=50000)
        self.stop_event = self.ctx.Event()
        self.processes: list[mp.Process] = []
        self.dispatcher = threading.Thread(
            target=self._dispatch_events,
            name="udp-receiver-dispatcher",
            daemon=True,
        )

    def start(self) -> None:
        self.stop_event.clear()
        for spec in self.specs:
            process = self.ctx.Process(
                target=_udp_receiver_process_main,
                name=f"udp-receiver-process-{spec.port}",
                args=(
                    spec,
                    self.feature_bin_count,
                    self.include_payload,
                    self.event_queue,
                    self.stop_event,
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
        self.dispatcher.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        for process in self.processes:
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
        self.dispatcher.join(timeout=timeout)

    def _dispatch_events(self) -> None:
        while not self.stop_event.is_set() or any(
            process.is_alive() for process in self.processes
        ):
            try:
                event = self.event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "status":
                self.engine.set_udp_status(str(event.get("message", "")))
            elif event.get("type") == "packet":
                self.engine.process_receiver_event(event)


def build_receivers_for_config(
    engine: PacketEngine,
    config: SystemConfig,
    *,
    multiprocessing: bool = True,
    include_payload: bool = False,
) -> UdpReceiverGroup | MultiprocessUdpReceiverGroup:
    ports_by_node: dict[int, list[int]] = {}
    for node in config.enabled_nodes():
        port = config.node_target_port(node)
        ports_by_node.setdefault(int(port), []).append(node.node_id)

    if not ports_by_node:
        ports_by_node[int(config.host.udp_port)] = []

    specs = [
        ReceiverSpec(
            host=config.host.listen_host,
            port=port,
            label=_receiver_label(port, node_ids),
            expected_node_ids=tuple(sorted(node_ids)),
            forward_interval_ms=_receiver_forward_interval_ms(config, node_ids),
        )
        for port, node_ids in sorted(ports_by_node.items())
    ]

    if multiprocessing:
        return MultiprocessUdpReceiverGroup(
            engine,
            specs,
            feature_bin_count=config.fingerprinting.feature_bin_count,
            include_payload=include_payload,
        )

    receivers = [
        UdpReceiverThread(
            engine,
            spec.host,
            spec.port,
            label=spec.label,
            expected_node_ids=spec.expected_node_ids,
        )
        for spec in specs
    ]
    return UdpReceiverGroup(receivers)


def _receiver_label(port: int, node_ids: list[int]) -> str:
    if not node_ids:
        return f"fallback:{port}"
    if len(node_ids) == 1:
        return f"node {node_ids[0]}"
    joined = ",".join(str(node_id) for node_id in sorted(node_ids))
    return f"nodes {joined}"


def _udp_receiver_process_main(
    spec: ReceiverSpec,
    feature_bin_count: int,
    include_payload: bool,
    event_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.settimeout(0.5)
    try:
        sock.bind((spec.host, spec.port))
    except OSError as exc:
        _put_event(
            event_queue,
            {
                "type": "status",
                "message": f"UDP bind failed on {spec.label} {spec.host}:{spec.port}: {exc}",
            },
        )
        return

    _put_event(
        event_queue,
        {
            "type": "status",
            "message": f"Listening on {spec.label} UDP {spec.host}:{spec.port} in process",
        },
    )
    dropped_events = 0
    last_forward_by_node: dict[int, float] = {}
    while not stop_event.is_set():
        try:
            payload, source = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break

        received_at = time.time()
        frame = parse_adr018_frame(payload)
        source_text = f"{source[0]}:{source[1]} -> {spec.port}"
        if frame is None:
            event = {
                "type": "packet",
                "valid": False,
                "source": source_text,
                "received_at": received_at,
                "port": spec.port,
                "label": spec.label,
                "payload": payload if include_payload else None,
            }
        else:
            if _should_skip_forward(spec, frame.node_id, received_at, last_forward_by_node):
                continue
            feature = build_feature_frame(frame, source_text, received_at, feature_bin_count)
            event = {
                "type": "packet",
                "valid": True,
                "source": source_text,
                "received_at": received_at,
                "port": spec.port,
                "label": spec.label,
                "frame": frame,
                "feature": feature,
                "payload": payload if include_payload else None,
            }
        if not _put_event(event_queue, event):
            dropped_events += 1
            if dropped_events == 1 or dropped_events % 1000 == 0:
                _put_event(
                    event_queue,
                    {
                        "type": "status",
                        "message": (
                            f"Receiver queue full on {spec.label} UDP {spec.port}; "
                            f"dropped {dropped_events} parsed events"
                        ),
                    },
                )
    sock.close()


def _put_event(event_queue: mp.Queue, event: dict[str, object]) -> bool:
    try:
        event_queue.put_nowait(event)
    except queue.Full:
        return False
    return True


def _receiver_forward_interval_ms(config: SystemConfig, node_ids: list[int]) -> int:
    if not node_ids:
        return 0
    intervals = [
        node.csi_send_interval_ms
        for node in config.enabled_nodes()
        if node.node_id in node_ids
    ]
    if not intervals:
        return 0
    return max(0, min(intervals))


def _should_skip_forward(
    spec: ReceiverSpec,
    node_id: int,
    received_at: float,
    last_forward_by_node: dict[int, float],
) -> bool:
    if spec.forward_interval_ms <= 0:
        return False
    last_forward_at = last_forward_by_node.get(node_id)
    if (
        last_forward_at is not None
        and received_at - last_forward_at < spec.forward_interval_ms / 1000.0
    ):
        return True
    last_forward_by_node[node_id] = received_at
    return False
