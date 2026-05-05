from __future__ import annotations

import argparse
import select
import socket
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.protocol import parse_adr018_frame


def main() -> int:
    args = parse_args()
    ports = list(range(args.start_port, args.end_port + 1))
    sockets = open_sockets(ports, args.host, args.buffer_bytes)
    try:
        result = collect(sockets, args.seconds)
    finally:
        for sock in sockets:
            sock.close()
    print_report(result, ports, args.seconds)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ADR-018 UDP traffic per port.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--start-port", type=int, default=5001)
    parser.add_argument("--end-port", type=int, default=5009)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--buffer-bytes", type=int, default=4 * 1024 * 1024)
    return parser.parse_args()


def open_sockets(ports: list[int], host: str, buffer_bytes: int) -> list[socket.socket]:
    sockets: list[socket.socket] = []
    try:
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
            sock.bind((host, port))
            sock.setblocking(False)
            sockets.append(sock)
    except OSError:
        for sock in sockets:
            sock.close()
        raise
    return sockets


def collect(sockets: list[socket.socket], seconds: float) -> dict[str, object]:
    counts: dict[int, dict[int, int]] = {
        sock.getsockname()[1]: {} for sock in sockets
    }
    sources: dict[int, dict[int, str]] = {
        sock.getsockname()[1]: {} for sock in sockets
    }
    first_seq: dict[tuple[int, int], int] = {}
    last_seq: dict[tuple[int, int], int] = {}
    invalid: dict[int, int] = {sock.getsockname()[1]: 0 for sock in sockets}

    end = time.time() + seconds
    while time.time() < end:
        readable, _, _ = select.select(sockets, [], [], 0.25)
        for sock in readable:
            port = sock.getsockname()[1]
            while True:
                try:
                    payload, source = sock.recvfrom(4096)
                except BlockingIOError:
                    break
                frame = parse_adr018_frame(payload)
                if frame is None:
                    invalid[port] += 1
                    continue
                counts[port][frame.node_id] = counts[port].get(frame.node_id, 0) + 1
                sources[port][frame.node_id] = f"{source[0]}:{source[1]}"
                key = (port, frame.node_id)
                first_seq.setdefault(key, frame.sequence)
                last_seq[key] = frame.sequence
    return {
        "counts": counts,
        "sources": sources,
        "first_seq": first_seq,
        "last_seq": last_seq,
        "invalid": invalid,
    }


def print_report(result: dict[str, object], ports: list[int], seconds: float) -> None:
    counts = result["counts"]
    sources = result["sources"]
    first_seq = result["first_seq"]
    last_seq = result["last_seq"]
    invalid = result["invalid"]
    assert isinstance(counts, dict)
    assert isinstance(sources, dict)
    assert isinstance(first_seq, dict)
    assert isinstance(last_seq, dict)
    assert isinstance(invalid, dict)

    print(f"PORT CHECK {seconds:.1f}s")
    for port in ports:
        node_counts = counts.get(port, {})
        total = sum(node_counts.values())
        if not node_counts:
            print(f"{port}: total=0 no packets invalid={invalid.get(port, 0)}")
            continue
        details = []
        for node_id, count in sorted(node_counts.items()):
            key = (port, node_id)
            pps = count / seconds if seconds > 0 else 0.0
            details.append(
                f"node{node_id}:count={count},pps={pps:.1f},"
                f"src={sources[port][node_id]},seq={first_seq[key]}->{last_seq[key]}"
            )
        print(
            f"{port}: total={total} "
            + "; ".join(details)
            + f" invalid={invalid.get(port, 0)}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
