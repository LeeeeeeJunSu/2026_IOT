from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import math
import random

from Config.config_loader import SystemConfig

from .adr018 import (
    DEFAULT_N_ANTENNAS,
    DEFAULT_N_SUBCARRIERS,
    Adr018Frame,
    channel_to_frequency_mhz,
    clamp_int8,
)
from .pathing import Cell


def _stable_int(*parts: object) -> int:
    hasher = blake2b(digest_size=16)
    for part in parts:
        encoded = str(part).encode("utf-8")
        hasher.update(encoded)
        hasher.update(b"|")
    return int.from_bytes(hasher.digest(), "little", signed=False)


def _unit_float(seed: int, salt: int = 0) -> float:
    mixed = _stable_int(seed, salt)
    return mixed / float(1 << 128)


@dataclass(frozen=True)
class NodeAnchor:
    anchor_x: float
    anchor_y: float
    anchor_angle: float
    anchor_radius: float


@dataclass(frozen=True)
class TapProfile:
    delay: float
    gain: float
    phase: float


@dataclass(frozen=True)
class CellSignature:
    taps: tuple[TapProfile, ...]
    direct_gain: float
    direct_phase: float
    attenuation: float
    phase_bias: float
    rssi_base: int
    noise_base: int


class RoomFingerprintLibrary:
    def __init__(self, config: SystemConfig, seed: int | None = None) -> None:
        self._config = config
        self._seed = _stable_int(
            "simulator",
            config.grid.cols,
            config.grid.rows,
            config.fingerprinting.feature_bin_count,
            seed if seed is not None else "default",
        )
        self._nodes = sorted(config.enabled_nodes(), key=lambda node: node.node_id)
        if not self._nodes:
            raise ValueError("simulation requires at least one enabled node")
        self._cell_cache: dict[tuple[int, int, int], CellSignature] = {}
        self._node_anchor_cache: dict[int, NodeAnchor] = {}

    @property
    def subcarrier_count(self) -> int:
        return DEFAULT_N_SUBCARRIERS

    @property
    def antenna_count(self) -> int:
        return DEFAULT_N_ANTENNAS

    def build_frame(
        self,
        node_id: int,
        cell: Cell,
        sequence: int,
        frame_index: int,
        burst_index: int,
    ) -> Adr018Frame:
        node = self._node_by_id(node_id)
        signature = self._cell_signature(node.node_id, cell)
        freq_mhz = channel_to_frequency_mhz(node.wifi_channel)
        iq_bytes, rssi_dbm, noise_floor_dbm = self._render_iq_payload(
            node.node_id,
            cell,
            signature,
            sequence=sequence,
            frame_index=frame_index,
            burst_index=burst_index,
        )
        return Adr018Frame(
            node_id=node.node_id,
            n_antennas=self.antenna_count,
            n_subcarriers=self.subcarrier_count,
            freq_mhz=freq_mhz,
            sequence=sequence,
            rssi_dbm=rssi_dbm,
            noise_floor_dbm=noise_floor_dbm,
            iq_bytes=iq_bytes,
        )

    def describe_node(self, node_id: int) -> str:
        node = self._node_by_id(node_id)
        anchor = self._node_anchor(node.node_id)
        return (
            f"node={node.node_id} label={node.label!r} "
            f"channel={node.wifi_channel} anchor=({anchor.anchor_x:.2f},"
            f"{anchor.anchor_y:.2f})"
        )

    def _node_by_id(self, node_id: int):
        for node in self._nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"unknown node_id: {node_id}")

    def _node_anchor(self, node_id: int) -> NodeAnchor:
        cached = self._node_anchor_cache.get(node_id)
        if cached is not None:
            return cached

        node = self._node_by_id(node_id)
        seed = _stable_int(self._seed, "node-anchor", node_id, node.label)
        angle = _unit_float(seed, 1) * math.tau
        radius = 0.25 + 0.12 * _unit_float(seed, 2)
        anchor = NodeAnchor(
            anchor_x=0.5 + math.cos(angle) * radius,
            anchor_y=0.5 + math.sin(angle) * radius,
            anchor_angle=angle,
            anchor_radius=radius,
        )
        self._node_anchor_cache[node_id] = anchor
        return anchor

    def _cell_signature(self, node_id: int, cell: Cell) -> CellSignature:
        key = (node_id, cell.x, cell.y)
        cached = self._cell_cache.get(key)
        if cached is not None:
            return cached

        anchor = self._node_anchor(node_id)
        cols = self._config.grid.cols
        rows = self._config.grid.rows
        cell_center_x = (cell.x + 0.5) / cols
        cell_center_y = (cell.y + 0.5) / rows
        dx = cell_center_x - anchor.anchor_x
        dy = cell_center_y - anchor.anchor_y
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        attenuation = max(0.0, 1.0 - distance * 0.9)

        seed = _stable_int(self._seed, "cell", node_id, cell.x, cell.y)
        rng = random.Random(seed)
        tap_count = max(4, int(self._config.fingerprinting.feature_bin_count))
        taps: list[TapProfile] = []
        for index in range(tap_count):
            tap_delay = 0.03 + 0.02 * index + rng.random() * 0.05 + distance * 0.11
            tap_gain = (0.55 + rng.random() * 0.75) * (0.9 + attenuation * 0.7)
            tap_phase = rng.random() * math.tau + bearing * (0.25 + index * 0.03)
            taps.append(
                TapProfile(
                    delay=tap_delay,
                    gain=tap_gain,
                    phase=tap_phase,
                )
            )

        signature = CellSignature(
            taps=tuple(taps),
            direct_gain=1.1 + attenuation * 1.4 + rng.random() * 0.4,
            direct_phase=bearing * 0.6 + rng.random() * math.tau,
            attenuation=attenuation,
            phase_bias=bearing * 0.5 + distance * 1.9,
            rssi_base=-41 - int(distance * 22) - node_id % 5,
            noise_base=-94 + int(distance * 10) + (node_id % 3),
        )
        self._cell_cache[key] = signature
        return signature

    def _render_iq_payload(
        self,
        node_id: int,
        cell: Cell,
        signature: CellSignature,
        sequence: int,
        frame_index: int,
        burst_index: int,
    ) -> tuple[bytes, int, int]:
        noise_seed = _stable_int(
            self._seed,
            "noise",
            node_id,
            cell.x,
            cell.y,
            sequence,
            frame_index,
            burst_index,
        )
        rng = random.Random(noise_seed)
        amplitude_noise = self._config.simulation.amplitude_noise
        phase_jitter = self._config.simulation.phase_jitter
        drift_phase = math.sin(
            frame_index * 0.11
            + burst_index * 0.37
            + node_id * 0.53
            + cell.x * 0.41
            + cell.y * 0.29
        )
        drift_gain = 1.0 + amplitude_noise * 0.35 * drift_phase
        drift_angle = phase_jitter * 1.35 * drift_phase
        burst_interference = rng.random() < 0.025
        if burst_interference:
            interference_gain = 1.0 + amplitude_noise * (1.4 + rng.random() * 1.2)
            interference_phase = rng.gauss(0.0, max(phase_jitter, 0.02) * 6.0)
            rssi_penalty = rng.randint(5, 12)
            noise_penalty = rng.randint(3, 8)
        else:
            interference_gain = 1.0
            interference_phase = 0.0
            rssi_penalty = 0
            noise_penalty = 0

        iq_values: list[int] = []
        base_scale = 18.0 + signature.attenuation * 8.0
        phase_offset = (
            signature.phase_bias
            + drift_angle
            + (rng.random() - 0.5) * phase_jitter * 3.0
            + interference_phase
        )
        carrier_center = (self.subcarrier_count - 1) / 2.0

        for sc_idx in range(self.subcarrier_count):
            sc_norm = (sc_idx - carrier_center) / max(1.0, carrier_center)
            ripple = 1.0 + 0.06 * math.sin(
                sc_idx * 0.35 + frame_index * 0.07 + node_id * 0.19
            )
            response = complex(
                signature.direct_gain * math.cos(
                    signature.direct_phase + sc_norm * math.tau * 0.5 + phase_offset
                ),
                signature.direct_gain * math.sin(
                    signature.direct_phase + sc_norm * math.tau * 0.5 + phase_offset
                ),
            )
            for tap_index, tap in enumerate(signature.taps):
                tap_phase = (
                    tap.phase
                    + sc_norm * math.tau * tap.delay
                    + phase_offset * (0.15 + tap_index * 0.01)
                )
                response += complex(
                    tap.gain * math.cos(tap_phase),
                    tap.gain * math.sin(tap_phase),
                )

            amp_scale = drift_gain * ripple
            amp_scale += rng.gauss(
                0.0,
                amplitude_noise * (0.12 + signature.attenuation * 0.08),
            )
            phase_noise = rng.gauss(0.0, phase_jitter) + drift_angle * 0.25
            response *= amp_scale * interference_gain
            response *= complex(math.cos(phase_noise), math.sin(phase_noise))

            magnitude = abs(response)
            phase = math.atan2(response.imag, response.real)
            amplitude_byte = max(4.0, min(108.0, magnitude * base_scale + 16.0))
            i_val = clamp_int8(amplitude_byte * math.cos(phase))
            q_val = clamp_int8(amplitude_byte * math.sin(phase))
            iq_values.append(i_val & 0xFF)
            iq_values.append(q_val & 0xFF)

        rssi = clamp_int8(
            signature.rssi_base
            + int(round(drift_phase * 2.5))
            - rng.randint(0, 4)
            - rssi_penalty
        )
        noise_floor = clamp_int8(
            signature.noise_base
            + int(round(-drift_phase * 1.5))
            + rng.randint(-2, 2)
            + noise_penalty
        )
        return bytes(iq_values), rssi, noise_floor
