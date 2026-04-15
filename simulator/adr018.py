from __future__ import annotations

from dataclasses import dataclass
import struct

ADR018_MAGIC = 0xC5110001
ADR018_HEADER_SIZE = 20
ADR018_FRAME_FMT = "<IBBHIIbbH"
DEFAULT_N_ANTENNAS = 1
DEFAULT_N_SUBCARRIERS = 56
MAX_ANTENNAS = 4
MAX_SUBCARRIERS = 256


def channel_to_frequency_mhz(channel: int) -> int:
    if 1 <= channel <= 13:
        return 2412 + (channel - 1) * 5
    if channel == 14:
        return 2484
    if 36 <= channel <= 177:
        return 5000 + channel * 5
    return 0


def clamp_int8(value: float | int) -> int:
    return max(-128, min(127, int(round(value))))


@dataclass(frozen=True)
class Adr018Frame:
    node_id: int
    n_antennas: int
    n_subcarriers: int
    freq_mhz: int
    sequence: int
    rssi_dbm: int
    noise_floor_dbm: int
    iq_bytes: bytes

    def to_bytes(self) -> bytes:
        expected = self.n_antennas * self.n_subcarriers * 2
        if len(self.iq_bytes) != expected:
            raise ValueError(
                f"invalid I/Q payload size: expected {expected}, got {len(self.iq_bytes)}"
            )

        header = struct.pack(
            ADR018_FRAME_FMT,
            ADR018_MAGIC,
            self.node_id & 0xFF,
            self.n_antennas & 0xFF,
            self.n_subcarriers & 0xFFFF,
            self.freq_mhz & 0xFFFFFFFF,
            self.sequence & 0xFFFFFFFF,
            clamp_int8(self.rssi_dbm),
            clamp_int8(self.noise_floor_dbm),
            0,
        )
        return header + self.iq_bytes
