from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from statistics import pstdev
from typing import Sequence


ADR018_MAGIC = 0xC5110001
HEADER_SIZE = 20
PILOT_SUBCARRIER_INDEXES = {-21, -7, 7, 21}
ACTIVE_SUBCARRIER_INDEXES = tuple(
    index
    for index in range(-25, 26)
    if index != 0 and index not in PILOT_SUBCARRIER_INDEXES
)
ACTIVE_SUBCARRIER_COUNT = len(ACTIVE_SUBCARRIER_INDEXES)


@dataclass
class ParsedFrame:
    node_id: int
    n_antennas: int
    n_subcarriers: int
    freq_mhz: int
    sequence: int
    rssi_dbm: float
    noise_floor_dbm: float
    amplitudes: list[float]
    phases: list[float]


@dataclass
class FeatureFrame:
    node_id: int
    source: str
    captured_at: float
    sequence: int
    n_subcarriers: int
    rssi_dbm: float
    noise_floor_dbm: float
    snr_db: float
    amplitude_mean: float
    amplitude_std: float
    amplitude_rms: float
    amplitude_p90: float
    gradient_mean: float
    phase_step_std: float
    feature_vector: list[float]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) >= 2 else 0.0


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    ratio = max(0.0, min(1.0, ratio))
    position = ratio * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def _phase_delta(left: float, right: float) -> float:
    delta = left - right
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _pool_vector(values: Sequence[float], bins: int) -> list[float]:
    if bins <= 0:
        return []
    if not values:
        return [0.0 for _ in range(bins)]
    result: list[float] = []
    size = len(values)
    for index in range(bins):
        start = int(index * size / bins)
        end = int((index + 1) * size / bins)
        if end <= start:
            end = min(size, start + 1)
        bucket = values[start:end]
        result.append(_mean(bucket) if bucket else 0.0)
    return result


def _fft_shift(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    middle = len(values) // 2
    return list(values[middle:]) + list(values[:middle])


def _select_active_subcarriers(shifted_values: Sequence[float]) -> list[float]:
    if not shifted_values:
        return []
    size = len(shifted_values)
    if size == 64:
        center = size // 2
        return [
            float(shifted_values[center + subcarrier_index])
            for subcarrier_index in ACTIVE_SUBCARRIER_INDEXES
        ]
    without_dc = [
        float(value)
        for index, value in enumerate(shifted_values)
        if index != size // 2
    ]
    return _pool_vector(without_dc, ACTIVE_SUBCARRIER_COUNT)


def parse_adr018_frame(payload: bytes) -> ParsedFrame | None:
    if len(payload) < HEADER_SIZE:
        return None
    magic = struct.unpack_from("<I", payload, 0)[0]
    if magic != ADR018_MAGIC:
        return None

    node_id = payload[4]
    n_antennas = payload[5]
    n_subcarriers = struct.unpack_from("<H", payload, 6)[0]
    freq_mhz = struct.unpack_from("<I", payload, 8)[0]
    sequence = struct.unpack_from("<I", payload, 12)[0]
    rssi_dbm = float(struct.unpack_from("<b", payload, 16)[0])
    noise_floor_dbm = float(struct.unpack_from("<b", payload, 17)[0])

    pair_count = n_antennas * n_subcarriers
    expected = HEADER_SIZE + pair_count * 2
    if pair_count <= 0 or len(payload) < expected:
        return None

    amplitudes: list[float] = []
    phases: list[float] = []
    for index in range(pair_count):
        i_value = struct.unpack_from("<b", payload, HEADER_SIZE + index * 2)[0]
        q_value = struct.unpack_from("<b", payload, HEADER_SIZE + index * 2 + 1)[0]
        amplitudes.append(math.sqrt(i_value * i_value + q_value * q_value))
        phases.append(math.atan2(q_value, i_value))

    return ParsedFrame(
        node_id=node_id,
        n_antennas=n_antennas,
        n_subcarriers=n_subcarriers,
        freq_mhz=freq_mhz,
        sequence=sequence,
        rssi_dbm=rssi_dbm,
        noise_floor_dbm=noise_floor_dbm,
        amplitudes=amplitudes,
        phases=phases,
    )


def build_feature_frame(
    frame: ParsedFrame,
    source: str,
    captured_at: float,
    _feature_bin_count: int,
) -> FeatureFrame:
    shifted_amplitudes = _fft_shift(frame.amplitudes)
    shifted_phases = _fft_shift(frame.phases)
    amplitudes = _select_active_subcarriers(shifted_amplitudes)
    phases = _select_active_subcarriers(shifted_phases)
    if not amplitudes:
        amplitudes = [float(value) for value in frame.amplitudes]
    if not phases:
        phases = [float(value) for value in frame.phases]
    amplitude_mean = _mean(amplitudes)
    amplitude_std = _std(amplitudes)
    amplitude_rms = _rms(amplitudes)
    amplitude_p90 = _percentile(amplitudes, 0.90)
    gradients = [abs(amplitudes[index + 1] - amplitudes[index]) for index in range(len(amplitudes) - 1)]
    phase_steps = [_phase_delta(phases[index + 1], phases[index]) for index in range(len(phases) - 1)]
    gradient_mean = _mean(gradients)
    phase_step_std = _std(phase_steps)
    snr_db = frame.rssi_dbm - frame.noise_floor_dbm

    feature_vector = [float(value) for value in amplitudes]
    return FeatureFrame(
        node_id=frame.node_id,
        source=source,
        captured_at=captured_at,
        sequence=frame.sequence,
        n_subcarriers=len(feature_vector),
        rssi_dbm=frame.rssi_dbm,
        noise_floor_dbm=frame.noise_floor_dbm,
        snr_db=snr_db,
        amplitude_mean=amplitude_mean,
        amplitude_std=amplitude_std,
        amplitude_rms=amplitude_rms,
        amplitude_p90=amplitude_p90,
        gradient_mean=gradient_mean,
        phase_step_std=phase_step_std,
        feature_vector=feature_vector,
    )

