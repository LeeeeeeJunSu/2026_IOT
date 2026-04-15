from __future__ import annotations

from dataclasses import dataclass
import socket
import time
from typing import Callable

from Config.config_loader import SystemConfig

from .fingerprint import RoomFingerprintLibrary
from .pathing import Cell, build_path, clamp_cell, rotate_path_to_start


@dataclass(frozen=True)
class SimulationTick:
    elapsed_seconds: float
    current_cell: Cell
    path_index: int
    path_length: int
    phase_label: str = ""


@dataclass(frozen=True)
class ScenarioStep:
    cell: Cell
    duration_seconds: float
    label: str = ""


class Esp32TrafficSimulator:
    def __init__(
        self,
        config: SystemConfig,
        seed: int | None = None,
        dry_run: bool = False,
        scenario_steps: list[ScenarioStep] | None = None,
        logger: Callable[[str], None] = print,
    ) -> None:
        self._config = config
        self._logger = logger
        self._dry_run = dry_run
        self._library = RoomFingerprintLibrary(config=config, seed=seed)
        self._nodes = sorted(config.enabled_nodes(), key=lambda node: node.node_id)
        if not self._nodes:
            raise ValueError("simulation requires at least one enabled node")

        raw_start = clamp_cell(
            config.simulation.start_cell.x,
            config.simulation.start_cell.y,
            config.grid.cols,
            config.grid.rows,
        )
        path = build_path(config.grid.cols, config.grid.rows, config.simulation.path_mode)
        self._path = rotate_path_to_start(path, raw_start)
        if raw_start not in self._path:
            self._path = [raw_start] + [cell for cell in self._path if cell != raw_start]

        self._scenario_steps = [
            ScenarioStep(
                cell=clamp_cell(step.cell.x, step.cell.y, config.grid.cols, config.grid.rows),
                duration_seconds=max(0.01, float(step.duration_seconds)),
                label=step.label,
            )
            for step in (scenario_steps or [])
        ]
        self._scenario_total_duration = sum(
            step.duration_seconds for step in self._scenario_steps
        )
        self._sequence_by_node = {node.node_id: 0 for node in self._nodes}
        self._destination = (config.host.target_ip, int(config.host.udp_port))
        self._socket = None if dry_run else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._log_node_summary()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def run(self, duration_seconds: float | None = None) -> None:
        tick_hz = max(1.0, float(self._config.simulation.tick_hz))
        tick_interval = 1.0 / tick_hz
        movement_interval = max(0.01, float(self._config.simulation.movement_interval_seconds))
        frame_burst_size = max(1, int(self._config.simulation.frame_burst_size))

        start = time.monotonic()
        next_tick = start
        tick_index = 0
        last_path_index = -1

        self._logger(
            f"simulator: target={self._destination[0]}:{self._destination[1]} "
            f"grid={self._config.grid.cols}x{self._config.grid.rows} "
            f"path_mode={self._config.simulation.path_mode} "
            f"nodes={len(self._nodes)} dry_run={self._dry_run} "
            f"scenario_steps={len(self._scenario_steps)}"
        )

        try:
            while True:
                now = time.monotonic()
                elapsed = now - start
                if duration_seconds is not None and elapsed >= duration_seconds:
                    break

                if now < next_tick:
                    time.sleep(min(0.05, next_tick - now))
                    continue

                tick = self._current_tick(elapsed, movement_interval)
                if tick.path_index != last_path_index:
                    if tick.phase_label:
                        self._logger(
                            f"simulator: phase={tick.phase_label} cell=({tick.current_cell.x}, "
                            f"{tick.current_cell.y}) at t={tick.elapsed_seconds:.2f}s"
                        )
                    else:
                        self._logger(
                            f"simulator: occupant moved to cell ({tick.current_cell.x}, "
                            f"{tick.current_cell.y}) at t={tick.elapsed_seconds:.2f}s"
                        )
                    last_path_index = tick.path_index

                for burst_index in range(frame_burst_size):
                    for node in self._nodes:
                        sequence = self._sequence_by_node[node.node_id]
                        frame = self._library.build_frame(
                            node_id=node.node_id,
                            cell=tick.current_cell,
                            sequence=sequence,
                            frame_index=tick_index,
                            burst_index=burst_index,
                        )
                        payload = frame.to_bytes()
                        if self._socket is not None:
                            self._socket.sendto(payload, self._destination)
                        else:
                            self._logger(
                                "simulator: frame "
                                f"node={node.node_id} seq={sequence} "
                                f"cell=({tick.current_cell.x},{tick.current_cell.y}) "
                                f"bytes={len(payload)} rssi={frame.rssi_dbm} "
                                f"noise={frame.noise_floor_dbm}"
                            )
                        self._sequence_by_node[node.node_id] = sequence + 1

                tick_index += 1
                next_tick += tick_interval

                if now - next_tick > tick_interval * 4:
                    next_tick = now + tick_interval
        except KeyboardInterrupt:
            self._logger("simulator: interrupted by user")
        finally:
            self.close()

    def _current_tick(self, elapsed_seconds: float, movement_interval: float) -> SimulationTick:
        if self._scenario_steps:
            return self._current_scenario_tick(elapsed_seconds)
        path_length = len(self._path)
        if path_length == 0:
            raise RuntimeError("simulation path is empty")
        path_index = int(elapsed_seconds // movement_interval) % path_length
        return SimulationTick(
            elapsed_seconds=elapsed_seconds,
            current_cell=self._path[path_index],
            path_index=path_index,
            path_length=path_length,
        )

    def _current_scenario_tick(self, elapsed_seconds: float) -> SimulationTick:
        if not self._scenario_steps or self._scenario_total_duration <= 0.0:
            raise RuntimeError("simulation scenario is empty")
        scenario_time = elapsed_seconds % self._scenario_total_duration
        elapsed_in_scenario = 0.0
        for index, step in enumerate(self._scenario_steps):
            elapsed_in_scenario += step.duration_seconds
            if scenario_time < elapsed_in_scenario:
                phase_label = (
                    step.label
                    if step.label
                    else f"step {index + 1}/{len(self._scenario_steps)}"
                )
                return SimulationTick(
                    elapsed_seconds=elapsed_seconds,
                    current_cell=step.cell,
                    path_index=index,
                    path_length=len(self._scenario_steps),
                    phase_label=phase_label,
                )
        final = self._scenario_steps[-1]
        return SimulationTick(
            elapsed_seconds=elapsed_seconds,
            current_cell=final.cell,
            path_index=len(self._scenario_steps) - 1,
            path_length=len(self._scenario_steps),
            phase_label=final.label or f"step {len(self._scenario_steps)}",
        )

    def _log_node_summary(self) -> None:
        for node in self._nodes:
            self._logger(f"simulator: {self._library.describe_node(node.node_id)}")


def parse_cell_sequence(spec: str, cols: int, rows: int) -> list[ScenarioStep]:
    steps: list[ScenarioStep] = []
    for raw_chunk in (spec or "").split(";"):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if "@" in chunk:
            chunk, label = chunk.split("@", 1)
            label = label.strip()
        else:
            label = ""
        try:
            cell_text, duration_text = chunk.split(":", 1)
            x_text, y_text = cell_text.split(",", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid cell-sequence segment {chunk!r}; expected x,y:seconds or x,y:seconds@label"
            ) from exc
        cell = clamp_cell(int(x_text), int(y_text), cols, rows)
        duration_seconds = max(0.01, float(duration_text))
        steps.append(
            ScenarioStep(
                cell=cell,
                duration_seconds=duration_seconds,
                label=label,
            )
        )
    return steps
