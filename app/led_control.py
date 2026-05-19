from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from Config.config_loader import LedZoneConfig, SystemConfig

from .core import EMPTY_ROOM_CLASS_KEY, FingerprintEngine
from .location_monitor import LocationMonitorConfig, StableLocationMonitor


AUTO_MODE = "auto"
ON_MODE = "on"
OFF_MODE = "off"
VALID_MODES = {AUTO_MODE, ON_MODE, OFF_MODE}


@dataclass
class _ZoneRuntime:
    config: LedZoneConfig
    output: "_LedOutput"
    mode: str = AUTO_MODE
    is_on: bool = False


class ZoneLedControllerThread(threading.Thread):
    def __init__(
        self,
        engine: FingerprintEngine,
        system_config: SystemConfig,
        *,
        update_seconds: float = 0.2,
        monitor_config: LocationMonitorConfig | None = None,
    ) -> None:
        super().__init__(name="zone-led-controller", daemon=True)
        self.engine = engine
        self.config = system_config.smart_home
        self.update_seconds = max(0.05, float(update_seconds))
        self.monitor = StableLocationMonitor(monitor_config)
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.backend = "disabled"
        self.message = "LED control is disabled in Config/system_config.json."
        self.hardware_ready = False
        self.active_cell_key: str | None = None
        self.zones: list[_ZoneRuntime] = []
        self._build_outputs()

    def set_mode(self, cell_key: str, mode: str) -> dict[str, Any]:
        mode = str(mode).lower().strip()
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported LED mode: {mode}")
        with self.lock:
            zone = self._find_zone_locked(cell_key)
            if zone is None:
                raise KeyError(f"Unknown LED zone: {cell_key}")
            zone.mode = mode
            self._apply_outputs_locked()
            return self.snapshot_locked()

    def set_all_mode(self, mode: str) -> dict[str, Any]:
        mode = str(mode).lower().strip()
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported LED mode: {mode}")
        with self.lock:
            for zone in self.zones:
                zone.mode = mode
            self._apply_outputs_locked()
            return self.snapshot_locked()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.snapshot_locked()

    def snapshot_locked(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "auto_enabled": bool(self.config.auto_enabled),
            "backend": self.backend,
            "hardware_ready": self.hardware_ready,
            "active_cell_key": self.active_cell_key,
            "message": self.message,
            "zones": [
                {
                    "cell_key": zone.config.cell_key,
                    "label": zone.config.label,
                    "pin_bcm": zone.config.pin_bcm,
                    "enabled": zone.config.enabled,
                    "active_high": zone.config.active_high,
                    "mode": zone.mode,
                    "is_on": zone.is_on,
                    "source": "auto" if zone.mode == AUTO_MODE else "manual",
                }
                for zone in self.zones
            ],
        }

    def run(self) -> None:
        while not self.stop_event.is_set():
            if self.config.enabled:
                snapshot = self.engine.snapshot()
                payload = self.monitor.update(snapshot)
                stable = payload.get("stable", {})
                label_key = stable.get("label_key") if isinstance(stable, dict) else None
                with self.lock:
                    self.active_cell_key = self._label_key_to_cell_key_locked(label_key)
                    self._apply_outputs_locked()
            self.stop_event.wait(self.update_seconds)

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            for zone in self.zones:
                zone.is_on = False
                zone.output.off()
                zone.output.close()

    def _build_outputs(self) -> None:
        if not self.config.enabled:
            return
        self.backend, self.message, factory = _load_led_factory()
        self.zones = [
            _ZoneRuntime(
                config=zone,
                output=factory(
                    pin_bcm=zone.pin_bcm,
                    active_high=zone.active_high,
                    enabled=zone.enabled,
                ),
            )
            for zone in self.config.zones
        ]
        self.hardware_ready = bool(self.zones) and all(
            zone.output.hardware_ready for zone in self.zones if zone.config.enabled
        )
        if self.hardware_ready:
            self.message = "GPIO LED control is ready."
        elif self.backend == "gpiozero":
            self.message = (
                "GPIO package loaded, but one or more LED pins could not be opened. "
                "Check Raspberry Pi permissions, pin wiring, and python3-lgpio."
            )

    def _apply_outputs_locked(self) -> None:
        for zone in self.zones:
            should_on = False
            if zone.config.enabled:
                if zone.mode == ON_MODE:
                    should_on = True
                elif zone.mode == AUTO_MODE and self.config.auto_enabled:
                    should_on = zone.config.cell_key == self.active_cell_key
            zone.is_on = should_on
            if should_on:
                zone.output.on()
            else:
                zone.output.off()

    def _find_zone_locked(self, cell_key: str) -> _ZoneRuntime | None:
        for zone in self.zones:
            if zone.config.cell_key == cell_key:
                return zone
        return None

    def _label_key_to_cell_key_locked(self, label_key: object) -> str | None:
        if label_key is None:
            return None
        text = str(label_key)
        if text == EMPTY_ROOM_CLASS_KEY or text == "0":
            return None
        if text.isdigit():
            index = int(text) - 1
            if 0 <= index < len(self.zones):
                return self.zones[index].config.cell_key
        return text


class _LedOutput:
    def __init__(
        self,
        *,
        pin_bcm: int,
        active_high: bool,
        enabled: bool,
        hardware: Any | None = None,
        hardware_ready: bool = False,
    ) -> None:
        self.pin_bcm = pin_bcm
        self.active_high = active_high
        self.enabled = enabled
        self.hardware = hardware
        self.hardware_ready = hardware_ready
        self.mock_state = False

    def on(self) -> None:
        self.mock_state = True
        if self.enabled and self.hardware is not None:
            self.hardware.on()

    def off(self) -> None:
        self.mock_state = False
        if self.enabled and self.hardware is not None:
            self.hardware.off()

    def close(self) -> None:
        if self.hardware is not None and hasattr(self.hardware, "close"):
            self.hardware.close()


def _load_led_factory() -> tuple[str, str, Any]:
    try:
        from gpiozero import LED
    except Exception as exc:
        message = (
            "GPIO packages are unavailable; running LED control in mock mode. "
            f"Install python3-gpiozero and python3-lgpio on Raspberry Pi 5. "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        return "mock", message, _mock_led_factory

    def factory(*, pin_bcm: int, active_high: bool, enabled: bool) -> _LedOutput:
        if not enabled:
            return _LedOutput(
                pin_bcm=pin_bcm,
                active_high=active_high,
                enabled=False,
                hardware_ready=False,
            )
        try:
            hardware = LED(
                pin_bcm,
                active_high=active_high,
                initial_value=False,
            )
        except Exception:
            return _LedOutput(
                pin_bcm=pin_bcm,
                active_high=active_high,
                enabled=enabled,
                hardware_ready=False,
            )
        return _LedOutput(
            pin_bcm=pin_bcm,
            active_high=active_high,
            enabled=enabled,
            hardware=hardware,
            hardware_ready=True,
        )

    return "gpiozero", "GPIO package loaded; initializing LED pins.", factory


def _mock_led_factory(*, pin_bcm: int, active_high: bool, enabled: bool) -> _LedOutput:
    return _LedOutput(
        pin_bcm=pin_bcm,
        active_high=active_high,
        enabled=enabled,
        hardware_ready=False,
    )
