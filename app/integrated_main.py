from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import FingerprintEngine
from app.dashboard import LocationDashboardThread
from app.led_control import ZoneLedControllerThread
from app.raw_replay_source import RawDataReplayThread, RawReplayConfig
from app.receiver import build_receivers_for_config
from app.runtime import EngineRuntimeThread
from app.stimulus import build_stimulus_for_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated Raspberry Pi CSI app: ESP32 UDP receiver, "
            "model runtime, web dashboard, optional Tkinter GUI, and raw-data fallback."
        )
    )
    parser.add_argument("--headless", action="store_true", help="Run without the Tkinter GUI.")
    parser.add_argument(
        "--dashboard-host",
        default="0.0.0.0",
        help="Dashboard bind host. Use 0.0.0.0 for phone access on the LAN.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8000,
        help="Dashboard port. If busy, the app tries the next few ports.",
    )
    parser.add_argument(
        "--no-raw-fallback",
        action="store_true",
        help="Disable replaying app/raw_data when no ESP32 signal is detected.",
    )
    parser.add_argument(
        "--fallback-after-seconds",
        type=float,
        default=15.0,
        help="Start raw-data fallback after this many seconds without live ESP32 packets.",
    )
    parser.add_argument(
        "--replay-speedup",
        type=float,
        default=20.0,
        help="Raw-data fallback replay speedup. 20 means a 60s capture replays in about 3s.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw_data",
        help="Raw JSONL capture directory used for fallback replay.",
    )
    parser.add_argument(
        "--no-leds",
        action="store_true",
        help="Disable Raspberry Pi GPIO LED zone control.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(__file__).resolve().parent
    engine = FingerprintEngine(workspace)
    receiver = build_receivers_for_config(engine, engine.system_config)
    stimulus = build_stimulus_for_config(engine.system_config)
    runtime = EngineRuntimeThread(engine)
    leds = (
        None
        if args.no_leds
        else ZoneLedControllerThread(engine, engine.system_config)
    )
    dashboard = LocationDashboardThread(
        engine,
        host=str(args.dashboard_host),
        port=int(args.dashboard_port),
        led_controller=leds,
    )
    replay = None
    if not args.no_raw_fallback:
        replay = RawDataReplayThread(
            engine,
            RawReplayConfig(
                raw_dir=args.raw_dir,
                fallback_after_seconds=max(1.0, float(args.fallback_after_seconds)),
                replay_speedup=max(1.0, float(args.replay_speedup)),
            ),
        )

    receiver.start()
    stimulus.start()
    runtime.start()
    if leds is not None:
        leds.start()
    dashboard.start()
    if replay is not None:
        replay.start()

    dashboard.ready_event.wait(timeout=1.5)
    _print_startup(engine, dashboard, replay, leds)

    gui_window: Any | None = None
    try:
        if not args.headless:
            gui_window = _try_build_gui(engine, receiver)
        if gui_window is not None:
            gui_window.run()
        else:
            _run_console_loop(engine, replay)
    finally:
        receiver.stop()
        stimulus.stop()
        runtime.stop()
        if leds is not None:
            leds.stop()
        dashboard.stop()
        if replay is not None:
            replay.stop()
        receiver.join(timeout=1.5)
        stimulus.join(timeout=1.5)
        runtime.join(timeout=1.5)
        if leds is not None:
            leds.join(timeout=1.5)
        dashboard.join(timeout=1.5)
        if replay is not None:
            replay.join(timeout=1.5)
    return 0


def _try_build_gui(engine: FingerprintEngine, receiver: Any) -> Any | None:
    try:
        from app.gui import FingerprintAppWindow
    except Exception as exc:
        print(
            "Tkinter GUI is unavailable; continuing with the web dashboard only. "
            f"Reason: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
    return FingerprintAppWindow(engine, receiver)


def _run_console_loop(
    engine: FingerprintEngine,
    replay: RawDataReplayThread | None,
) -> None:
    print("Headless mode is running. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            snapshot = engine.snapshot()
            prediction = snapshot.get("prediction", {})
            metrics = snapshot.get("metrics", {})
            if isinstance(prediction, dict) and isinstance(metrics, dict):
                label = prediction.get("best_label_display") or prediction.get("best_label_key") or "-"
                probability = float(prediction.get("best_probability") or 0.0)
                replay_text = ""
                if replay is not None and replay.active:
                    replay_text = " | raw fallback"
                print(
                    "status: "
                    f"{label} {probability * 100.0:.1f}% | "
                    f"nodes={metrics.get('active_nodes', 0)} "
                    f"packets={metrics.get('packet_count', 0)}"
                    f"{replay_text}",
                    flush=True,
                )
            time.sleep(5.0)
    except KeyboardInterrupt:
        pass


def _print_startup(
    engine: FingerprintEngine,
    dashboard: LocationDashboardThread,
    replay: RawDataReplayThread | None,
    leds: ZoneLedControllerThread | None,
) -> None:
    snapshot = engine.snapshot()
    training = snapshot.get("training", {})
    model_ready = bool(training.get("model_ready")) if isinstance(training, dict) else False
    if dashboard.actual_port is not None:
        print(
            "Location dashboard: "
            f"{dashboard.local_url} "
            f"(phone/LAN: {dashboard.lan_url_hint})",
            flush=True,
        )
    elif dashboard.error:
        print(f"Location dashboard failed to start: {dashboard.error}", flush=True)
    print(
        "Integrated CSI app started: "
        f"model_ready={model_ready}, "
        f"raw_fallback={'on' if replay is not None else 'off'}, "
        f"leds={'off' if leds is None else leds.snapshot().get('backend')}",
        flush=True,
    )
    if leds is not None:
        led_state = leds.snapshot()
        print(
            "LED zones: "
            f"{len(led_state.get('zones', []))} configured, "
            f"hardware_ready={led_state.get('hardware_ready')}, "
            f"message={led_state.get('message')}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
