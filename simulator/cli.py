from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulator",
        description="Run a deterministic ESP32 CSI UDP traffic simulator.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to system_config.json. Defaults to Config/system_config.json.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="How long to run in seconds. Use 0 to run until Ctrl+C.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for deterministic fingerprint generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send UDP. Print the generated frame summaries instead.",
    )
    parser.add_argument(
        "--cell-sequence",
        type=str,
        default="",
        help=(
            "Optional scripted cell schedule as x,y:seconds;... "
            "for mock training or inference replays. "
            "Example: 0,0:10@learn_a;1,0:10@learn_b;0,0:5@test_a"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if __package__ in {None, ""}:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    from Config.config_loader import load_system_config

    from simulator.engine import Esp32TrafficSimulator, parse_cell_sequence

    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_system_config(args.config)
    scenario_steps = parse_cell_sequence(
        args.cell_sequence,
        config.grid.cols,
        config.grid.rows,
    )
    if args.duration > 0:
        duration = float(args.duration)
    elif scenario_steps:
        duration = sum(step.duration_seconds for step in scenario_steps)
    else:
        duration = None

    simulator = Esp32TrafficSimulator(
        config=config,
        seed=args.seed,
        dry_run=bool(args.dry_run),
        scenario_steps=scenario_steps,
    )
    simulator.run(duration_seconds=duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
