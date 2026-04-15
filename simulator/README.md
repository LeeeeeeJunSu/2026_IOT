# Simulator

This folder contains a runnable ESP32 CSI traffic simulator that emits ADR-018
UDP frames to the host configured in `Config/system_config.json`.

## What It Does

- Reads the shared config contract from `Config/config_loader.py`
- Uses the configured `host.target_ip` and `host.udp_port`
- Sends one stream per enabled node in `nodes[]`
- Supports more nodes later without code changes
- Moves a synthetic occupant across the configured grid
- Generates deterministic per-cell CSI fingerprints with repeatable noise
- Produces ADR-018 frames that the Tkinter app can ingest as if they came from
  real ESP32 hardware

## Run It

From the repository root:

```bash
python -m simulator
```

Useful options:

```bash
python -m simulator --duration 30
python -m simulator --dry-run
python -m simulator --config Config/system_config.json
python -m simulator --seed 1234
python -m simulator --cell-sequence "0,0:10@learn_left;1,0:10@learn_right;0,0:5@test_left;1,0:5@test_right"
python -m simulator.diagnose_mlp --seed 1234
```

`--duration` controls how long the simulator runs. If omitted or set to `0`,
the simulator keeps running until you press `Ctrl+C`.

## Configuration

The simulator uses the existing shared config file, so you only need to edit
one place when initial values change:

- `host.target_ip` and `host.udp_port` determine where UDP packets are sent
- `grid.cols` and `grid.rows` define the room grid
- `simulation.tick_hz` controls packet timing
- `simulation.frame_burst_size` controls how many frames each node emits per
  tick
- `simulation.movement_interval_seconds` controls how long the synthetic
  occupant stays in each cell
- `simulation.path_mode` currently supports `snake`, `row_major`,
  `column_major`, and `loop`
- `--cell-sequence` overrides the automatic path and replays an explicit
  schedule, which is useful for mock training sessions that need the simulator
  to stay in one cell for the full learn window
- `nodes[]` controls how many ESP32s are simulated

## Frame Format

Each UDP datagram is an ADR-018 CSI frame:

- 20-byte binary header
- raw signed I/Q bytes for each subcarrier

The simulator keeps the same header layout used by the real ESP32 firmware, so
the app-side parser can treat simulated and hardware traffic the same way.

## Notes

- The generated CSI is deterministic for a given config and seed
- Noise is added per frame so captures are not perfectly flat
- Slow temporal drift and occasional burst interference are injected so the MLP
  sees realistic but still learnable variation
- The simulator prints the current cell to the console when the occupant moves,
  which helps align manual training
