# Tkinter App

This app listens for ADR-018 CSI frames from real ESP32 nodes or from the
simulator, lets you capture fingerprints per grid cell, and shows a live
probability heatmap after you train the saved Learn datasets.

## Shared Config

Startup settings come from `Config/system_config.json`.

- `host.listen_host` and `host.udp_port` control the UDP listener
- `grid.cols` and `grid.rows` define the cell layout
- `fingerprinting.capture_seconds` controls the learn window
- `fingerprinting.effective_packets_per_second`, `fingerprinting.window_sample_count`, and `fingerprinting.window_step_samples` define the paper-style sample-based sliding window used for both dataset generation and live inference
- `fingerprinting.window_seconds` and `fingerprinting.window_step_seconds` are derived views of the sample-based window settings
- `fingerprinting.baseline_start_delay_seconds` adds a short delay before empty-room baseline capture begins
- `nodes[]` provides friendly labels, COM ports for provisioning, per-node UDP
  target ports, and optional CSI send intervals for known ESP32 nodes

When you apply a new grid size inside the app, it writes the updated values
back to the shared config file so the simulator and provisioning flow stay in
sync.

## Run

From the repository root:

```powershell
python -m app
```

Or:

```powershell
cd app
python main.py
```

To verify the live UDP stream without opening the UI:

```powershell
python app\check_udp_ports.py --seconds 10
```

Each active node should appear on its configured port. With the updated
firmware and the default `20 ms` CSI send interval, expect roughly 50 packets
per second per node instead of several hundred.

The app also sends a tiny UDP broadcast stimulus while it is running. This
keeps enough WiFi frames on the channel for the ESP nodes to produce CSI even
when the room network is otherwise quiet. Tune it with `host.stimulus_*` in the
shared config.

## Raw Capture App

Use this when you want to save the raw ESP32 stream for offline validation
without live inference or model training:

```powershell
python -m app.raw_main
```

The raw app uses the same `Config/system_config.json` UDP and grid settings.
Press `Empty Room` or a cell `Learn` button to create one JSONL file per
capture session under `app/raw_data/`. Each packet line includes the capture
start timestamp, packet timestamp, source address, raw packet bytes as base64,
parsed ADR-018 fields, and the derived feature vector. Set `Duration sec` to
`0` for manual stop, or to a positive value for automatic stop.

## Workflow

1. Start the app.
2. Start the simulator or power on the ESP32 nodes.
3. Leave the room empty and press `Capture Baseline`.
4. Stand in a cell and press `Learn`.
5. Repeat `Learn` as many times as needed per cell to accumulate more data.
6. After every cell has saved data, click `Train Models`.
7. The app trains `RandomForestDualStage` and `RandomForestUnified`.
8. Choose the active inference model from the dropdown and watch the live probability heatmap.

## Stored Data

Windowed cell datasets are stored in `app/data/fingerprints.json`.
During `Learn`, the app creates overlapping training samples with the
configured window size and window step, and repeated Learn captures append more
samples to the same cell.

The empty-room baseline used by the paper-style preprocessing pipeline is saved
alongside the dataset in `app/data/fingerprints.json`.

The trained model bundle for `RandomForestDualStage` and
`RandomForestUnified` is stored in `app/data/model_bundle.pkl`.

Communication logs are written to `app/data/communication.log` and are also
shown in the app UI.
