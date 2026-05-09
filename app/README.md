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

`python -m app` now starts the integrated Raspberry Pi runtime:

- UDP receivers for the configured ESP32 nodes
- the model inference runtime
- the Tkinter GUI when Tkinter is installed
- the phone-friendly web dashboard
- raw-data replay fallback when no live ESP32 signal is detected

Open the dashboard from a phone on the same WiFi:

```text
http://raspberrypi-csi.local:8000
```

For a web-only/headless run:

```powershell
python -m app.integrated_main --headless
```

Raw-data replay fallback starts after 15 seconds without live ESP32 packets by
default. Override it when needed:

```powershell
python -m app --fallback-after-seconds 30
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

## Offline Raw-Data Training

When you want to benchmark the raw JSONL captures directly instead of going
through the live app dataset store, run:

```powershell
python -m app.train_raw_model
```

The offline trainer:

- uses the empty-room capture both as an explicit `Empty Room` class and as the
  per-node baseline used to center CSI amplitudes
- resamples each node stream to the configured effective packet rate
- fills bucket gaps with forward/backward fill plus a baseline fallback
- builds overlapping time windows and concatenates node-wise summary features
- uses a chronological train/val split with a purge gap to reduce overlap
  leakage between adjacent windows

Artifacts are written to `app/data/raw_training/`:

- `raw_training_report.json`
- `raw_training_model.pkl`

## Workflow

1. Start the app.
2. Start the simulator or power on the ESP32 nodes.
3. Leave the room empty and press `Capture Baseline`.
4. Stand in a cell and press `Learn`.
5. Repeat `Learn` as many times as needed per cell to accumulate more data.
6. After every cell has saved data, click `Train Models`.
7. The app trains `ExtraTreesWindowed`.
8. Choose the active inference model from the dropdown and watch the live probability heatmap.

## Stored Data

Windowed cell datasets are stored in `app/data/fingerprints.json`.
During `Learn`, the app creates overlapping training samples with the
configured window size and window step. Each sample concatenates node-wise CSI
window statistics: amplitude mean/std, amplitude quantiles, and scalar
telemetry mean/std. Repeated Learn captures append more samples to the same
cell.

The empty-room baseline used to center each node's CSI amplitudes is saved
alongside the dataset in `app/data/fingerprints.json`.

The trained `ExtraTreesWindowed` model bundle is stored in
`app/data/model_bundle.pkl`.

Communication logs are written to `app/data/communication.log` and are also
shown in the app UI.
