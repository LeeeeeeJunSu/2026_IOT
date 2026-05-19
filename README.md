# CSI Fingerprinting Workspace

This workspace is a clean, separated codebase for estimating a person's room
location from ESP32 CSI using a grid-based fingerprinting workflow.

## Structure

- `Config/` - Shared settings for host IP, UDP port, grid size, fingerprinting,
  simulation, and node provisioning
- `firmware/` - ESP32-S3 firmware plus build, flash, and provision helpers
- `app/` - Tkinter desktop app for cell learning and probability visualization
- `simulator/` - ESP32 CSI traffic simulator for hardware-free testing

## Current GT Workflow

From this repository root, activate the project environment first:

```bash
source .venv/bin/activate
```

Capture a new ground-truth sample. `--gt` must be `0` through `6`; use `0` for
the empty/no-person state. The command saves raw CSI under `app/raw_data/`.
After the configured nodes are active, recording starts after a 10 second
countdown.

```bash
python -m app.capture_gt --gt 1 --seconds 60
```

Retrain from all saved GT captures when you are ready:

```bash
python -m app.train_gt_model
```

Run the live sensor receiver, model inference, and web dashboard:

```bash
python -m app --headless
```

Open:

```text
http://127.0.0.1:8000
```

## Single Source Of Truth

Edit `Config/system_config.json`
first when you want to change:

- the UDP host IP and port
- the grid size
- capture defaults
- simulator behavior
- per-node IDs, WiFi, and COM port mapping

The goal is to avoid scattering initial setup values across multiple scripts.
