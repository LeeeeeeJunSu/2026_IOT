# Shared Config

Edit `system_config.json` to change the shared settings for the new workspace.

## What lives here

- `host.listen_host`: The UDP bind address used by the desktop app.
- `host.target_ip`: The IP that firmware or the simulator should send CSI frames to.
- `host.udp_port`: Shared UDP port for app, simulator, and provisioning.
- `grid.cols`, `grid.rows`: The Tkinter grid size.
- `fingerprinting.*`: Capture duration, sample-based windowing, baseline delay/capture, preprocessing, and the windowed ExtraTrees feature defaults.
- `simulation.*`: ESP32 traffic simulator behavior.
- `nodes[]`: Per-node IDs, COM ports, WiFi credentials, and channel values.

## Intention

This folder is the single source of truth for values that usually need to be
decided early and reused across the app, simulator, and firmware provisioning
flow.
