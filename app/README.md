# Tkinter App

This app listens for ADR-018 CSI frames from real ESP32 nodes or from the
simulator, lets you capture fingerprints per grid cell, and shows a live
probability heatmap after you train the saved Learn datasets.

## Shared Config

Startup settings come from `Config/system_config.json`.

- `host.listen_host` and `host.udp_port` control the UDP listener
- `host.keepalive_pings_per_second` controls how often the app sends keepalive pings to each discovered node
- `grid.cols` and `grid.rows` define the cell layout
- `fingerprinting.capture_seconds` controls the learn window
- `fingerprinting.window_seconds` controls the window size used for both dataset generation and live inference
- `fingerprinting.window_step_seconds` controls how far the learn window slides each time when creating overlapping training samples
- `nodes[]` provides friendly labels for known ESP32 nodes

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

## Workflow

1. Start the app.
2. Start the simulator or power on the ESP32 nodes.
3. Stand in a cell and press `Learn`.
4. Repeat `Learn` as many times as needed per cell to accumulate more data.
5. After every cell has saved data, click `Train Models`.
6. The app trains `MLP`, `LogisticRegression`, and `SVM`.
7. Choose the active inference model from the dropdown and watch the live probability heatmap.

## Stored Data

Windowed cell datasets are stored in `app/data/fingerprints.json`.
During `Learn`, the app creates overlapping training samples with the
configured window size and window step, and repeated Learn captures append more
samples to the same cell.

The trained model bundle for `MLP`, `LogisticRegression`, and `SVM` is stored
in `app/data/mlp_model.pkl`.

Communication logs are written to `app/data/communication.log` and are also
shown in the app UI.
