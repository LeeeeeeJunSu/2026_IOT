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

For Raspberry Pi 5 on Ubuntu 24.04, use a CPU-only PyTorch install:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-tk python3-gpiozero python3-lgpio
python3 -m venv .venv-pi
source .venv-pi/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-pi-cpu.txt
```

Check that the Pi runtime is CPU-only:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

`torch.cuda.is_available()` should print `False` on Raspberry Pi.

For RTX/CUDA training on Windows or Linux, install the common requirements and
then the CUDA 12.8 PyTorch wheel:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda-cu128.txt
```

## Environment-Specific Runbooks

### Windows / RTX 5070 Ti Training And Replay

Use this environment for model training and offline validation.

```powershell
cd "C:\Users\rltjr\Desktop\아주대학교\4학년\융합시스템공학종합설계\2026_IOT"
.\.venv5070\Scripts\Activate.ps1
python -m app --list-models
```

Train deep models:

```powershell
python -m app.train_deep_gt_model --models cnn_v1,cnn_v2,gru_v1,lstm_v1,transformer_v1 --epochs 20 --batch-size 256
```

Run the web dashboard with a selected saved model:

```powershell
python -m app --headless --model DeepCNNV1
```

Run dashboard-only mode:

```powershell
python -m app.dashboard_main --model DeepCNNV1
```

### Raspberry Pi 5 / Ubuntu 24.04 Live Runtime

Use this environment for final live ESP32 receiver, inference, web dashboard,
and optional GPIO LED control. The Pi uses CPU inference, not CUDA.

```bash
cd ~/2026_IOT
source .venv-pi/bin/activate
python -m app --list-models
python -m app --headless --model DeepCNNV1
```

Open the dashboard from another device on the same network:

```text
http://<raspberry-pi-ip>:8000
```

If deep models are too slow on CPU, use the existing ExtraTrees model or export
the CNN to ONNX for CPU inference.

### Raspberry Pi 5 Without Deep Model Dependencies

If PyTorch is not installed or fails to install on the Pi, the app still runs
with any available ExtraTrees model bundle:

```bash
source .venv-pi/bin/activate
python -m app --list-models
python -m app --headless --model VariableNodeAggregateExtraTrees
```

### Raw Data Fallback / Demo Mode

The integrated app can replay `app/raw_data` when live ESP32 packets are not
arriving:

```bash
python -m app --headless --model DeepCNNV1 --fallback-after-seconds 5 --replay-speedup 20
```

If the dashboard shows packets and nodes but no location probabilities, check:

- `python -m app --list-models` shows the selected model.
- GT0 empty-room raw captures exist under `app/raw_data/*gt_0.jsonl`.
- The app log contains a baseline message. If `app/data/fingerprints.json` is
  missing, the app automatically rebuilds the empty-room baseline from GT0 raw
  JSONL files.
- For raw replay demos, replay packets are timestamped with current time so the
  live inference window can form correctly.

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

Train or refresh the PyTorch deep models on an RTX/CUDA machine:

```bash
python -m app.train_deep_gt_model --models cnn_v1,cnn_v2,gru_v1,lstm_v1,transformer_v1 --epochs 20 --batch-size 256
```

Currently supported deep model versions are `cnn_v1`, `cnn_v2`, `gru_v1`,
`lstm_v1`, and `transformer_v1`. Short aliases such as `cnn`, `gru`, `lstm`,
and `transformer` are accepted.

List models that the web app can load:

```bash
python -m app --list-models
```

Run the live sensor receiver, model inference, and web dashboard:

```bash
python -m app --headless
```

Pick a model when starting the dashboard:

```bash
python -m app --headless --model DeepCNNV1
```

The current local deep model artifacts are loaded from
`app/data/deep_gt_training/`. On Raspberry Pi, these models run on CPU; CUDA is
only available on NVIDIA GPU systems.

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
