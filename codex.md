# Codex Analysis Notes

This document summarizes the CSI fingerprinting analysis performed on the
`app/raw_data` ground-truth captures and the follow-up model experiments.

## Project Setup Observed

- Goal: track a person in a 2 x 3 room grid using CSI from 10 ESP32 nodes.
- GT labels:
  - `GT0`: empty room baseline
  - `GT1`: front-left
  - `GT2`: front-right
  - `GT3`: middle-left
  - `GT4`: middle-right
  - `GT5`: back-left
  - `GT6`: back-right
- Sensor placement:
  - Right wall: ESP 1, 2, 3, 4
  - Left wall: ESP 5, 6, 7, 8
  - Front: ESP 9
  - Back: ESP 10
  - Wi-Fi AP/router: near the center

## Raw Data Summary

- Raw files analyzed: 35 JSONL files in `app/raw_data`
- Valid packet records parsed: 587,430
- GT labels present: 0 through 6
- ESP nodes present: 1 through 10
- Main generated outputs:
  - `app/analysis_plots/gt_node_summary.csv`
  - `app/analysis_plots/feature_vector_means.csv`
  - `app/analysis_plots/feature_profiles_by_gt_per_node.svg`
  - `app/analysis_plots/geometry_aware_analysis.md`
  - `app/analysis_plots/geometry_delta_sensor_layout_en.svg`
  - `app/analysis_plots/geometry_delta_feature_l2_matrix_en.svg`
  - `app/analysis_plots/geometry_delta_amp_matrix_en.svg`
  - `app/analysis_plots/geometry_delta_rssi_matrix_en.svg`

## ESP32 CSI Collection Flow

Firmware path:

- `firmware/esp32-csi-fingerprint-node/main/csi_fingerprint.c`
- `firmware/esp32-csi-fingerprint-node/main/adr018.c`
- `firmware/esp32-csi-fingerprint-node/main/udp_sender.c`

The ESP32 firmware:

1. Connects to Wi-Fi in station mode.
2. Enables promiscuous mode and CSI capture.
3. Uses CSI settings with LLTF, HT-LTF, and STBC HT-LTF2 enabled.
4. Applies a send throttle using `csi_send_interval_ms`, usually 20 ms.
5. Serializes CSI into ADR-018 UDP frames.
6. Sends raw signed int8 I/Q CSI bytes over UDP.

The ESP does not compute the final ML features. It mainly forwards raw CSI I/Q
with metadata:

- node id
- antenna count
- subcarrier count
- frequency
- sequence number
- RSSI
- noise floor
- raw I/Q bytes

## Host-Side CSI Feature Processing

Host parsing path:

- `app/protocol.py`
- `app/raw_capture.py`
- `app/train_gt_model.py`

Host processing:

1. Decode ADR-018 payload.
2. Convert each I/Q pair into:
   - amplitude: `sqrt(i*i + q*q)`
   - phase: `atan2(q, i)`
3. FFT-shift subcarriers.
4. Select active subcarriers:
   - from -25 to +25
   - exclude DC
   - exclude pilot subcarriers -21, -7, 7, 21
   - result: 46 active subcarrier values
5. Store JSONL packet records with:
   - raw base64
   - amplitudes
   - phases
   - 46-value `feature_vector`
   - scalar features such as RSSI, SNR, amplitude mean/std/rms/p90,
     gradient mean, and phase step std

## Important Data Findings

The raw subcarrier profile shape is mostly stable across GT labels. This is
expected because the dominant shape is the static channel response from:

- ESP/AP placement
- antenna characteristics
- room geometry
- walls/furniture
- static multipath
- ESP32 CSI scaling/quantization

The human location signal appears as a smaller perturbation on top of this
stable channel shape.

### GT Separation by Simple Amplitude Mean

The first coarse analysis showed that GT-to-GT differences are often smaller
than within-GT variation. For example:

- Best node by `amplitude_mean` separation: ESP 2
- ESP 2 between-GT std: 1.118
- ESP 2 within-GT std average: 5.350
- Ratio: 0.209

This means simple average amplitude alone is not enough for robust location
classification.

## Geometry-Aware Findings

Using GT0 as the empty-room baseline, feature-vector L2 deltas were compared
against the sensor layout.

### Front Cells and ESP9

ESP9 does respond to front cells:

- GT1 on ESP9:
  - feature L2 delta: 1.403
  - amplitude mean delta: +0.781
  - RSSI delta: +2.254 dB
- GT2 on ESP9:
  - feature L2 delta: 1.954
  - amplitude mean delta: -0.234
  - RSSI delta: +2.275 dB

However, ESP9 also changes strongly for other locations such as GT5 and GT6.
So it is not a clean "front-only" sensor.

### Left Cells and ESP5-8

Left-wall sensors show meaningful response for left-side GT labels:

- GT1 left-wall average L2: 2.186
- GT3 left-wall average L2: 2.658
- GT5 left-wall average L2: 2.390

But the response is not isolated to the left wall. ESP2 and ESP9 also show
large changes for some left-side labels, likely due to multipath and AP-person-ESP
path interactions.

### Right Cells and ESP1-4

Right-wall response is present but not perfectly clean:

- GT2 right-wall average L2: 1.277
- GT4 right-wall average L2: 2.031
- GT6 right-wall average L2: 1.719

GT2, for example, had ESP8 as the strongest feature delta, which shows that
simple nearest-sensor intuition is not enough for CSI.

## Why Tree Accuracy Was High but Real Operation Failed

The original GT model used `ExtraTreesClassifier`, specifically:

- model name: `VariableNodeAggregateExtraTrees`
- estimator count: 320
- max features: `sqrt`
- class weight: `balanced_subsample`

Original reported metrics in `app/data/gt_training_report.json`:

- sample count: 19,138
- feature dimension: 984
- baseline source: GT0
- chronological tail test accuracy: about 0.8197

The likely issue is that the split was still too close to the training data:
the original validation split used a tail segment from each capture session.
CSI samples have strong temporal autocorrelation, so this can overestimate
real-world performance.

When ExtraTrees was evaluated using a stricter split, holding out the last
session for each GT label:

- accuracy: 0.7770
- macro F1: 0.7666

Main confusion cases:

- GT3 confused with GT4/GT6
- GT5 confused heavily with GT6
- GT4 confused with GT3

This is consistent with the geometry plots: some neighboring or path-related
locations have similar CSI perturbations.

## Deep Learning Experiment

A new virtual environment was created:

- `.venv5070`

GPU stack confirmed:

- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- GPU: `NVIDIA GeForce RTX 5070 Ti`

For Raspberry Pi 5 / Ubuntu 24.04 deployment, use the CPU-only PyTorch
requirements instead:

```bash
python3 -m venv .venv-pi
source .venv-pi/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-pi-cpu.txt
```

CUDA is not available on Raspberry Pi. The app loads deep models onto CPU when
`torch.cuda.is_available()` is false. For CUDA workstation training, install
`requirements-cuda-cu128.txt` after the common runtime requirements.

New training script:

- `app/train_deep_gt_model.py`
- The script now supports versioned model experiments:
  - `cnn_v1`: original temporal CNN
  - `cnn_v2`: residual/dilated temporal CNN
  - `gru_v1`: bidirectional GRU
  - `lstm_v1`: bidirectional LSTM
  - `transformer_v1`: compact Transformer encoder
- Aliases are also accepted:
  - `cnn` -> `cnn_v1`
  - `tcn` -> `cnn_v2`
  - `gru` or `rnn` -> `gru_v1`
  - `lstm` -> `lstm_v1`
  - `transformer` -> `transformer_v1`

The deep dataset uses live-like rolling horizon tensors:

- sample count: 19,138
- input shape: `[54, 540]`
- horizon samples: 54
- input size per time step: 540
- split: `leave_last_session_per_gt`
- train samples: 15,309
- test samples: 3,829

The input dimension is:

- 10 ESP nodes
- each node contributes 46 active subcarrier amplitudes plus 8 scalar features
- 10 x (46 + 8) = 540 features per time step

### Model Results

Using the same leave-last-session-per-GT split:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| ExtraTrees | 0.7770 | 0.7666 |
| Temporal CNN | 0.9593 | 0.9596 |
| GRU/RNN | 0.9274 | 0.9279 |

Saved outputs:

- `app/data/deep_gt_training/deep_gt_training_report.json`
- `app/data/deep_gt_training/cnn_gt_model.pt`
- `app/data/deep_gt_training/gru_gt_model.pt`

The live app now scans `app/data/deep_gt_training/` and exposes saved `.pt`
models alongside any existing ExtraTrees model bundle. On the current local
workspace, the loadable models are:

- `DeepCNNV1`
- `DeepGRUV1`

List them with:

```powershell
.venv5070\Scripts\python.exe -m app --list-models
```

Run the dashboard with a selected model:

```powershell
.venv5070\Scripts\python.exe -m app --headless --model DeepCNNV1
```

For the dashboard-only entrypoint:

```powershell
.venv5070\Scripts\python.exe -m app.dashboard_main --model DeepCNNV1
```

## Environment Runbook

### Windows RTX 5070 Ti

Purpose: CUDA model training, local validation, dashboard smoke tests.

```powershell
cd "C:\Users\rltjr\Desktop\아주대학교\4학년\융합시스템공학종합설계\2026_IOT"
.\.venv5070\Scripts\Activate.ps1
python -m app --list-models
python -m app.train_deep_gt_model --models cnn_v1,cnn_v2,gru_v1,lstm_v1,transformer_v1 --epochs 20 --batch-size 256
python -m app --headless --model DeepCNNV1
```

### Raspberry Pi 5 / Ubuntu 24.04

Purpose: live ESP32 UDP receiver, model inference, web dashboard, optional GPIO
LED control. CUDA is not available; deep models run on CPU.

```bash
cd ~/2026_IOT
source .venv-pi/bin/activate
python -m app --list-models
python -m app --headless --model DeepCNNV1
```

If PyTorch CPU inference is too slow or unavailable:

```bash
python -m app --headless --model VariableNodeAggregateExtraTrees
```

### Existing Raw Data Demo

```bash
python -m app --headless --model DeepCNNV1 --fallback-after-seconds 5 --replay-speedup 20
```

### Deep Model Replay Fix

The dashboard originally showed packets and all 10 nodes but no location
probabilities with `DeepCNNV1`. Two issues were found and fixed:

1. `app/data/fingerprints.json` was missing, so no empty-room baseline was
   loaded. Deep live inference now rebuilds the baseline from
   `app/raw_data/*gt_0.jsonl` when saved baseline metadata is absent.
2. Raw replay was feeding packets with their original capture timestamps. Live
   inference prunes old timestamps from rolling windows, so replay now injects
   packets with current wall-clock time.

Smoke validation after the fix:

- `DeepCNNV1` loaded successfully.
- GT0 baseline rebuilt from raw JSONL.
- A live deep sample with shape `(54, 540)` was generated.
- GT0 replay produced `GT 0` with high confidence.

Example command for a broader model sweep:

```powershell
.venv5070\Scripts\python.exe -m app.train_deep_gt_model --models cnn_v1,cnn_v2,gru_v1,lstm_v1,transformer_v1 --epochs 20 --batch-size 256
```

### Interpretation

The Temporal CNN performed best on the stricter session holdout split. This
suggests that time-window structure contains useful information that the
aggregate ExtraTrees features were not fully preserving.

The GRU also beat ExtraTrees, but the CNN did better. For this dataset, a CNN
appears to be a better fit than an RNN because the useful CSI pattern is likely
local in time and channel dimensions rather than requiring long recurrent
memory.

## Remaining Cautions

The CNN result is promising, but it still needs validation under real deployment
conditions:

- different day/time
- person facing different directions
- walking instead of standing still
- furniture or door state changes
- different Wi-Fi traffic levels
- live packet loss or node dropouts
- exact node-id-to-physical-position consistency

The current result says:

- The dataset contains location information.
- CNN/RNN can extract more of it than the previous tree model.
- The earlier ExtraTrees validation likely overestimated real performance.

It does not yet prove robust live tracking across changing conditions.

## Recommended Next Steps

1. Integrate the trained CNN into live inference or replay inference.
2. Run raw UDP replay using held-out sessions and compare predictions over time.
3. Collect a new validation set on a different day.
4. Collect dynamic samples with natural movement and body orientation changes.
5. Add per-session/day holdout evaluation as the default benchmark.
6. Confirm physical ESP placement and firmware node IDs before every capture.
7. Consider adding probability smoothing and confidence rejection for unstable
   live predictions.
