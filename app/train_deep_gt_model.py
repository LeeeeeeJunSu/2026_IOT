from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from app.core import FingerprintEngine
    from app.raw_training import SCALAR_KEYS
    from app.train_gt_model import (
        GtSession,
        build_baseline,
        build_session_rows_for_live_horizons,
        load_gt_sessions,
    )
except ImportError:  # pragma: no cover
    from core import FingerprintEngine
    from raw_training import SCALAR_KEYS
    from train_gt_model import (
        GtSession,
        build_baseline,
        build_session_rows_for_live_horizons,
        load_gt_sessions,
    )


@dataclass(frozen=True)
class DeepDataset:
    x: np.ndarray
    y: np.ndarray
    session_ids: list[str]
    labels: list[str]
    input_size: int
    horizon_samples: int


class TemporalCnn(nn.Module):
    def __init__(self, input_size: int, class_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.transpose(1, 2)))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=dilation * 2,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TemporalCnnV2(nn.Module):
    def __init__(self, input_size: int, class_count: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, 192, kernel_size=1),
            nn.BatchNorm1d(192),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(192, dilation=1, dropout=0.12),
            ResidualTemporalBlock(192, dilation=2, dropout=0.12),
            ResidualTemporalBlock(192, dilation=4, dropout=0.12),
            ResidualTemporalBlock(192, dilation=1, dropout=0.12),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 160),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(160, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.stem(x.transpose(1, 2))
        values = self.blocks(values)
        return self.head(self.pool(values))


class TemporalGru(nn.Module):
    def __init__(self, input_size: int, class_count: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=192,
            num_layers=2,
            batch_first=True,
            dropout=0.20,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 160),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(160, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :])


class TemporalLstm(nn.Module):
    def __init__(self, input_size: int, class_count: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=160,
            num_layers=2,
            batch_first=True,
            dropout=0.20,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(320),
            nn.Linear(320, 160),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(160, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


class TemporalTransformer(nn.Module):
    def __init__(self, input_size: int, class_count: int, max_steps: int = 128) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, 192)
        self.position = nn.Parameter(torch.zeros(1, max_steps, 192))
        layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=6,
            dim_feedforward=384,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = x.shape[1]
        values = self.projection(x) + self.position[:, :steps, :]
        values = self.encoder(values)
        return self.head(values.mean(dim=1))


MODEL_ALIASES = {
    "cnn": "cnn_v1",
    "tcn": "cnn_v2",
    "gru": "gru_v1",
    "rnn": "gru_v1",
    "lstm": "lstm_v1",
    "transformer": "transformer_v1",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train versioned deep GT classifiers from raw CSI JSONL.")
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument("--raw-dir", type=Path, default=root / "app" / "raw_data")
    parser.add_argument("--output-dir", type=Path, default=root / "app" / "data" / "deep_gt_training")
    parser.add_argument(
        "--models",
        default="cnn_v1,cnn_v2,gru_v1",
        help=(
            "Comma-separated model versions. Supported: "
            "cnn_v1, cnn_v2, gru_v1, lstm_v1, transformer_v1. "
            "Aliases: cnn, tcn, gru, rnn, lstm, transformer."
        ),
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_model_names(raw: str) -> list[str]:
    names: list[str] = []
    supported = {"cnn_v1", "cnn_v2", "gru_v1", "lstm_v1", "transformer_v1"}
    for part in raw.split(","):
        requested = part.strip().lower()
        if not requested:
            continue
        name = MODEL_ALIASES.get(requested, requested)
        if name not in supported:
            raise ValueError(f"Unsupported model version: {requested}")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("At least one model version must be requested.")
    return names


def main() -> int:
    args = parse_args()
    result = train_deep_models(
        workspace_root=args.workspace_root,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        model_names=parse_model_names(args.models),
        epochs=max(1, int(args.epochs)),
        batch_size=max(16, int(args.batch_size)),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        random_state=int(args.random_state),
        n_workers=max(0, int(args.n_workers)),
    )
    print(json.dumps(result, indent=2))
    return 0


def train_deep_models(
    *,
    workspace_root: Path,
    raw_dir: Path,
    output_dir: Path,
    model_names: list[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    random_state: int,
    n_workers: int,
) -> dict[str, Any]:
    set_seed(random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_deep_dataset(workspace_root=workspace_root, raw_dir=raw_dir)
    train_idx, test_idx, split_summary = build_leave_last_session_split(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = torch.from_numpy(dataset.x[train_idx]).float()
    y_train = torch.from_numpy(dataset.y[train_idx]).long()
    x_test = torch.from_numpy(dataset.x[test_idx]).float()
    y_test = torch.from_numpy(dataset.y[test_idx]).long()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=device.type == "cuda",
    )

    results: dict[str, Any] = {
        "created_at_unix": time.time(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "raw_dir": str(Path(raw_dir).resolve()),
        "labels": dataset.labels,
        "sample_count": int(dataset.x.shape[0]),
        "train_sample_count": int(len(train_idx)),
        "test_sample_count": int(len(test_idx)),
        "input_shape": list(dataset.x.shape[1:]),
        "input_size": int(dataset.input_size),
        "horizon_samples": int(dataset.horizon_samples),
        "split_kind": "leave_last_session_per_gt",
        "split_summary": split_summary,
        "models": {},
    }

    for model_name in model_names:
        model = make_model(model_name, dataset.input_size, len(dataset.labels)).to(device)
        model_result = fit_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            test_loader=test_loader,
            y_test=y_test.numpy(),
            label_count=len(dataset.labels),
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        model_path = output_dir / f"{model_name}_gt_model.pt"
        torch.save(
            {
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "labels": dataset.labels,
                "input_size": dataset.input_size,
                "horizon_samples": dataset.horizon_samples,
                "input_shape": list(dataset.x.shape[1:]),
                "parameter_count": count_parameters(model),
                "metrics": model_result,
            },
            model_path,
        )
        model_result["model_path"] = str(model_path.resolve())
        model_result["parameter_count"] = count_parameters(model)
        results["models"][model_name] = model_result

    report_path = output_dir / "deep_gt_training_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["report_path"] = str(report_path.resolve())
    return results


def make_model(model_name: str, input_size: int, class_count: int) -> nn.Module:
    if model_name == "cnn_v1":
        return TemporalCnn(input_size, class_count)
    if model_name == "cnn_v2":
        return TemporalCnnV2(input_size, class_count)
    if model_name == "gru_v1":
        return TemporalGru(input_size, class_count)
    if model_name == "lstm_v1":
        return TemporalLstm(input_size, class_count)
    if model_name == "transformer_v1":
        return TemporalTransformer(input_size, class_count)
    raise ValueError(f"Unsupported deep model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def fit_model(
    *,
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    y_test: np.ndarray,
    label_count: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * int(yb.numel())
            total_count += int(yb.numel())
            correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
        scheduler.step()
        test_predictions = predict(model, test_loader, device)
        test_accuracy = float(accuracy_score(y_test, test_predictions))
        train_accuracy = correct / max(1, total_count)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(1, total_count),
                "train_accuracy": float(train_accuracy),
                "test_accuracy": test_accuracy,
            }
        )
        print(
            f"{model_name} epoch {epoch:03d}/{epochs} "
            f"loss={history[-1]['train_loss']:.4f} "
            f"train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f}",
            flush=True,
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = predict(model, test_loader, device)
    return {
        "best_test_accuracy": float(accuracy_score(y_test, predictions)),
        "best_test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=list(range(label_count)),
        ).tolist(),
        "history": history,
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for xb, _ in loader:
        logits = model(xb.to(device, non_blocking=True))
        outputs.append(logits.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.int64)


def build_deep_dataset(*, workspace_root: Path, raw_dir: Path) -> DeepDataset:
    app_root = Path(workspace_root) / "app"
    engine = FingerprintEngine(app_root)
    sessions = load_gt_sessions(raw_dir)
    if not sessions:
        raise RuntimeError(f"No GT sessions found in {raw_dir}")

    required_node_ids = engine.required_node_ids
    baseline_by_node, baseline_counts, _baseline_source = build_baseline(sessions, required_node_ids)
    engine.empty_room_baseline_by_node = baseline_by_node
    engine.empty_room_baseline_counts = baseline_counts

    labels = sorted({str(session.gt_location) for session in sessions}, key=int)
    label_to_index = {label: index for index, label in enumerate(labels)}
    horizon_samples = max(
        engine.window_sample_count + engine.smoothing_half_window * 2,
        engine.window_sample_count * engine.LIVE_PREPROCESS_HORIZON_WINDOWS,
    )
    feature_size_per_node = 46 + len(SCALAR_KEYS)
    input_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    sample_session_ids: list[str] = []

    with engine.lock:
        for session in sessions:
            x_session = build_session_tensor(
                engine=engine,
                session=session,
                horizon_samples=horizon_samples,
                feature_size_per_node=feature_size_per_node,
            )
            if x_session.size == 0:
                continue
            input_rows.append(x_session)
            target_rows.append(
                np.full((x_session.shape[0],), label_to_index[str(session.gt_location)], dtype=np.int64)
            )
            sample_session_ids.extend([session.session_id] * x_session.shape[0])

    if not input_rows:
        raise RuntimeError("No deep learning samples could be generated.")
    x = np.concatenate(input_rows, axis=0).astype(np.float16, copy=False)
    y = np.concatenate(target_rows, axis=0)
    return DeepDataset(
        x=x,
        y=y,
        session_ids=sample_session_ids,
        labels=labels,
        input_size=len(required_node_ids) * feature_size_per_node,
        horizon_samples=horizon_samples,
    )


def build_session_tensor(
    *,
    engine: FingerprintEngine,
    session: GtSession,
    horizon_samples: int,
    feature_size_per_node: int,
) -> np.ndarray:
    all_frames = [frame for frames in session.frames_by_node.values() for frame in frames]
    if not all_frames:
        return np.empty((0, horizon_samples, engine.required_node_ids.__len__() * feature_size_per_node))
    start_time = min(frame.captured_at for frame in all_frames)
    end_time = max(frame.captured_at for frame in all_frames)
    aligned_vectors_by_node, aligned_scalars_by_node, observed_by_node, total_slots = (
        build_session_rows_for_live_horizons(engine, session, start_time=start_time, end_time=end_time)
    )
    if total_slots < horizon_samples:
        return np.empty((0, horizon_samples, len(engine.required_node_ids) * feature_size_per_node))

    node_tensors: list[np.ndarray] = []
    required_recent_samples = engine.window_sample_count + engine.smoothing_half_window * 2
    valid_mask = np.ones(total_slots, dtype=bool)
    for node_id in engine.required_node_ids:
        valid_mask &= np.asarray(observed_by_node[node_id][:total_slots], dtype=bool)
        node_tensors.append(
            preprocess_node_windows(
                engine=engine,
                node_id=node_id,
                amplitude_rows=aligned_vectors_by_node[node_id][:total_slots],
                scalar_rows=aligned_scalars_by_node[node_id][:total_slots],
                horizon_samples=horizon_samples,
            )
        )

    if any(tensor.size == 0 for tensor in node_tensors):
        return np.empty((0, horizon_samples, len(engine.required_node_ids) * feature_size_per_node))

    cumulative_valid = np.concatenate(([0], np.cumsum(valid_mask.astype(np.int32))))
    valid_counts = cumulative_valid[horizon_samples:] - cumulative_valid[:-horizon_samples]
    keep = valid_counts >= required_recent_samples
    if not np.any(keep):
        return np.empty((0, horizon_samples, len(engine.required_node_ids) * feature_size_per_node))

    x = np.concatenate(node_tensors, axis=2)
    return x[keep].astype(np.float16, copy=False)


def preprocess_node_windows(
    *,
    engine: FingerprintEngine,
    node_id: int,
    amplitude_rows: list[list[float]],
    scalar_rows: list[list[float]],
    horizon_samples: int,
) -> np.ndarray:
    amplitude_array = np.asarray(amplitude_rows, dtype=np.float32)
    scalar_array = np.asarray(scalar_rows, dtype=np.float32)
    if amplitude_array.shape[0] < horizon_samples:
        return np.empty((0, horizon_samples, 46 + len(SCALAR_KEYS)), dtype=np.float32)

    amplitude_windows = np.lib.stride_tricks.sliding_window_view(
        amplitude_array,
        window_shape=horizon_samples,
        axis=0,
    ).transpose(0, 2, 1)
    scalar_windows = np.lib.stride_tricks.sliding_window_view(
        scalar_array,
        window_shape=horizon_samples,
        axis=0,
    ).transpose(0, 2, 1)

    baseline = np.asarray(
        engine._baseline_vector_for_node_locked(node_id, amplitude_rows),
        dtype=np.float32,
    )
    centered = amplitude_windows - baseline.reshape(1, 1, -1)
    scales = np.percentile(np.abs(centered), 95.0, axis=1)
    normalized_amplitudes = centered / np.maximum(scales[:, np.newaxis, :], 1e-6)

    scalar_mean = scalar_windows.mean(axis=1)
    scalar_std = scalar_windows.std(axis=1)
    normalized_scalars = (
        scalar_windows - scalar_mean[:, np.newaxis, :]
    ) / np.maximum(scalar_std[:, np.newaxis, :], 1e-6)
    return np.concatenate([normalized_amplitudes, normalized_scalars], axis=2).astype(
        np.float32,
        copy=False,
    )


def build_leave_last_session_split(dataset: DeepDataset) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    by_label_session: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (label, session_id) in enumerate(zip(dataset.y, dataset.session_ids)):
        by_label_session[int(label)][session_id].append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []
    summary: list[dict[str, Any]] = []
    for label in sorted(by_label_session):
        sessions = sorted(by_label_session[label])
        test_session = sessions[-1]
        for session_id in sessions:
            target = test_indices if session_id == test_session else train_indices
            target.extend(by_label_session[label][session_id])
        summary.append(
            {
                "label": dataset.labels[label],
                "test_session_id": test_session,
                "train_sessions": len(sessions) - 1,
                "test_samples": len(by_label_session[label][test_session]),
            }
        )
    return (
        np.asarray(sorted(train_indices), dtype=np.int64),
        np.asarray(sorted(test_indices), dtype=np.int64),
        summary,
    )


if __name__ == "__main__":
    raise SystemExit(main())
