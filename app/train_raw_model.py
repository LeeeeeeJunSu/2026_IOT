from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .raw_training import (
        FeatureConfig,
        ModelConfig,
        default_workspace_root,
        display_label,
        format_candidate_summary,
        load_default_node_ids,
        load_training_system_config,
        train_and_save_raw_model,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from app.raw_training import (
        FeatureConfig,
        ModelConfig,
        default_workspace_root,
        display_label,
        format_candidate_summary,
        load_default_node_ids,
        load_training_system_config,
        train_and_save_raw_model,
    )


def parse_csv_integers(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    workspace_root = default_workspace_root()
    system_config = load_training_system_config(workspace_root)
    default_window = int(system_config.fingerprinting.window_sample_count)
    suggested_windows = sorted({max(3, default_window), max(3, default_window + 2), max(3, default_window + 4)})

    parser = argparse.ArgumentParser(
        description=(
            "Train an offline raw-data classifier from app/raw_data using "
            "chronological train/val splits with a purge gap."
        )
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=workspace_root,
        help="Repository root. Defaults to the current project root.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=workspace_root / "app" / "raw_data",
        help="Directory containing raw JSONL capture files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root / "app" / "data" / "raw_training",
        help="Directory for the report and trained model bundle.",
    )
    parser.add_argument(
        "--effective-pps",
        type=float,
        default=float(system_config.fingerprinting.effective_packets_per_second),
        help="Resampling rate used before windowing.",
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default=",".join(str(value) for value in suggested_windows),
        help="Comma-separated candidate window sizes in resampled samples.",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=int(system_config.fingerprinting.window_step_samples),
        help="Window stride in resampled samples.",
    )
    parser.add_argument(
        "--amplitude-smoothing-candidates",
        type=str,
        default="0,2",
        help="Comma-separated temporal smoothing half-window candidates for CSI amplitudes.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Chronological fraction of each session reserved for training before purge.",
    )
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=0,
        help=(
            "Explicit purge gap in windows between train and val. "
            "Use 0 to auto-select max(15, window_size * 3)."
        ),
    )
    parser.add_argument(
        "--search-cv-splits",
        type=int,
        default=3,
        help="Temporal CV fold count used to rank candidate feature configs on the training split.",
    )
    parser.add_argument(
        "--search-trees",
        type=int,
        default=220,
        help="Tree count used during candidate ranking.",
    )
    parser.add_argument(
        "--final-trees",
        type=int,
        default=320,
        help="Tree count used for the saved final model.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for ExtraTrees.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="ExtraTrees parallelism. Defaults to 1 for sandbox compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_ids = load_default_node_ids(args.workspace_root)

    feature_configs: list[FeatureConfig] = []
    amplitude_smoothing_candidates = parse_csv_integers(args.amplitude_smoothing_candidates)
    for window_size in parse_csv_integers(args.window_sizes):
        if window_size < 2:
            continue
        for amplitude_smoothing in amplitude_smoothing_candidates:
            scalar_smoothing = 0 if amplitude_smoothing <= 0 else max(1, amplitude_smoothing // 2)
            feature_configs.append(
                FeatureConfig(
                    effective_packets_per_second=float(args.effective_pps),
                    window_size=int(window_size),
                    window_step=max(1, int(args.window_step)),
                    amplitude_smoothing_half_window=max(0, int(amplitude_smoothing)),
                    scalar_smoothing_half_window=int(scalar_smoothing),
                    include_quantiles=True,
                    train_ratio=max(0.55, min(0.90, float(args.train_ratio))),
                    purge_gap_windows=max(0, int(args.purge_gap)),
                )
            )
    if not feature_configs:
        raise SystemExit("No valid feature configurations were requested.")

    search_model_config = ModelConfig(
        n_estimators=max(50, int(args.search_trees)),
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
    )
    final_model_config = ModelConfig(
        n_estimators=max(50, int(args.final_trees)),
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
    )

    result = train_and_save_raw_model(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        node_ids=node_ids,
        feature_configs=feature_configs,
        search_model_config=search_model_config,
        final_model_config=final_model_config,
        cv_splits=max(2, int(args.search_cv_splits)),
    )

    print("=== RAW TRAINING ===")
    print(f"nodes={','.join(str(node_id) for node_id in node_ids)}")
    print(f"candidate_count={len(result.candidate_results)}")
    for index, candidate in enumerate(result.candidate_results, start=1):
        print(f"candidate_{index}: {format_candidate_summary(candidate)}")
    print()
    print("=== BEST CONFIG ===")
    print(format_candidate_summary(result.best_candidate))
    print(
        "train_shape="
        f"{result.best_split.X_train.shape[0]}x{result.best_split.X_train.shape[1]} "
        f"val_shape={result.best_split.X_val.shape[0]}x{result.best_split.X_val.shape[1]}"
    )
    print()
    print("=== TRAIN ===")
    print(f"accuracy={result.best_candidate.train_metrics['accuracy']:.4f}")
    print(f"macro_f1={result.best_candidate.train_metrics['macro_f1']:.4f}")
    print()
    print("=== VALIDATION ===")
    print(f"accuracy={result.best_candidate.val_metrics['accuracy']:.4f}")
    print(f"macro_f1={result.best_candidate.val_metrics['macro_f1']:.4f}")
    print("labels=" + ", ".join(display_label(label) for label in result.label_order))
    print("confusion_matrix_rows=true_cols=pred")
    for row in result.best_candidate.val_metrics["confusion_matrix"]:
        print(row)
    print(
        "report_saved_to="
        f"{result.report_path.resolve()}"
    )
    print(
        "model_saved_to="
        f"{result.model_path.resolve()}"
    )


if __name__ == "__main__":
    main()
