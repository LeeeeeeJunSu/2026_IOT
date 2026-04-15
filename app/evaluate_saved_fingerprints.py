from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved fingerprint datasets with RandomForest classifiers."
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "fingerprints.json",
        help="Path to fingerprints.json",
    )
    return parser.parse_args()


def display_cell(cell_key: str) -> str:
    grid_x, grid_y = [int(value) for value in cell_key.split(",")]
    return f"({grid_x + 1},{grid_y + 1})"


def make_model(_input_size: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=320,
        random_state=200,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        n_jobs=-1,
    )


def main() -> None:
    args = parse_args()
    raw = json.loads(args.store.read_text(encoding="utf-8"))
    cells = OrderedDict(
        sorted(raw.get("cells", {}).items(), key=lambda item: (item[1]["grid_y"], item[1]["grid_x"]))
    )
    if not cells:
        raise SystemExit("No saved cell datasets found.")

    window_seconds = float(raw.get("window_seconds", 1.0))
    window_step_seconds = float(raw.get("window_step_seconds", window_seconds))
    input_size = int(raw.get("input_size", 0))
    node_ids = [int(value) for value in raw.get("node_ids", [])]

    features: list[list[float]] = []
    labels: list[str] = []
    counts: OrderedDict[str, int] = OrderedDict()
    for cell_key, payload in cells.items():
        samples = payload.get("samples", [])
        counts[cell_key] = len(samples)
        for sample in samples:
            features.append([float(value) for value in sample])
            labels.append(cell_key)

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[0] == 0:
        raise SystemExit("Saved dataset is empty.")
    if input_size <= 0:
        input_size = int(X.shape[1])

    ordered_labels = list(cells.keys())
    target_names = [display_cell(cell_key) for cell_key in ordered_labels]

    train_model = make_model(input_size)
    train_model.fit(X, y)
    train_pred = train_model.predict(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=200)
    cv_pred = cross_val_predict(make_model(input_size), X, y, cv=cv)

    gap = max(1, int(math.ceil(window_seconds / max(window_step_seconds, 1e-9))))
    train_indices: list[int] = []
    test_indices: list[int] = []
    purged_summary: list[tuple[str, int, int, int]] = []
    start_index = 0
    for cell_key in ordered_labels:
        total = counts[cell_key]
        split_index = int(round(total * 0.7))
        train_end = max(1, split_index - gap)
        test_start = min(total - 1, split_index + gap)
        if test_start <= train_end:
            midpoint = total // 2
            train_end = max(1, midpoint - gap)
            test_start = min(total - 1, midpoint + gap)
        local_train = list(range(start_index, start_index + train_end))
        local_test = list(range(start_index + test_start, start_index + total))
        train_indices.extend(local_train)
        test_indices.extend(local_test)
        purged_summary.append((cell_key, total, len(local_train), len(local_test)))
        start_index += total

    purged_results: dict[str, object] = {
        "gap": gap,
        "summary": purged_summary,
        "available": False,
    }
    if train_indices and test_indices and len(set(y[train_indices])) == len(ordered_labels):
        purged_model = make_model(input_size)
        purged_model.fit(X[train_indices], y[train_indices])
        purged_pred = purged_model.predict(X[test_indices])
        purged_results = {
            "gap": gap,
            "summary": purged_summary,
            "available": True,
            "accuracy": accuracy_score(y[test_indices], purged_pred),
            "macro_f1": f1_score(y[test_indices], purged_pred, average="macro"),
            "report": classification_report(
                y[test_indices],
                purged_pred,
                labels=ordered_labels,
                target_names=target_names,
                digits=4,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y[test_indices],
                purged_pred,
                labels=ordered_labels,
            ).tolist(),
        }

    print("=== DATASET ===")
    print(
        "cells="
        f"{len(ordered_labels)} total_samples={len(X)} input_size={X.shape[1]} "
        f"nodes={len(node_ids)} window={window_seconds:.2f}s step={window_step_seconds:.2f}s"
    )
    for cell_key, count in counts.items():
        print(f"cell {display_cell(cell_key)} samples={count}")
    print()

    print("=== TRAIN FIT ===")
    print(f"train_accuracy={accuracy_score(y, train_pred):.4f}")
    print(f"train_macro_f1={f1_score(y, train_pred, average='macro'):.4f}")
    print()

    print("=== 5-FOLD STRATIFIED CV ===")
    print(f"cv_accuracy={accuracy_score(y, cv_pred):.4f}")
    print(f"cv_macro_f1={f1_score(y, cv_pred, average='macro'):.4f}")
    print("labels=" + ", ".join(target_names))
    print("confusion_matrix_rows=true_cols=pred")
    for row in confusion_matrix(y, cv_pred, labels=ordered_labels).tolist():
        print(row)
    print(
        classification_report(
            y,
            cv_pred,
            labels=ordered_labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    print("=== PURGED TIME SPLIT ===")
    print(f"purge_gap_windows={purged_results['gap']}")
    for cell_key, total, train_count, test_count in purged_results["summary"]:
        print(
            f"cell {display_cell(cell_key)} total={total} "
            f"train={train_count} test={test_count}"
        )
    if not purged_results["available"]:
        print("purged_split=not_enough_data")
        return
    print(f"purged_accuracy={purged_results['accuracy']:.4f}")
    print(f"purged_macro_f1={purged_results['macro_f1']:.4f}")
    print("confusion_matrix_rows=true_cols=pred")
    for row in purged_results["confusion_matrix"]:
        print(row)
    print(purged_results["report"])


if __name__ == "__main__":
    main()
