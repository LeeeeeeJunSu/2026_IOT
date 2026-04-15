from __future__ import annotations

import json
import pickle
from pathlib import Path


def load_fingerprint_store(path: str | Path) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        return {"cells": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_fingerprint_store(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_pickle_store(path: str | Path) -> object | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle_store(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def remove_store(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
