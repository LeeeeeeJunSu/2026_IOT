from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Cell:
    x: int
    y: int


def clamp_cell(x: int, y: int, cols: int, rows: int) -> Cell:
    return Cell(
        x=max(0, min(cols - 1, x)),
        y=max(0, min(rows - 1, y)),
    )


def _snake_path(cols: int, rows: int) -> list[Cell]:
    path: list[Cell] = []
    for y in range(rows):
        xs = range(cols) if y % 2 == 0 else range(cols - 1, -1, -1)
        for x in xs:
            path.append(Cell(x=x, y=y))
    return path


def _row_major_path(cols: int, rows: int) -> list[Cell]:
    return [Cell(x=x, y=y) for y in range(rows) for x in range(cols)]


def _column_major_path(cols: int, rows: int) -> list[Cell]:
    return [Cell(x=x, y=y) for x in range(cols) for y in range(rows)]


def _loop_path(cols: int, rows: int) -> list[Cell]:
    if cols == 1 and rows == 1:
        return [Cell(0, 0)]

    path: list[Cell] = []
    top = 0
    bottom = rows - 1
    left = 0
    right = cols - 1

    while left <= right and top <= bottom:
        for x in range(left, right + 1):
            path.append(Cell(x, top))
        for y in range(top + 1, bottom + 1):
            path.append(Cell(right, y))
        if bottom > top:
            for x in range(right - 1, left - 1, -1):
                path.append(Cell(x, bottom))
        if right > left:
            for y in range(bottom - 1, top, -1):
                path.append(Cell(left, y))
        left += 1
        right -= 1
        top += 1
        bottom -= 1

    return _dedupe_cells(path)


def _dedupe_cells(cells: Iterable[Cell]) -> list[Cell]:
    seen: set[tuple[int, int]] = set()
    unique: list[Cell] = []
    for cell in cells:
        key = (cell.x, cell.y)
        if key not in seen:
            seen.add(key)
            unique.append(cell)
    return unique


def build_path(cols: int, rows: int, mode: str) -> list[Cell]:
    mode_name = (mode or "snake").strip().lower()
    if cols < 1 or rows < 1:
        raise ValueError("grid dimensions must be at least 1x1")

    if mode_name in {"snake", "zigzag"}:
        return _snake_path(cols, rows)
    if mode_name in {"row_major", "row-major"}:
        return _row_major_path(cols, rows)
    if mode_name in {"column_major", "column-major"}:
        return _column_major_path(cols, rows)
    if mode_name in {"loop", "perimeter"}:
        return _loop_path(cols, rows)
    return _snake_path(cols, rows)


def rotate_path_to_start(path: list[Cell], start_cell: Cell) -> list[Cell]:
    if not path:
        return []

    try:
        start_index = path.index(start_cell)
    except ValueError:
        return path

    return path[start_index:] + path[:start_index]
