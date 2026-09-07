from __future__ import annotations
from typing import Sequence
import numpy as np

def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-08:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm

def contiguous_runs(values: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for value in sorted((int(item) for item in values)):
        if current and value != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(value)
    if current:
        runs.append(current)
    return runs

def _derive_run_positions(residue_positions: dict[int, np.ndarray], run: Sequence[int], *, first_residue: int, last_residue: int, target_step: float) -> list[tuple[int, np.ndarray, str]]:
    start = int(run[0])
    end = int(run[-1])
    n_missing = end - start + 1
    left = start - 1
    while left >= first_residue and left not in residue_positions:
        left -= 1
    right = end + 1
    while right <= last_residue and right not in residue_positions:
        right += 1
    if left >= first_residue and right <= last_residue:
        left_pos = residue_positions[left]
        right_pos = residue_positions[right]
        return [(residue, (1.0 - frac) * left_pos + frac * right_pos, 'interpolate') for offset, residue in enumerate(range(start, end + 1), start=1) for frac in [offset / (n_missing + 1)]]
    if right <= last_residue:
        if right + 1 in residue_positions:
            direction = _unit(residue_positions[right] - residue_positions[right + 1])
        elif right + 2 in residue_positions:
            direction = _unit(residue_positions[right] - residue_positions[right + 2])
        else:
            direction = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        derived = []
        for offset, residue in enumerate(range(end, start - 1, -1), start=1):
            derived.append((residue, residue_positions[right] + direction * target_step * offset, 'extrapolate_5p'))
        return list(reversed(derived))
    if left >= first_residue:
        if left - 1 in residue_positions:
            direction = _unit(residue_positions[left] - residue_positions[left - 1])
        elif left - 2 in residue_positions:
            direction = _unit(residue_positions[left] - residue_positions[left - 2])
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return [(residue, residue_positions[left] + direction * target_step * offset, 'extrapolate_3p') for offset, residue in enumerate(range(start, end + 1), start=1)]
    return []
