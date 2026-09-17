from __future__ import annotations
from typing import List, Sequence
import numpy as np
from scipy.spatial.distance import cdist
import elkai
COST_SCALE = 100

def _open_tsp_to_path(tour: Sequence[int], dummy_idx: int) -> List[int]:
    if tour and tour[0] == tour[-1]:
        tour = tour[:-1]
    if dummy_idx not in tour:
        raise ValueError('Dummy node not found in tour.')
    k = tour.index(dummy_idx)
    rotated = list(tour[k + 1:]) + list(tour[:k])
    return rotated

def _solve_open_tsp(cost_matrix: np.ndarray, runs: int=10) -> List[int]:
    n = cost_matrix.shape[0]
    if n <= 1:
        return list(range(n))
    big = np.zeros((n + 1, n + 1), dtype=np.int64)
    big[:n, :n] = cost_matrix
    dist_mat = elkai.DistanceMatrix(big.tolist())
    tour = dist_mat.solve_tsp(runs=runs)
    return _open_tsp_to_path(tour, dummy_idx=n)

def _integerize(distances: np.ndarray, scale: int=COST_SCALE) -> np.ndarray:
    sym = 0.5 * (distances + distances.T)
    np.fill_diagonal(sym, 0.0)
    integer = np.rint(sym * scale).astype(np.int64)
    np.fill_diagonal(integer, 0)
    return integer

def lkh_backbone_trace(positions: np.ndarray, *, runs: int=10, cost_scale: int=COST_SCALE) -> List[int]:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f'Expected (N, 3) positions, got {positions.shape}')
    dists = cdist(positions, positions)
    cost = _integerize(dists, scale=cost_scale)
    return _solve_open_tsp(cost, runs=runs)
