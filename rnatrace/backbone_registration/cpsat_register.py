from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.spatial.distance import cdist
BP_DIST_MIN = 15.0
BP_DIST_MAX = 22.0
GAP_LONG_STEP_PENALTY_WEIGHT = 0.0
MAX_LOCAL_GAP_REPAIR_CANDIDATES = 32
MAX_LOCAL_GAP_REPAIR_COMBINATIONS = 1024

@dataclass(frozen=True)
class FragmentAssignment:
    atom_idx: int
    residue_num: int
    path_pos: int
    confidence: float
    margin: float
    total_vote: float
    source: str = 'consensus'

@dataclass(frozen=True)
class FragmentAnchor:
    path_start: int
    path_end: int
    residue_start: int
    residue_end: int
    direction: int
    path_positions: Tuple[int, ...]
    atom_indices: Tuple[int, ...]
    residue_nums: Tuple[int, ...]
    mean_confidence: float
    min_confidence: float
    mean_margin: float
    min_margin: float

    def __post_init__(self) -> None:
        n_path = len(self.path_positions)
        if not n_path == len(self.atom_indices) == len(self.residue_nums):
            raise ValueError('FragmentAnchor path_positions, atom_indices, and residue_nums must have equal length')
        if n_path == 0:
            raise ValueError('FragmentAnchor cannot be empty')

    @property
    def length(self) -> int:
        return len(self.atom_indices)

    @property
    def seq_start(self) -> int:
        return min(self.residue_start, self.residue_end)

    @property
    def seq_end(self) -> int:
        return max(self.residue_start, self.residue_end)

    @cached_property
    def _atom_by_residue(self) -> Dict[int, int]:
        return {int(residue): int(atom_idx) for atom_idx, residue in zip(self.atom_indices, self.residue_nums)}

    @cached_property
    def _path_by_residue(self) -> Dict[int, int]:
        return {int(residue): int(path_pos) for path_pos, residue in zip(self.path_positions, self.residue_nums)}

    def atom_for_residue(self, residue_num: int) -> Optional[int]:
        return self._atom_by_residue.get(int(residue_num))

    def path_for_residue(self, residue_num: int) -> Optional[int]:
        return self._path_by_residue.get(int(residue_num))

@dataclass
class RegistrationResult:
    assignments: List[Tuple[int, int]]
    direction: str
    bp_satisfied: int
    bp_total: int
    coverage: float
    unassigned_residues: List[int]
    status: str
    flips: List[Tuple[int, int]] = field(default_factory=list)
    fragment_assignments: List[FragmentAssignment] = field(default_factory=list)
    fragment_anchors: List[FragmentAnchor] = field(default_factory=list)

def _pair_distance_ok(d: float) -> bool:
    return BP_DIST_MIN <= d <= BP_DIST_MAX

def _normalize_bp(base_pairs: Sequence[Tuple[int, int]], seq_len: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for i, j in base_pairs:
        if not (1 <= i <= seq_len and 1 <= j <= seq_len) or i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out

def _bp_satisfaction(res_to_atom: np.ndarray, canonical_bp: Sequence[Tuple[int, int]], dists: np.ndarray, unreliable: Optional[set]=None) -> int:
    sat = 0
    for ri, rj in canonical_bp:
        ai = res_to_atom[ri]
        aj = res_to_atom[rj]
        if ai < 0 or aj < 0:
            continue
        if unreliable is not None and (ai in unreliable or aj in unreliable):
            continue
        if _pair_distance_ok(dists[ai, aj]):
            sat += 1
    return sat

def _build_pos_to_atom(path: Sequence[int], seq_len: int) -> np.ndarray:
    pos_to_atom = -np.ones(seq_len + 2, dtype=np.int64)
    for k, atom in enumerate(path):
        p = k + 1
        if 1 <= p <= seq_len:
            pos_to_atom[p] = atom
    return pos_to_atom

def _apply_flip(path: List[int], i: int, j: int) -> List[int]:
    return path[:i] + path[i:j][::-1] + path[j:]

def _seam_delta(path: Sequence[int], dists: np.ndarray, i: int, j: int) -> float:
    n = len(path)
    before = 0.0
    after = 0.0
    if i > 0:
        before += dists[path[i - 1], path[i]]
        after += dists[path[i - 1], path[j - 1]]
    if j < n:
        before += dists[path[j - 1], path[j]]
        after += dists[path[i], path[j]]
    return float(before - after)

def _build_stacks(canonical_bp: Sequence[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    bp_set = {(a, b) for a, b in canonical_bp}
    stacks: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for a, b in canonical_bp:
        cand = (a + 1, b - 1)
        if cand in bp_set:
            stacks.append(((a, b), cand))
    return stacks

def _count_stacks_satisfied(pos_to_atom: np.ndarray, stacks: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]], dists: np.ndarray) -> int:
    n = 0
    for (a1, b1), (a2, b2) in stacks:
        ai1, aj1 = (pos_to_atom[a1], pos_to_atom[b1])
        ai2, aj2 = (pos_to_atom[a2], pos_to_atom[b2])
        if ai1 < 0 or aj1 < 0 or ai2 < 0 or (aj2 < 0):
            continue
        if _pair_distance_ok(dists[ai1, aj1]) and _pair_distance_ok(dists[ai2, aj2]):
            n += 1
    return n

def _sequence_base_groups(sequence: Optional[str], seq_len: int) -> Optional[np.ndarray]:
    if not sequence:
        return None
    groups = np.zeros(seq_len + 2, dtype=np.int8)
    for i, base in enumerate(str(sequence).upper()[:seq_len], start=1):
        if base in ('A', 'G'):
            groups[i] = 1
        elif base in ('C', 'U', 'T'):
            groups[i] = -1
    return groups

def _atom_base_group_arrays(atom_base_groups: Optional[Sequence[object]], atom_base_group_confidences: Optional[Sequence[float]], n_atoms: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if atom_base_groups is None:
        return (None, None)
    groups = np.zeros(n_atoms, dtype=np.int8)
    for i, value in enumerate(atom_base_groups[:n_atoms]):
        text = str(value).strip().lower()
        if text in ('purine', '3', 'a', 'g', 'r'):
            groups[i] = 1
        elif text in ('pyrimidine', '4', 'c', 'u', 't', 'y'):
            groups[i] = -1
    conf = np.ones(n_atoms, dtype=np.float64)
    if atom_base_group_confidences is not None:
        for i, value in enumerate(atom_base_group_confidences[:n_atoms]):
            conf[i] = max(0.0, min(1.0, float(value)))
    return (groups, conf)

def _base_flip_gain_table(path: Sequence[int], seq_groups: Optional[np.ndarray], atom_groups: Optional[np.ndarray], atom_group_conf: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if seq_groups is None or atom_groups is None or atom_group_conf is None:
        return None
    n = len(path)
    if n == 0:
        return None
    path_atoms = np.asarray(path, dtype=np.int64)
    valid_atoms = (path_atoms >= 0) & (path_atoms < len(atom_groups))
    seq_values = np.zeros(n, dtype=np.int8)
    upto = min(n, len(seq_groups) - 1)
    if upto > 0:
        seq_values[:upto] = seq_groups[1:upto + 1]
    atom_values = np.zeros(n, dtype=np.int8)
    atom_conf = np.zeros(n, dtype=np.float64)
    if np.any(valid_atoms):
        atom_values[valid_atoms] = atom_groups[path_atoms[valid_atoms]]
        atom_conf[valid_atoms] = atom_group_conf[path_atoms[valid_atoms]]
    if not np.any(seq_values) or not np.any(atom_values):
        return None
    compatibility = seq_values[:, None].astype(np.int16) * atom_values[None, :].astype(np.int16)
    score = np.where(compatibility > 0, atom_conf[None, :], np.where(compatibility < 0, -atom_conf[None, :], 0.0))
    diag = np.diag(score)
    gain = np.zeros((n, n + 1), dtype=np.float64)
    for length in range(2, n + 1):
        starts = np.arange(0, n - length + 1)
        ends = starts + length
        gain[starts, ends] = gain[starts + 1, ends - 1] + score[starts, ends - 1] + score[ends - 1, starts] - diag[starts] - diag[ends - 1]
    return gain

def _anchor_flip_gain_table(path: Sequence[int], anchor_seq_by_atom: Optional[np.ndarray], anchor_conf_by_atom: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if anchor_seq_by_atom is None or anchor_conf_by_atom is None:
        return None
    n = len(path)
    if n == 0:
        return None
    path_atoms = np.asarray(path, dtype=np.int64)
    valid_atoms = (path_atoms >= 0) & (path_atoms < len(anchor_seq_by_atom)) & (anchor_seq_by_atom[np.clip(path_atoms, 0, len(anchor_seq_by_atom) - 1)] > 0)
    if not np.any(valid_atoms):
        return None
    anchors = np.zeros(n, dtype=np.int64)
    conf = np.zeros(n, dtype=np.float64)
    anchors[valid_atoms] = anchor_seq_by_atom[path_atoms[valid_atoms]]
    conf[valid_atoms] = anchor_conf_by_atom[path_atoms[valid_atoms]]
    seq_positions = np.arange(1, n + 1, dtype=np.int64)
    score = np.where(anchors[None, :] == seq_positions[:, None], conf[None, :], np.where(valid_atoms[None, :], -conf[None, :], 0.0))
    diag = np.diag(score)
    gain = np.zeros((n, n + 1), dtype=np.float64)
    for length in range(2, n + 1):
        starts = np.arange(0, n - length + 1)
        ends = starts + length
        gain[starts, ends] = gain[starts + 1, ends - 1] + score[starts, ends - 1] + score[ends - 1, starts] - diag[starts] - diag[ends - 1]
    return gain

def _best_flip(path: Sequence[int], seq_len: int, dists: np.ndarray, min_flip_len: int, max_flip_len: int, min_gain: int, bp_partners: Dict[int, List[int]], seam_weight: float=5.0, stacks: Optional[Sequence[Tuple[Tuple[int, int], Tuple[int, int]]]]=None, stack_weight: float=0.3, unreliable: Optional[set]=None, seq_base_groups: Optional[np.ndarray]=None, atom_base_groups: Optional[np.ndarray]=None, atom_base_group_conf: Optional[np.ndarray]=None, base_match_weight: float=0.0, anchor_seq_by_atom: Optional[np.ndarray]=None, anchor_conf_by_atom: Optional[np.ndarray]=None, anchor_match_weight: float=0.0) -> Optional[Tuple[int, int, int]]:
    n = len(path)
    stacks = stacks or ()
    best = None
    base_gain_table = _base_flip_gain_table(path, seq_base_groups, atom_base_groups, atom_base_group_conf) if base_match_weight > 0.0 else None
    anchor_gain_table = _anchor_flip_gain_table(path, anchor_seq_by_atom, anchor_conf_by_atom) if anchor_match_weight > 0.0 else None

    def _pair_ok_after(p: int, q: int, i: int, j: int, p_lo: int, p_hi: int):
        in_p = p_lo <= p <= p_hi
        in_q = p_lo <= q <= p_hi
        ap = path[i + j - 1 - (p - 1)] if in_p else path[p - 1]
        aq = path[i + j - 1 - (q - 1)] if in_q else path[q - 1]
        return _pair_distance_ok(dists[ap, aq])
    for i in range(n - min_flip_len + 1):
        j_cap = min(n, i + max_flip_len)
        for j in range(i + min_flip_len, j_cap + 1):
            p_lo, p_hi = (i + 1, j)
            gain = 0
            seen_keys = set()
            bp_before: Dict[Tuple[int, int], bool] = {}
            bp_after: Dict[Tuple[int, int], bool] = {}
            for p in range(p_lo, p_hi + 1):
                if p - 1 >= n:
                    break
                for partner in bp_partners.get(p, ()):
                    if partner - 1 >= n:
                        continue
                    key = (p, partner) if p < partner else (partner, p)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    a_before_p = path[p - 1]
                    a_before_q = path[partner - 1]
                    in_p = p_lo <= p <= p_hi
                    in_q = p_lo <= partner <= p_hi
                    a_after_p = path[i + j - 1 - (p - 1)] if in_p else a_before_p
                    a_after_q = path[i + j - 1 - (partner - 1)] if in_q else a_before_q
                    if unreliable is not None and (a_before_p in unreliable or a_before_q in unreliable or a_after_p in unreliable or (a_after_q in unreliable)):
                        bp_before[key] = False
                        bp_after[key] = False
                        continue
                    before_ok = _pair_distance_ok(dists[a_before_p, a_before_q])
                    after_ok = _pair_distance_ok(dists[a_after_p, a_after_q])
                    gain += int(after_ok) - int(before_ok)
                    bp_before[key] = before_ok
                    bp_after[key] = after_ok
            base_gain = 0.0
            if base_gain_table is not None:
                base_gain = float(base_gain_table[i, j])
            anchor_gain = 0.0
            if anchor_gain_table is not None:
                anchor_gain = float(anchor_gain_table[i, j])
            stack_gain = 0
            if stacks:
                for (a1, b1), (a2, b2) in stacks:
                    if max(a1, b1, a2, b2) > n:
                        continue
                    touches = p_lo <= a1 <= p_hi or p_lo <= b1 <= p_hi or p_lo <= a2 <= p_hi or (p_lo <= b2 <= p_hi)
                    if not touches:
                        continue
                    k1 = (a1, b1) if a1 < b1 else (b1, a1)
                    k2 = (a2, b2) if a2 < b2 else (b2, a2)

                    def _state(key, a, b, after: bool) -> bool:
                        cache = bp_after if after else bp_before
                        if key in cache:
                            return cache[key]
                        if after:
                            return _pair_ok_after(a, b, i, j, p_lo, p_hi)
                        return _pair_distance_ok(dists[path[a - 1], path[b - 1]])
                    before = _state(k1, a1, b1, False) and _state(k2, a2, b2, False)
                    after = _state(k1, a1, b1, True) and _state(k2, a2, b2, True)
                    stack_gain += int(after) - int(before)
            seam = _seam_delta(path, dists, i, j)
            score = gain + stack_weight * stack_gain + seam_weight * seam + base_match_weight * base_gain + anchor_match_weight * anchor_gain
            if score <= 1e-10:
                continue
            if best is None or score > best[0]:
                best = (score, gain, i, j)
    if best is None:
        return None
    return (best[2], best[3], best[1])

def register_path(path: Sequence[int], positions: np.ndarray, base_pairs: Sequence[Tuple[int, int]], seq_len: int, *, first_residue: int=1, min_flip_len: int=20, max_flip_len: Optional[int]=None, min_gain_per_flip: int=4, max_flips: Optional[int]=None, stack_weight: float=0.3, unreliable_atoms: Optional[Sequence[int]]=None, sequence: Optional[str]=None, atom_base_groups: Optional[Sequence[object]]=None, atom_base_group_confidences: Optional[Sequence[float]]=None, base_match_weight: float=0.0, anchor_assignments: Optional[Sequence[Tuple[int, int]]]=None, anchor_match_weight: float=0.0, verbose: bool=False) -> Optional[RegistrationResult]:
    path = list(path)
    n = len(path)
    if n == 0:
        return None
    positions = np.asarray(positions, dtype=np.float64)
    dists = cdist(positions, positions)
    canonical_bp = _normalize_bp(base_pairs, seq_len)
    bp_partners: Dict[int, List[int]] = defaultdict(list)
    for a, b in canonical_bp:
        bp_partners[a].append(b)
        bp_partners[b].append(a)
    stacks = _build_stacks(canonical_bp)
    unreliable = set(unreliable_atoms) if unreliable_atoms else None
    seq_groups = _sequence_base_groups(sequence, seq_len)
    atom_groups, atom_group_conf = _atom_base_group_arrays(atom_base_groups, atom_base_group_confidences, len(positions))
    anchor_seq_by_atom = None
    anchor_conf_by_atom = None
    if anchor_assignments is not None:
        anchor_seq_by_atom = np.zeros(len(positions), dtype=np.int64)
        anchor_conf_by_atom = np.zeros(len(positions), dtype=np.float64)
        for atom_idx, residue_num in anchor_assignments:
            if 0 <= atom_idx < len(positions):
                seq_pos = int(residue_num) - first_residue + 1
                if 1 <= seq_pos <= seq_len:
                    anchor_seq_by_atom[atom_idx] = seq_pos
                    anchor_conf_by_atom[atom_idx] = 1.0
    flips: List[Tuple[int, int]] = []
    current = list(path)
    seen_states = {tuple(current)}
    pos_to_atom = _build_pos_to_atom(current, seq_len)
    baseline = _bp_satisfaction(pos_to_atom, canonical_bp, dists, unreliable)
    baseline_stacks = _count_stacks_satisfied(pos_to_atom, stacks, dists)
    if max_flip_len is None:
        max_flip_len = n
    if verbose:
        print(f'[flip] start: bp={baseline}/{len(canonical_bp)} stacks={baseline_stacks}/{len(stacks)} min_flip_len={min_flip_len} max_flip_len={max_flip_len} base_weight={base_match_weight} anchor_weight={anchor_match_weight}')
    it = 0
    while max_flips is None or it < max_flips:
        it += 1
        best = _best_flip(current, seq_len, dists, min_flip_len, max_flip_len, min_gain_per_flip, bp_partners, stacks=stacks, stack_weight=stack_weight, unreliable=unreliable, seq_base_groups=seq_groups, atom_base_groups=atom_groups, atom_base_group_conf=atom_group_conf, base_match_weight=base_match_weight, anchor_seq_by_atom=anchor_seq_by_atom, anchor_conf_by_atom=anchor_conf_by_atom, anchor_match_weight=anchor_match_weight)
        if best is None:
            break
        i, j, gain = best
        candidate = _apply_flip(current, i, j)
        state = tuple(candidate)
        if state in seen_states:
            if verbose:
                print(f'[flip] stop: repeated path state after flip [{i},{j})')
            break
        current = candidate
        seen_states.add(state)
        flips.append((i, j))
        baseline += gain
        if verbose:
            pos_to_atom = _build_pos_to_atom(current, seq_len)
            n_stacks = _count_stacks_satisfied(pos_to_atom, stacks, dists)
            print(f'[flip] iter {it + 1}: flip [{i},{j}) len={j - i} gain={gain} bp={baseline}/{len(canonical_bp)} stacks={n_stacks}/{len(stacks)}')
    assignments: List[Tuple[int, int]] = []
    assigned_res: set = set()
    for k, atom in enumerate(current):
        r = first_residue + k
        if k >= seq_len:
            break
        assignments.append((atom, r))
        assigned_res.add(r)
    pos_to_atom = _build_pos_to_atom(current, seq_len)
    bp_sat = _bp_satisfaction(pos_to_atom, canonical_bp, dists, unreliable)
    bp_total = sum((1 for pi, pj in canonical_bp if pos_to_atom[pi] >= 0 and pos_to_atom[pj] >= 0 and (unreliable is None or (pos_to_atom[pi] not in unreliable and pos_to_atom[pj] not in unreliable))))
    coverage = len(assigned_res) / max(1, seq_len)
    unassigned = sorted(set((first_residue + k for k in range(seq_len))) - assigned_res)
    if not flips:
        direction = 'forward'
    else:
        direction = 'mixed'
    return RegistrationResult(assignments=assignments, direction=direction, bp_satisfied=bp_sat, bp_total=bp_total, coverage=coverage, unassigned_residues=unassigned, status=f'flip_greedy(flips={len(flips)})', flips=flips)

def _score_base_group_pairs(residue_nums: Sequence[int], atom_indices: Sequence[int], *, first_residue: int, seq_base_groups: Optional[np.ndarray], atom_base_groups: Optional[np.ndarray], atom_base_group_conf: Optional[np.ndarray]) -> float:
    if seq_base_groups is None or atom_base_groups is None or atom_base_group_conf is None:
        return 0.0
    score = 0.0
    for residue_num, atom_idx in zip(residue_nums, atom_indices):
        seq_pos = int(residue_num) - first_residue + 1
        if seq_pos <= 0 or seq_pos >= len(seq_base_groups):
            continue
        if atom_idx < 0 or atom_idx >= len(atom_base_groups):
            continue
        seq_group = int(seq_base_groups[seq_pos])
        atom_group = int(atom_base_groups[atom_idx])
        if seq_group == 0 or atom_group == 0:
            continue
        conf = float(atom_base_group_conf[atom_idx])
        score += conf if seq_group == atom_group else -conf
    return score

def extract_fragment_anchors(fragment_assignments: Sequence[FragmentAssignment], *, min_length: int=8, trim_ends: int=2) -> List[FragmentAnchor]:
    if min_length <= 0:
        raise ValueError('min_length must be positive')
    if trim_ends < 0:
        raise ValueError('trim_ends must be non-negative')
    ordered = sorted(fragment_assignments, key=lambda item: item.path_pos)
    if not ordered:
        return []
    runs: List[List[FragmentAssignment]] = []
    current: List[FragmentAssignment] = []
    current_direction: Optional[int] = None
    for item in ordered:
        if not current:
            current = [item]
            current_direction = None
            continue
        prev = current[-1]
        residue_delta = item.residue_num - prev.residue_num
        if item.path_pos == prev.path_pos + 1 and abs(residue_delta) == 1:
            step_direction = 1 if residue_delta > 0 else -1
            if current_direction is None:
                current_direction = step_direction
            if step_direction == current_direction:
                current.append(item)
                continue
        runs.append(current)
        current = [item]
        current_direction = None
    if current:
        runs.append(current)
    anchors: List[FragmentAnchor] = []
    for run in runs:
        trimmed = run
        if trim_ends:
            if len(run) <= 2 * trim_ends:
                continue
            trimmed = run[trim_ends:-trim_ends]
        if len(trimmed) < min_length:
            continue
        if len(trimmed) >= 2:
            direction = 1 if trimmed[-1].residue_num > trimmed[0].residue_num else -1
        else:
            direction = 1
        confidences = np.asarray([item.confidence for item in trimmed], dtype=np.float64)
        margins = np.asarray([item.margin for item in trimmed], dtype=np.float64)
        anchors.append(FragmentAnchor(path_start=trimmed[0].path_pos, path_end=trimmed[-1].path_pos + 1, residue_start=trimmed[0].residue_num, residue_end=trimmed[-1].residue_num, direction=direction, path_positions=tuple((item.path_pos for item in trimmed)), atom_indices=tuple((item.atom_idx for item in trimmed)), residue_nums=tuple((item.residue_num for item in trimmed)), mean_confidence=float(np.mean(confidences)), min_confidence=float(np.min(confidences)), mean_margin=float(np.mean(margins)), min_margin=float(np.min(margins))))
    return anchors

def _score_gap_geometry(chain_atoms: Sequence[int], positions: np.ndarray, *, target_step: float, max_step: float) -> Tuple[float, float, float]:
    coords = np.asarray([positions[int(atom)] for atom in chain_atoms], dtype=np.float64)
    if len(coords) < 2:
        return (-float('inf'), float('inf'), float('inf'))
    steps = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    deviations = steps - float(target_step)
    long_excess = np.maximum(0.0, steps - float(max_step))
    score = -float(np.sum(deviations * deviations) + GAP_LONG_STEP_PENALTY_WEIGHT * np.sum(long_excess * long_excess))
    return (score, float(np.max(steps)), float(np.mean(steps)))

def register_path_fragment_consensus(path: Sequence[int], positions: np.ndarray, base_pairs: Sequence[Tuple[int, int]], seq_len: int, *, first_residue: int=1, sequence: Optional[str]=None, atom_base_groups: Optional[Sequence[object]]=None, atom_base_group_confidences: Optional[Sequence[float]]=None, window_size: int=24, stride: int=2, top_k: int=1, min_confidence: float=0.5, min_margin: float=0.3, anchor_min_length: int=8, anchor_trim_ends: int=2, verbose: bool=False) -> Optional[RegistrationResult]:
    path = list(path)
    n_path = len(path)
    if n_path == 0:
        return None
    if window_size <= 0 or stride <= 0 or top_k <= 0:
        raise ValueError('window_size, stride, and top_k must be positive')
    positions = np.asarray(positions, dtype=np.float64)
    dists = cdist(positions, positions)
    seq_groups = _sequence_base_groups(sequence, seq_len)
    atom_groups, atom_group_conf = _atom_base_group_arrays(atom_base_groups, atom_base_group_confidences, len(positions))
    if seq_groups is None or atom_groups is None or atom_group_conf is None:
        return None
    win = min(window_size, n_path, seq_len)
    if win <= 0:
        return None
    starts = list(range(0, max(1, n_path - win + 1), stride))
    tail_start = max(0, n_path - win)
    if starts[-1] != tail_start:
        starts.append(tail_start)
    votes: List[Dict[int, float]] = [defaultdict(float) for _ in range(n_path)]
    n_windows = 0
    for fs in starts:
        length = min(win, n_path - fs)
        if length < max(4, win // 2):
            continue
        candidates: List[Tuple[float, float, int, int, int]] = []
        for direction in (1, -1):
            if direction == 1:
                seq_starts = range(1, seq_len - length + 2)
            else:
                seq_starts = range(length, seq_len + 1)
            for start_res in seq_starts:
                score = 0.0
                known = 0
                for k in range(length):
                    atom_idx = path[fs + k]
                    seq_pos = start_res + direction * k
                    seq_group = int(seq_groups[seq_pos])
                    atom_group = int(atom_groups[atom_idx])
                    if seq_group == 0 or atom_group == 0:
                        continue
                    conf = float(atom_group_conf[atom_idx])
                    score += conf if seq_group == atom_group else -conf
                    known += 1
                if known:
                    candidates.append((score / known, score, start_res, direction, known))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        for norm_score, _score, start_res, direction, _known in candidates[:top_k]:
            weight = max(0.001, norm_score + 1.0)
            for k in range(length):
                seq_pos = start_res + direction * k
                if 1 <= seq_pos <= seq_len:
                    abs_res = first_residue + seq_pos - 1
                    votes[fs + k][abs_res] += weight
        n_windows += 1
    raw_assignments: List[Tuple[int, int, float, float, float, int]] = []
    for path_pos, atom_idx in enumerate(path):
        if not votes[path_pos]:
            continue
        ranked = sorted(votes[path_pos].items(), key=lambda item: item[1], reverse=True)
        total_vote = float(sum((v for _, v in ranked)))
        if total_vote <= 0.0:
            continue
        top_res, top_vote = ranked[0]
        second_vote = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = top_vote / total_vote
        margin = (top_vote - second_vote) / total_vote
        if confidence < min_confidence or margin < min_margin:
            continue
        raw_assignments.append((atom_idx, top_res, confidence, margin, total_vote, path_pos))
    picked_by_res: Dict[int, Tuple[int, int, float, float, float, int]] = {}
    for item in raw_assignments:
        _, res_num, confidence, margin, total_vote, _ = item
        prev = picked_by_res.get(res_num)
        key = (confidence, margin, total_vote)
        if prev is None or key > (prev[2], prev[3], prev[4]):
            picked_by_res[res_num] = item
    picked = sorted(picked_by_res.values(), key=lambda item: item[5])
    fragment_assignments = [FragmentAssignment(atom_idx=atom_idx, residue_num=res_num, confidence=float(confidence), margin=float(margin), total_vote=float(total_vote), path_pos=int(path_pos)) for atom_idx, res_num, confidence, margin, total_vote, path_pos in picked]
    fragment_anchors = extract_fragment_anchors(fragment_assignments, min_length=anchor_min_length, trim_ends=anchor_trim_ends)
    assignments = [(item.atom_idx, item.residue_num) for item in fragment_assignments]
    if not assignments:
        return None
    canonical_bp = _normalize_bp(base_pairs, seq_len)
    res_to_atom_arr = -np.ones(seq_len + 2, dtype=np.int64)
    assigned_seq_res = set()
    for atom_idx, abs_r in assignments:
        seq_r = abs_r - first_residue + 1
        if 1 <= seq_r <= seq_len:
            res_to_atom_arr[seq_r] = atom_idx
            assigned_seq_res.add(seq_r)
    bp_sat = _bp_satisfaction(res_to_atom_arr, canonical_bp, dists)
    bp_total = sum((1 for ri, rj in canonical_bp if res_to_atom_arr[ri] >= 0 and res_to_atom_arr[rj] >= 0))
    unassigned = [first_residue + r - 1 for r in sorted(set(range(1, seq_len + 1)) - assigned_seq_res)]
    coverage = len(assigned_seq_res) / max(1, seq_len)
    if verbose:
        print(f'[fragment] windows={n_windows} assigned={len(assignments)}/{n_path} coverage={coverage:.1%} min_conf={min_confidence:g} min_margin={min_margin:g}')
    return RegistrationResult(assignments=assignments, direction='fragment_consensus', bp_satisfied=bp_sat, bp_total=bp_total, coverage=coverage, unassigned_residues=unassigned, status=f'fragment_consensus(window={win},stride={stride},assigned={len(assignments)})', fragment_assignments=fragment_assignments, fragment_anchors=fragment_anchors)
