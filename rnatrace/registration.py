from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from .backbone_registration.adaptive import (
    _adaptive_candidate_summary,
    _adaptive_fragment_trigger,
    _bp_satisfied_for_oriented_path,
    _select_adaptive_fragment_candidate,
    _trimmed_anchor_assignments,
)
from .backbone_registration.cpsat_register import (
    MAX_LOCAL_GAP_REPAIR_CANDIDATES,
    MAX_LOCAL_GAP_REPAIR_COMBINATIONS,
    _atom_base_group_arrays,
    _score_base_group_pairs,
    _score_gap_geometry,
    _sequence_base_groups,
    register_path,
    register_path_fragment_consensus,
)
from .backbone_registration.derived_p_refill import _derive_run_positions, contiguous_runs
from .backbone_registration.lkh_trace import lkh_backbone_trace
from .backbone_registration.map_features import MapGridTransform, _components_for_labels, _choose_component, feature_confidence


def _base_evidence(positions, labels, origin, voxel_size):
    groups, confidences = [], []
    shape = np.asarray(labels.shape)
    spacing = voxel_size[::-1]
    for position in positions:
        center = ((position - origin) / voxel_size)[::-1]
        lower = np.maximum(0, np.floor(center - 12.0 / spacing).astype(int))
        upper = np.minimum(shape, np.ceil(center + 12.0 / spacing).astype(int) + 1)
        if np.any(lower >= upper):
            groups.append('unknown')
            confidences.append(0.0)
            continue
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        indices = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
        distance2 = sum(((index - center[axis]) * spacing[axis]) ** 2 for axis, index in enumerate(indices))
        window = labels[slices]
        mask = distance2 <= 144.0
        purine = int(np.count_nonzero((window == 3) & mask))
        pyrimidine = int(np.count_nonzero((window == 4) & mask))
        total = purine + pyrimidine
        known = total >= 5 and purine != pyrimidine
        groups.append(('purine' if purine > pyrimidine else 'pyrimidine') if known else 'unknown')
        confidences.append(abs(purine - pyrimidine) / total if known else 0.0)
    return groups, confidences


def _atom_record(atom, residue, positions, sequence):
    return {'atom_idx': int(atom), 'residue_num': int(residue), 'base': sequence[residue - 1], 'position': positions[atom].tolist(), 'source': 'observed'}


def _gap_candidates(anchors, path, seq_len):
    ordered = sorted(anchors, key=lambda anchor: anchor.seq_start)
    occupied = {int(p) for anchor in ordered for p in anchor.path_positions}
    for left, right in zip(ordered, ordered[1:]):
        start, end = left.seq_end + 1, right.seq_start - 1
        if start > end or left.direction != right.direction:
            continue
        a = left.path_for_residue(left.seq_end)
        b = right.path_for_residue(right.seq_start)
        direction = left.direction
        if (b - a) * direction <= 0:
            continue
        candidates = list(range(a + direction, b, direction))
        if any(p in occupied for p in candidates):
            continue
        yield list(range(start, end + 1)), [path[p] for p in candidates]
    if not ordered:
        return
    first, last = ordered[0], ordered[-1]
    if first.seq_start > 1:
        endpoint = first.path_for_residue(first.seq_start)
        if first.direction == 1:
            candidates = list(range(endpoint))
        else:
            candidates = list(range(len(path) - 1, endpoint, -1))
        if not any(p in occupied for p in candidates):
            yield list(range(1, first.seq_start)), [path[p] for p in candidates]
    if last.seq_end < seq_len:
        endpoint = last.path_for_residue(last.seq_end)
        if last.direction == 1:
            candidates = list(range(endpoint + 1, len(path)))
        else:
            candidates = list(range(endpoint - 1, -1, -1))
        if not any(p in occupied for p in candidates):
            yield list(range(last.seq_end + 1, seq_len + 1)), [path[p] for p in candidates]


def _repair_gap(residues, candidates, fixed, positions, sequence, seq_groups, atom_groups, confidences):
    n_residues, n_candidates = len(residues), len(candidates)
    count = comb(max(n_residues, n_candidates), min(n_residues, n_candidates))
    report = {'residue_start': residues[0], 'residue_end': residues[-1], 'n_candidates': n_candidates, 'combinations': count}
    if max(n_candidates, n_residues) > MAX_LOCAL_GAP_REPAIR_CANDIDATES or count > MAX_LOCAL_GAP_REPAIR_COMBINATIONS:
        return None, dict(report, status='skipped_search_limit')
    options = combinations(candidates, n_residues) if n_candidates >= n_residues else combinations(residues, n_candidates)
    fixed_positions = {r: np.asarray(record['position']) for r, record in fixed.items()}
    best, best_score = None, -float('inf')
    for option in options:
        selected_atoms = list(option) if n_candidates >= n_residues else candidates
        selected_residues = residues if n_candidates >= n_residues else list(option)
        records = {r: _atom_record(a, r, positions, sequence) for a, r in zip(selected_atoms, selected_residues)}
        residue_positions = dict(fixed_positions)
        residue_positions.update({r: np.asarray(record['position']) for r, record in records.items()})
        missing = [r for r in residues if r not in records]
        for run in contiguous_runs(missing):
            derived = _derive_run_positions(residue_positions, run, first_residue=1, last_residue=len(sequence), target_step=6.0)
            for residue, coordinate, method in derived:
                records[residue] = {'residue_num': residue, 'base': sequence[residue - 1], 'position': coordinate.tolist(), 'source': 'derived_p_refill', 'derive_method': method}
                residue_positions[residue] = coordinate
        if len(records) != n_residues:
            continue
        chain = ([residues[0] - 1] if residues[0] - 1 in fixed else []) + residues + ([residues[-1] + 1] if residues[-1] + 1 in fixed else [])
        coordinates = np.asarray([residue_positions[r] for r in chain])
        geometry, _, _ = _score_gap_geometry(range(len(coordinates)), coordinates, target_step=6.0, max_step=float('inf'))
        base = _score_base_group_pairs(selected_residues, selected_atoms, first_residue=1, seq_base_groups=seq_groups, atom_base_groups=atom_groups, atom_base_group_conf=confidences)
        score = geometry + 2.0 * base
        if score > best_score:
            best, best_score = records, score
    return best, dict(report, status='repaired' if best is not None else 'unresolved', score=best_score if best is not None else None)


def _splice_fragments(registration, fragments, path, positions, sequence, groups, confidences):
    anchor_pairs = _trimmed_anchor_assignments(fragments)
    fixed = {r: _atom_record(a, r, positions, sequence) for a, r in anchor_pairs}
    seq_groups = _sequence_base_groups(sequence, len(sequence))
    atom_groups, confidence_array = _atom_base_group_arrays(groups, confidences, len(positions))
    report = []
    consumed = set()
    for residues, candidates in _gap_candidates(fragments.fragment_anchors, path, len(sequence)):
        used = {record['atom_idx'] for record in fixed.values() if 'atom_idx' in record}
        candidates = [a for a in candidates if a not in used]
        repaired, gap_report = _repair_gap(residues, candidates, fixed, positions, sequence, seq_groups, atom_groups, confidence_array)
        report.append(gap_report)
        if repaired is not None:
            fixed.update(repaired)
            consumed.update(candidates)
    used = {record['atom_idx'] for record in fixed.values() if 'atom_idx' in record}
    for atom, residue in registration.assignments:
        if 1 <= residue <= len(sequence) and residue not in fixed and atom not in used and atom not in consumed:
            fixed[residue] = _atom_record(atom, residue, positions, sequence)
            used.add(atom)
    next_idx = len(positions)
    for record in fixed.values():
        if 'atom_idx' not in record:
            record['atom_idx'] = next_idx
            next_idx += 1
    return [fixed[r] for r in sorted(fixed)], report


def _attach_features(atoms, labels, origin, voxel_size):
    grid = MapGridTransform(tuple(origin), voxel_size, labels.shape)
    for atom in atoms:
        position = np.asarray(atom['position'])
        components = {}
        for name, label, radius in [('sugar', 2, 6.0), ('purine', 3, 10.0), ('pyrimidine', 4, 10.0)]:
            components[name] = _components_for_labels(labels, position, label_values={label}, label_name=name, apix=1.0, search_radius_a=radius, min_voxels=4, map_grid=grid)
        sugar = _choose_component(components['sugar'], target_distance_a=3.8)
        base = _choose_component(components['purine'] + components['pyrimidine'], target_distance_a=8.0)
        atom['sugar_anchor'] = sugar.to_dict() if sugar is not None else None
        atom['base_anchor'] = base.to_dict() if base is not None else None
        atom['map_feature_confidence'] = feature_confidence({'sugar': atom['sugar_anchor'], 'base': atom['base_anchor']})


def register(positions: np.ndarray, sequence: str, pairs: list[tuple[int, int]], labels: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray, output: Path) -> Path:
    positions = np.asarray(positions, dtype=float)
    origin = np.asarray(origin, dtype=float)
    voxel_size = np.asarray(voxel_size, dtype=float)
    labels = np.asarray(labels)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0 or not np.isfinite(positions).all():
        raise ValueError('Phosphorus positions must be a nonempty finite (N, 3) XYZ array')
    if not sequence or set(sequence) - set('ACGU'):
        raise ValueError('Sequence must contain only uppercase A, C, G and U')
    if origin.shape != (3,) or not np.isfinite(origin).all() or voxel_size.shape != (3,) or not np.isfinite(voxel_size).all() or np.any(voxel_size <= 0):
        raise ValueError('Origin and positive voxel size must be finite XYZ vectors')
    if labels.ndim != 3 or min(labels.shape) == 0 or not np.isin(labels, [0, 1, 2, 3, 4]).all():
        raise ValueError('Labels must be a nonempty ZYX array with values 0 through 4')
    if any(not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)) or not 0 <= i < j < len(sequence) for i, j in pairs):
        raise ValueError('Base pairs must be ordered zero-based sequence index pairs')
    base_pairs = sorted({(int(i) + 1, int(j) + 1) for i, j in pairs})
    groups, confidences = _base_evidence(positions, labels, origin, voxel_size)
    path = lkh_backbone_trace(positions, runs=20)
    orientation = {'first_residue': 1, 'seq_len': len(sequence)}
    forward = _bp_satisfied_for_oriented_path(path, positions, base_pairs, **orientation)
    reverse = _bp_satisfied_for_oriented_path(path[::-1], positions, base_pairs, **orientation)
    if reverse > forward:
        path.reverse()
    arguments = dict(first_residue=1, sequence=sequence, atom_base_groups=groups, atom_base_group_confidences=confidences)
    baseline = register_path(path, positions, base_pairs, len(sequence), base_match_weight=2.0, **arguments)
    fragments = register_path_fragment_consensus(path, positions, base_pairs, len(sequence), **arguments)
    anchors = _trimmed_anchor_assignments(fragments) if fragments is not None else []
    trigger, trigger_report = _adaptive_fragment_trigger(baseline, anchors, min_anchor_assignments=24, min_anchor_agreement=0.70, min_bp_ratio=0.65)
    selected, selected_report = baseline, None
    if trigger:
        candidate = register_path(path, positions, base_pairs, len(sequence), base_match_weight=2.0, anchor_assignments=anchors, anchor_match_weight=3.0, **arguments)
        baseline_summary = _adaptive_candidate_summary('baseline', baseline, anchors)
        candidate_summary = _adaptive_candidate_summary('anchor', candidate, anchors, weight=3.0)
        selected_report = _select_adaptive_fragment_candidate(baseline_summary, [candidate_summary], min_anchor_gain=0.10, max_bp_loss=0.15)
        if selected_report is not None:
            selected = candidate
    repairs = []
    if selected_report is not None:
        atoms, repairs = _splice_fragments(selected, fragments, path, positions, sequence, groups, confidences)
    else:
        atoms = [_atom_record(a, r, positions, sequence) for a, r in selected.assignments if 1 <= r <= len(sequence)]
    _attach_features(atoms, labels, origin, voxel_size)
    covered = {record['residue_num'] for record in atoms}
    metadata = {'sequence_length': len(sequence), 'observed_sites': len(positions), 'lkh_runs': 20, 'orientation': 'reverse' if reverse > forward else 'forward', 'selected': 'anchor' if selected_report is not None else 'baseline', 'trigger': trigger_report, 'selection': selected_report, 'gap_repairs': repairs, 'unassigned_residues': sorted(set(range(1, len(sequence) + 1)) - covered), 'unused_observed_sites': sorted(set(range(len(positions))) - {record['atom_idx'] for record in atoms}), 'base_pairs': base_pairs}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'p_atoms': atoms, 'sequence': sequence, 'registration': metadata}, indent=2) + '\n')
    return output
