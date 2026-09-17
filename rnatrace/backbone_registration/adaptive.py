from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist
from .cpsat_register import BP_DIST_MIN, BP_DIST_MAX

def _trimmed_anchor_assignments(reg) -> list[tuple[int, int]]:
    assignments: list[tuple[int, int]] = []
    seen_atoms: set[int] = set()
    seen_residues: set[int] = set()

    def add_assignment(atom_idx: int, residue_num: int) -> None:
        atom_idx = int(atom_idx)
        residue_num = int(residue_num)
        if atom_idx in seen_atoms or residue_num in seen_residues:
            return
        assignments.append((atom_idx, residue_num))
        seen_atoms.add(atom_idx)
        seen_residues.add(residue_num)
    for anchor in reg.fragment_anchors:
        for atom_idx, residue_num in zip(anchor.atom_indices, anchor.residue_nums):
            add_assignment(atom_idx, residue_num)
    assignments.sort(key=lambda item: item[1])
    return assignments

def _bp_ratio(reg) -> Optional[float]:
    if reg is None or reg.bp_total <= 0:
        return None
    return float(reg.bp_satisfied) / float(reg.bp_total)

def _anchor_agreement(assignments, anchor_assignments) -> dict:
    assignment_by_atom = {int(atom_idx): int(residue_num) for atom_idx, residue_num in assignments}
    total = 0
    matched = 0
    for atom_idx, residue_num in anchor_assignments:
        assigned = assignment_by_atom.get(int(atom_idx))
        total += 1
        if assigned == int(residue_num):
            matched += 1
    return {'matched': matched, 'total': total, 'ratio': matched / total if total else None, 'pct': round(100.0 * matched / total, 1) if total else None}

def _adaptive_fragment_trigger(baseline_reg, anchor_assignments, *, min_anchor_assignments: int, min_anchor_agreement: float, min_bp_ratio: float, base_group_evidence: Optional[dict]=None) -> tuple[bool, dict]:
    bp_ratio = _bp_ratio(baseline_reg)
    agreement = _anchor_agreement(baseline_reg.assignments if baseline_reg is not None else [], anchor_assignments)
    reasons: list[str] = []
    evidence = base_group_evidence or {}
    n_base_group_known = evidence.get('n_base_group_known')
    n_base_group_total = evidence.get('n_base_group_total')
    if n_base_group_known == 0 and n_base_group_total:
        reasons.append('no_base_group_evidence')
    has_anchor_evidence = len(anchor_assignments) >= min_anchor_assignments
    if not has_anchor_evidence:
        reasons.append('insufficient_fragment_anchor_evidence')
    else:
        anchor_ratio = agreement['ratio']
        if anchor_ratio is not None and anchor_ratio < min_anchor_agreement:
            reasons.append('low_fragment_anchor_agreement')
    if bp_ratio is not None and bp_ratio < min_bp_ratio:
        reasons.append('low_bp_satisfaction')
    should_try = has_anchor_evidence and any((reason in reasons for reason in ('low_fragment_anchor_agreement', 'low_bp_satisfaction')))
    return (should_try, {'bp_ratio': bp_ratio, 'anchor_agreement': agreement, 'n_anchor_assignments': len(anchor_assignments), 'base_group_evidence': evidence, 'min_anchor_assignments': min_anchor_assignments, 'min_anchor_agreement': min_anchor_agreement, 'min_bp_ratio': min_bp_ratio, 'reasons': reasons, 'should_try_fragment': should_try})

def _adaptive_candidate_summary(name: str, reg, anchor_assignments, *, weight: Optional[float]=None) -> dict:
    agreement = _anchor_agreement(reg.assignments if reg is not None else [], anchor_assignments)
    bp_ratio = _bp_ratio(reg)
    return {'name': name, 'weight': weight, 'status': reg.status if reg is not None else 'register_failed', 'bp_satisfied': reg.bp_satisfied if reg is not None else None, 'bp_total': reg.bp_total if reg is not None else None, 'bp_ratio': bp_ratio, 'anchor_agreement': agreement, 'n_flips': len(reg.flips) if reg is not None else None}

def _select_adaptive_fragment_candidate(baseline_summary: dict, candidate_summaries: list[dict], *, min_anchor_gain: float, max_bp_loss: float) -> Optional[dict]:
    baseline_anchor = baseline_summary['anchor_agreement']['ratio']
    baseline_bp = baseline_summary['bp_ratio']
    eligible: list[tuple[tuple[float, float, float], dict]] = []
    for candidate in candidate_summaries:
        candidate_anchor = candidate['anchor_agreement']['ratio']
        candidate_bp = candidate['bp_ratio']
        if candidate_anchor is None:
            continue
        if baseline_anchor is not None:
            if candidate_anchor + 1e-12 < baseline_anchor + min_anchor_gain:
                continue
        if baseline_bp is not None and candidate_bp is not None:
            if candidate_bp + 1e-12 < baseline_bp - max_bp_loss:
                continue
        anchor_component = candidate_anchor
        anchor_support = float(candidate.get('n_anchor_assignments_used') or candidate['anchor_agreement'].get('total') or 0)
        bp_component = candidate_bp if candidate_bp is not None else 0.0
        score = (anchor_component, anchor_support, bp_component, -float(candidate['n_flips'] or 0))
        eligible.append((score, candidate))
    if not eligible:
        return None
    eligible.sort(key=lambda item: item[0], reverse=True)
    selected = dict(eligible[0][1])
    selected['min_anchor_gain'] = min_anchor_gain
    selected['max_bp_loss'] = max_bp_loss
    return selected

def _bp_satisfied_for_oriented_path(path, positions: np.ndarray, base_pairs, *, first_residue: int, seq_len: int) -> int:
    if not base_pairs:
        return 0
    rel_ok = all((1 <= ri <= seq_len and 1 <= rj <= seq_len for ri, rj in base_pairs))
    auth_last = first_residue + seq_len - 1
    auth_ok = all((first_residue <= ri <= auth_last and first_residue <= rj <= auth_last for ri, rj in base_pairs))
    residue0 = first_residue if auth_ok and (not rel_ok) else 1
    residue_to_atom = {residue0 + k: atom for k, atom in enumerate(path)}
    dists = cdist(positions, positions)
    count = 0
    for ri, rj in base_pairs:
        ai = residue_to_atom.get(ri)
        aj = residue_to_atom.get(rj)
        if ai is None or aj is None:
            continue
        d = dists[ai, aj]
        if BP_DIST_MIN <= d <= BP_DIST_MAX:
            count += 1
    return count
