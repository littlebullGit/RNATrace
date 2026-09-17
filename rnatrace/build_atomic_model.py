import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
BACKBONE_ATOMS = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]
BASE_ATOMS = {
    "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
}
ALL_ATOMS = {base: BACKBONE_ATOMS + atoms for base, atoms in BASE_ATOMS.items()}
ATOM_ELEMENTS = {
    "P": "P",
    "OP1": "O",
    "OP2": "O",
    "O5'": "O",
    "C5'": "C",
    "C4'": "C",
    "O4'": "O",
    "C3'": "C",
    "O3'": "O",
    "C2'": "C",
    "O2'": "O",
    "C1'": "C",
    "N9": "N",
    "C8": "C",
    "N7": "N",
    "C5": "C",
    "C6": "C",
    "N6": "N",
    "O6": "O",
    "N1": "N",
    "C2": "C",
    "N2": "N",
    "N3": "N",
    "C4": "C",
    "O2": "O",
    "N4": "N",
    "O4": "O",
}


@dataclass
class Atom:
    name: str
    position: np.ndarray
    element: str
    occupancy: float = 1.0
    b_factor: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position.tolist(),
            "element": self.element,
            "occupancy": self.occupancy,
            "b_factor": self.b_factor,
        }


@dataclass
class Residue:
    residue_num: int
    base: str
    atoms: List[Atom] = field(default_factory=list)
    chain_id: str = "A"

    def get_atom(self, name: str) -> Optional[Atom]:
        for atom in self.atoms:
            if atom.name == name:
                return atom
        return None

    def get_p_position(self) -> Optional[np.ndarray]:
        p = self.get_atom("P")
        return p.position if p else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residue_num": self.residue_num,
            "base": self.base,
            "chain_id": self.chain_id,
            "atoms": [a.to_dict() for a in self.atoms],
        }


@dataclass
class NucleotideTemplate:
    base: str
    atoms: Dict[str, np.ndarray]

    def get_p_position(self) -> np.ndarray:
        return self.atoms["P"]

    def get_backbone_atoms(self) -> Dict[str, np.ndarray]:
        return {name: self.atoms[name] for name in BACKBONE_ATOMS if name in self.atoms}

    def get_base_atoms(self) -> Dict[str, np.ndarray]:
        return {
            name: self.atoms[name]
            for name in BASE_ATOMS.get(self.base, [])
            if name in self.atoms
        }


@dataclass
class AtomicModel:
    residues: List[Residue] = field(default_factory=list)
    chain_id: str = "A"

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    @property
    def n_atoms(self) -> int:
        return sum((len(r.atoms) for r in self.residues))

    def get_all_positions(self) -> np.ndarray:
        positions = []
        for res in self.residues:
            for atom in res.atoms:
                positions.append(atom.position)
        return np.array(positions) if positions else np.empty((0, 3))

    def get_p_positions(self) -> np.ndarray:
        positions = []
        for res in self.residues:
            p = res.get_p_position()
            if p is not None:
                positions.append(p)
        return np.array(positions) if positions else np.empty((0, 3))


def get_default_templates() -> Dict[str, NucleotideTemplate]:
    logger.info("Using built-in idealized templates (A-form RNA geometry)")
    return _get_builtin_templates()


def _get_builtin_templates() -> Dict[str, NucleotideTemplate]:
    templates = {}
    templates["A"] = NucleotideTemplate(
        base="A",
        atoms={
            "P": np.array([0.0, 0.0, 0.0]),
            "OP1": np.array([0.16, 0.83, -1.22]),
            "OP2": np.array([-1.17, -0.9, 0.01]),
            "O5'": np.array([1.33, -0.84, 0.25]),
            "C5'": np.array([2.56, 0.0, 0.3]),
            "C4'": np.array([3.66, -0.99, 0.63]),
            "O4'": np.array([3.43, -1.55, 1.95]),
            "C3'": np.array([3.71, -2.16, -0.33]),
            "O3'": np.array([4.33, -1.88, -1.58]),
            "C2'": np.array([4.5, -3.2, 0.48]),
            "O2'": np.array([5.9, -3.03, 0.6]),
            "C1'": np.array([3.81, -2.93, 1.82]),
            "N9": np.array([2.58, -3.72, 2.02]),
            "C8": np.array([1.31, -3.23, 2.26]),
            "N7": np.array([0.37, -4.11, 2.4]),
            "C5": np.array([1.02, -5.34, 2.25]),
            "C6": np.array([0.51, -6.65, 2.28]),
            "N6": np.array([-0.79, -6.92, 2.47]),
            "N1": np.array([1.37, -7.69, 2.07]),
            "C2": np.array([2.65, -7.45, 1.88]),
            "N3": np.array([3.22, -6.26, 1.84]),
            "C4": np.array([2.35, -5.23, 2.04]),
        },
    )
    templates["G"] = NucleotideTemplate(
        base="G",
        atoms={
            **{
                k: v.copy()
                for k, v in templates["A"].atoms.items()
                if k in BACKBONE_ATOMS
            },
            "N9": np.array([2.58, -3.72, 2.02]),
            "C8": np.array([1.31, -3.23, 2.26]),
            "N7": np.array([0.37, -4.11, 2.4]),
            "C5": np.array([1.02, -5.34, 2.25]),
            "C6": np.array([0.51, -6.65, 2.28]),
            "O6": np.array([-0.69, -6.92, 2.47]),
            "N1": np.array([1.37, -7.69, 2.07]),
            "C2": np.array([2.65, -7.45, 1.88]),
            "N2": np.array([3.35, -8.58, 1.68]),
            "N3": np.array([3.22, -6.26, 1.84]),
            "C4": np.array([2.35, -5.23, 2.04]),
        },
    )
    templates["C"] = NucleotideTemplate(
        base="C",
        atoms={
            **{
                k: v.copy()
                for k, v in templates["A"].atoms.items()
                if k in BACKBONE_ATOMS
            },
            "N1": np.array([2.58, -3.72, 2.02]),
            "C2": np.array([1.56, -4.69, 2.09]),
            "O2": np.array([0.39, -4.37, 2.24]),
            "N3": np.array([1.91, -5.97, 1.95]),
            "C4": np.array([3.18, -6.33, 1.74]),
            "N4": np.array([3.45, -7.62, 1.61]),
            "C5": np.array([4.21, -5.33, 1.68]),
            "C6": np.array([3.87, -4.05, 1.82]),
        },
    )
    templates["U"] = NucleotideTemplate(
        base="U",
        atoms={
            **{
                k: v.copy()
                for k, v in templates["A"].atoms.items()
                if k in BACKBONE_ATOMS
            },
            "N1": np.array([2.58, -3.72, 2.02]),
            "C2": np.array([1.56, -4.69, 2.09]),
            "O2": np.array([0.39, -4.37, 2.24]),
            "N3": np.array([1.91, -5.97, 1.95]),
            "C4": np.array([3.18, -6.33, 1.74]),
            "O4": np.array([3.45, -7.52, 1.61]),
            "C5": np.array([4.21, -5.33, 1.68]),
            "C6": np.array([3.87, -4.05, 1.82]),
        },
    )
    return templates


def rotate_around_axis(
    points: Dict[str, np.ndarray], axis: np.ndarray, center: np.ndarray, angle: float
) -> Dict[str, np.ndarray]:
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated = {}
    for name, pos in points.items():
        v = pos - center
        v_rot = (
            v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)
        )
        rotated[name] = v_rot + center
    return rotated


def optimize_base_orientation(
    transformed_atoms: Dict[str, np.ndarray],
    p_position: np.ndarray,
    x_axis: np.ndarray,
    p_partner: np.ndarray,
    base_type: str,
) -> Dict[str, np.ndarray]:
    if base_type in ["A", "G"]:
        key_atoms = ["N1", "N6"] if base_type == "A" else ["N1", "O6"]
    else:
        key_atoms = ["N3", "N4"] if base_type == "C" else ["N3", "O4"]
    existing_key = [a for a in key_atoms if a in transformed_atoms]
    if not existing_key:
        base_atoms = ["N9", "N1", "C2", "C4", "C5", "C6"]
        existing_key = [a for a in base_atoms if a in transformed_atoms]
    if not existing_key:
        return transformed_atoms
    partner_dir = p_partner - p_position
    partner_dir = partner_dir / (np.linalg.norm(partner_dir) + 1e-08)
    best_angle = 0.0
    best_score = -float("inf")
    for angle_deg in range(0, 360, 15):
        angle = np.radians(angle_deg)
        rotated = rotate_around_axis(transformed_atoms, x_axis, p_position, angle)
        score = 0.0
        for atom_name in existing_key:
            if atom_name in rotated:
                atom_dir = rotated[atom_name] - p_position
                atom_dir = atom_dir / (np.linalg.norm(atom_dir) + 1e-08)
                score += np.dot(atom_dir, partner_dir)
        if score > best_score:
            best_score = score
            best_angle = angle
    if best_angle != 0.0:
        return rotate_around_axis(transformed_atoms, x_axis, p_position, best_angle)
    return transformed_atoms


DEFAULT_MAP_FEATURE_MIN_CONFIDENCE = 0.75


def get_map_feature_orientation_hint(
    atom_data: Dict[str, Any],
    p_position: np.ndarray,
    min_confidence: float = DEFAULT_MAP_FEATURE_MIN_CONFIDENCE,
) -> Optional[np.ndarray]:
    confidence = atom_data.get("map_feature_confidence")
    if confidence is not None and float(confidence) < min_confidence:
        return None
    vectors = []
    weights = []
    base_anchor = atom_data.get("base_anchor") or {}
    sugar_anchor = atom_data.get("sugar_anchor") or {}
    if base_anchor.get("centroid") is not None:
        vectors.append(np.array(base_anchor["centroid"], dtype=np.float32) - p_position)
        weights.append(0.7)
    if sugar_anchor.get("centroid") is not None:
        vectors.append(
            np.array(sugar_anchor["centroid"], dtype=np.float32) - p_position
        )
        weights.append(0.3)
    if not vectors:
        return None
    direction = np.zeros(3, dtype=np.float32)
    for vec, weight in zip(vectors, weights):
        norm = np.linalg.norm(vec)
        if norm > 1.0:
            direction += weight * (vec / norm)
    norm = np.linalg.norm(direction)
    if norm <= 1e-06:
        return None
    return direction / norm


def compute_local_frame(
    p_prev: Optional[np.ndarray],
    p_curr: np.ndarray,
    p_next: Optional[np.ndarray],
    p_partner: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if p_prev is not None and p_next is not None:
        backbone_dir = p_next - p_prev
    elif p_next is not None:
        backbone_dir = p_next - p_curr
    elif p_prev is not None:
        backbone_dir = p_curr - p_prev
    else:
        backbone_dir = np.array([1.0, 0.0, 0.0])
    x_axis = backbone_dir / (np.linalg.norm(backbone_dir) + 1e-08)
    y_axis = None
    if p_partner is not None:
        partner_dir = p_partner - p_curr
        partner_dist = np.linalg.norm(partner_dir)
        if partner_dist > 1.0:
            partner_dir = partner_dir / partner_dist
            perp_component = partner_dir - np.dot(partner_dir, x_axis) * x_axis
            perp_magnitude = np.linalg.norm(perp_component)
            if perp_magnitude > 0.3:
                y_axis = perp_component / perp_magnitude
                logger.debug(
                    f"  Using partner perpendicular component (dist={partner_dist:.1f} Å, perp={perp_magnitude:.2f})"
                )
            else:
                logger.debug(
                    f"  Partner nearly parallel to backbone (perp={perp_magnitude:.2f}), using global up"
                )
    if y_axis is None:
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(x_axis, up)) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(x_axis, up)) > 0.9:
                up = np.array([1.0, 0.0, 0.0])
        y_axis = up - np.dot(up, x_axis) * x_axis
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-08)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-08)
    return (x_axis, y_axis, z_axis)


def load_2d_structure(
    structure_2d_path: Path, canonical_only: bool = True
) -> Dict[int, int]:
    with open(structure_2d_path) as f:
        data = json.load(f)
    partner_map = {}
    base_pairs = data.get("base_pairs", [])
    index_base = int(data.get("index_base", 0))
    if index_base not in (0, 1):
        raise ValueError(
            f"Unsupported 2D structure index_base={index_base} in {structure_2d_path}; expected 0 or 1"
        )
    first_residue = int(data.get("first_residue", 1))
    residue_offset = first_residue - index_base
    n_total = len(base_pairs)
    n_used = 0
    for bp in base_pairs:
        if canonical_only and (not bp.get("canonical", False)):
            continue
        i = int(bp["i"]) + residue_offset
        j = int(bp["j"]) + residue_offset
        partner_map[i] = j
        partner_map[j] = i
        n_used += 1
    logger.info(
        f"Loaded {n_total} base pairs from 2D structure (index_base={index_base}, first_residue={first_residue})"
    )
    if canonical_only:
        logger.info(
            f"  Using {n_used} canonical pairs (filtered out {n_total - n_used} non-canonical)"
        )
    logger.info(f"  {len(partner_map)} residues have pairing information")
    return partner_map


def compute_template_frame(
    template: NucleotideTemplate,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_pos = template.atoms["P"]
    if "O3'" in template.atoms:
        o3_pos = template.atoms["O3'"]
        x_axis = o3_pos - p_pos
    else:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-08)
    if "C1'" in template.atoms:
        c1_pos = template.atoms["C1'"]
        y_dir = c1_pos - p_pos
    else:
        y_dir = np.array([0.0, 1.0, 0.0])
    y_axis = y_dir - np.dot(y_dir, x_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-08)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-08)
    return (x_axis, y_axis, z_axis)


def compute_rotation_matrix(
    src_frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    dst_frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    src_x, src_y, src_z = src_frame
    R_src = np.column_stack([src_x, src_y, src_z])
    dst_x, dst_y, dst_z = dst_frame
    R_dst = np.column_stack([dst_x, dst_y, dst_z])
    R = R_dst @ R_src.T
    return R


def transform_template(
    template: NucleotideTemplate,
    target_p_position: np.ndarray,
    target_frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, np.ndarray]:
    template_frame = compute_template_frame(template)
    R = compute_rotation_matrix(template_frame, target_frame)
    template_p = template.atoms["P"]
    transformed = {}
    for atom_name, atom_pos in template.atoms.items():
        pos_centered = atom_pos - template_p
        pos_rotated = R @ pos_centered
        pos_final = pos_rotated + target_p_position
        transformed[atom_name] = pos_final
    return transformed


def load_backbone_json(backbone_path: Path) -> Tuple[List[Dict], Dict]:
    with open(backbone_path) as f:
        data = json.load(f)
    if "chains" in data:
        chain = data["chains"][0]
        atoms = chain.get("p_atoms", chain.get("atoms", []))
    elif "p_atoms" in data:
        atoms = data["p_atoms"]
    elif "atoms" in data:
        atoms = data["atoms"]
    else:
        raise ValueError(f"Could not find atom data in {backbone_path}")
    metadata = {
        "num_chains": data.get("num_chains", 1),
        "total_atoms": data.get("total_atoms", len(atoms)),
        "edge_correction": data.get("edge_correction"),
        "edge_statistics": data.get("edge_statistics"),
    }
    return (atoms, metadata)


def write_pdb(model: AtomicModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("REMARK   Generated by RNA Cryo-EM Pipeline Phase 4\n")
        f.write(f"REMARK   {model.n_residues} residues, {model.n_atoms} atoms\n")
        atom_serial = 1
        for residue in model.residues:
            res_name = residue.base
            res_num = residue.residue_num
            chain = residue.chain_id
            for atom in residue.atoms:
                if len(atom.name) < 4:
                    atom_name = f" {atom.name:<3s}"
                else:
                    atom_name = f"{atom.name:<4s}"
                x, y, z = atom.position
                f.write(
                    f"ATOM  {atom_serial:5d} {atom_name:4s} {res_name:>3s} {chain:1s}{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{atom.occupancy:6.2f}{atom.b_factor:6.2f}          {atom.element:>2s}\n"
                )
                atom_serial += 1
        f.write("TER\n")
        f.write("END\n")
    logger.info(f"Wrote PDB to {output_path}")


def write_mmcif(model: AtomicModel, output_path: Path, pdb_id: str = "XXXX") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"data_{pdb_id}\n")
        f.write("#\n")
        f.write("_entry.id   " + pdb_id + "\n")
        f.write("#\n")
        f.write("loop_\n")
        f.write("_atom_site.group_PDB\n")
        f.write("_atom_site.id\n")
        f.write("_atom_site.type_symbol\n")
        f.write("_atom_site.label_atom_id\n")
        f.write("_atom_site.label_alt_id\n")
        f.write("_atom_site.label_comp_id\n")
        f.write("_atom_site.label_asym_id\n")
        f.write("_atom_site.label_entity_id\n")
        f.write("_atom_site.label_seq_id\n")
        f.write("_atom_site.pdbx_PDB_ins_code\n")
        f.write("_atom_site.Cartn_x\n")
        f.write("_atom_site.Cartn_y\n")
        f.write("_atom_site.Cartn_z\n")
        f.write("_atom_site.occupancy\n")
        f.write("_atom_site.B_iso_or_equiv\n")
        f.write("_atom_site.pdbx_formal_charge\n")
        f.write("_atom_site.auth_seq_id\n")
        f.write("_atom_site.auth_comp_id\n")
        f.write("_atom_site.auth_asym_id\n")
        f.write("_atom_site.auth_atom_id\n")
        f.write("_atom_site.pdbx_PDB_model_num\n")
        atom_serial = 1
        for residue in model.residues:
            res_name = residue.base
            res_num = residue.residue_num
            chain = residue.chain_id
            for atom in residue.atoms:
                x, y, z = atom.position
                f.write(
                    f"ATOM {atom_serial} {atom.element} {atom.name} . {res_name} {chain} 1 {res_num} ? {x:.3f} {y:.3f} {z:.3f} {atom.occupancy:.2f} {atom.b_factor:.2f} ? {res_num} {res_name} {chain} {atom.name} 1\n"
                )
                atom_serial += 1
        f.write("#\n")
    logger.info(f"Wrote mmCIF to {output_path}")


def write_json_model(
    model: AtomicModel, output_path: Path, metadata: Dict = None
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "summary": {
            "n_residues": model.n_residues,
            "n_atoms": model.n_atoms,
            "chain_id": model.chain_id,
        },
        "residues": [r.to_dict() for r in model.residues],
    }
    if metadata:
        data["metadata"] = metadata
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote JSON to {output_path}")


def _centroid_direction(
    atoms: Dict[str, np.ndarray], atom_names: List[str], p_position: np.ndarray
) -> Optional[np.ndarray]:
    points = [atoms[name] for name in atom_names if name in atoms]
    if not points:
        return None
    centroid = np.mean(np.asarray(points, dtype=np.float64), axis=0)
    direction = centroid - p_position
    norm = np.linalg.norm(direction)
    if norm <= 1e-06:
        return None
    return direction / norm


def _anchor_direction(
    anchor: Dict[str, Any], p_position: np.ndarray
) -> Optional[np.ndarray]:
    centroid = anchor.get("centroid")
    if centroid is None:
        return None
    direction = np.array(centroid, dtype=np.float64) - p_position
    norm = np.linalg.norm(direction)
    if norm <= 1.0:
        return None
    return direction / norm


def optimize_orientation_from_map_features(
    transformed_atoms: Dict[str, np.ndarray],
    p_position: np.ndarray,
    x_axis: np.ndarray,
    atom_data: Dict[str, Any],
    base_type: str,
    p_partner: Optional[np.ndarray] = None,
    feature_weight: float = 0.75,
) -> Dict[str, np.ndarray]:
    targets = []
    base_direction = _anchor_direction(atom_data.get("base_anchor") or {}, p_position)
    if base_direction is not None:
        targets.append((BASE_ATOMS.get(base_type, []), base_direction, 1.0))
    sugar_direction = _anchor_direction(atom_data.get("sugar_anchor") or {}, p_position)
    if sugar_direction is not None:
        targets.append((BACKBONE_ATOMS[5:], sugar_direction, 0.5))
    if not targets and p_partner is None:
        return transformed_atoms
    partner_direction = None
    if p_partner is not None:
        partner_vec = p_partner - p_position
        partner_norm = np.linalg.norm(partner_vec)
        if partner_norm > 1.0:
            partner_direction = partner_vec / partner_norm
    wc_edge_atoms = {
        "A": ["N6", "N1"],
        "U": ["O4", "N3"],
        "G": ["O6", "N1", "N2"],
        "C": ["N4", "N3"],
    }.get(base_type, [])
    best_angle = 0.0
    best_score = float("inf")
    feature_weight = max(0.0, min(1.0, float(feature_weight)))
    for angle_deg in range(0, 360, 10):
        angle = np.radians(angle_deg)
        rotated = rotate_around_axis(transformed_atoms, x_axis, p_position, angle)
        score = 0.0
        feature_terms = 0
        for atom_names, target_direction, weight in targets:
            model_direction = _centroid_direction(rotated, atom_names, p_position)
            if model_direction is None:
                continue
            score += (
                feature_weight
                * weight
                * (1.0 - np.dot(model_direction, target_direction))
            )
            feature_terms += 1
        if feature_terms == 0 and targets:
            continue
        if partner_direction is not None:
            wc_direction = _centroid_direction(rotated, wc_edge_atoms, p_position)
            if wc_direction is not None:
                score += (1.0 - feature_weight) * (
                    1.0 - np.dot(wc_direction, partner_direction)
                )
        if score < best_score:
            best_score = score
            best_angle = angle
    if best_angle != 0.0:
        return rotate_around_axis(transformed_atoms, x_axis, p_position, best_angle)
    return transformed_atoms


def build_atomic_model(
    p_atoms: List[Dict],
    templates: Dict[str, NucleotideTemplate],
    chain_id: str = "A",
    partner_map: Optional[Dict[int, int]] = None,
    map_feature_min_confidence: float = DEFAULT_MAP_FEATURE_MIN_CONFIDENCE,
    optimize_map_features: bool = True,
) -> AtomicModel:
    model = AtomicModel(chain_id=chain_id)
    sorted_atoms = sorted(p_atoms, key=lambda a: a.get("residue_num", 0))
    n_atoms = len(sorted_atoms)
    if n_atoms == 0:
        raise ValueError("No P atoms to process")
    positions = np.array([a["position"] for a in sorted_atoms], dtype=np.float32)
    if positions.shape != (n_atoms, 3) or not np.isfinite(positions).all():
        raise ValueError("P coordinates must be finite with shape (N, 3)")
    residue_numbers = [a.get("residue_num", i + 1) for i, a in enumerate(sorted_atoms)]
    if len(set(residue_numbers)) != len(residue_numbers):
        raise ValueError("Duplicate residue numbers in backbone")
    resnum_to_idx = {}
    for i, atom_data in enumerate(sorted_atoms):
        res_num = atom_data.get("residue_num", i + 1)
        resnum_to_idx[res_num] = i
    n_with_partner = 0
    n_without_partner = 0
    n_map_feature_guided = 0
    n_partner_guided = 0
    n_arbitrary = 0
    logger.info(f"Building atomic model for {n_atoms} residues")
    if partner_map:
        logger.info(
            f"Using base pair information for {len(partner_map)} paired residues"
        )
    for i, atom_data in enumerate(sorted_atoms):
        residue_num = atom_data.get("residue_num", i + 1)
        base = atom_data.get("base")
        p_position = np.array(atom_data["position"], dtype=np.float32)
        if base not in templates:
            raise ValueError(f"No template for base {base} at residue {residue_num}")
        template = templates[base]
        p_prev = positions[i - 1] if i > 0 else None
        p_next = positions[i + 1] if i < n_atoms - 1 else None
        p_partner = None
        if partner_map and residue_num in partner_map:
            partner_resnum = partner_map[residue_num]
            if partner_resnum in resnum_to_idx:
                partner_idx = resnum_to_idx[partner_resnum]
                p_partner = positions[partner_idx]
                n_with_partner += 1
            else:
                logger.debug(f"Partner residue {partner_resnum} not found in backbone")
                n_without_partner += 1
        else:
            n_without_partner += 1
        map_feature_hint = get_map_feature_orientation_hint(
            atom_data, p_position, min_confidence=map_feature_min_confidence
        )
        if map_feature_hint is not None:
            orientation_hint = map_feature_hint
            local_frame = compute_local_frame(p_prev, p_position, p_next, None)
            x_axis, y_axis, z_axis = local_frame
            perp_component = (
                orientation_hint - np.dot(orientation_hint, x_axis) * x_axis
            )
            perp_mag = np.linalg.norm(perp_component)
            if perp_mag > 0.1:
                y_axis = perp_component / perp_mag
                z_axis = np.cross(x_axis, y_axis)
                z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-08)
                local_frame = (x_axis, y_axis, z_axis)
            n_map_feature_guided += 1
        else:
            local_frame = compute_local_frame(p_prev, p_position, p_next, p_partner)
            x_axis, y_axis, z_axis = local_frame
            if p_partner is not None:
                n_partner_guided += 1
            else:
                n_arbitrary += 1
        transformed_atoms = transform_template(template, p_position, local_frame)
        if map_feature_hint is not None and optimize_map_features:
            transformed_atoms = optimize_orientation_from_map_features(
                transformed_atoms,
                p_position,
                x_axis,
                atom_data,
                base,
                p_partner=p_partner,
                feature_weight=0.75,
            )
        elif p_partner is not None:
            transformed_atoms = optimize_base_orientation(
                transformed_atoms, p_position, x_axis, p_partner, base
            )
        residue = Residue(residue_num=residue_num, base=base, chain_id=chain_id)
        for atom_name in ALL_ATOMS.get(base, BACKBONE_ATOMS):
            if atom_name in transformed_atoms:
                element = ATOM_ELEMENTS.get(atom_name, atom_name[0])
                atom = Atom(
                    name=atom_name,
                    position=transformed_atoms[atom_name],
                    element=element,
                )
                residue.atoms.append(atom)
        model.residues.append(residue)
    logger.info(f"Built model with {model.n_residues} residues, {model.n_atoms} atoms")
    logger.info("Orientation sources:")
    logger.info(f"  Map-feature:     {n_map_feature_guided}")
    logger.info(f"  Partner-guided:  {n_partner_guided}")
    logger.info(f"  Arbitrary:       {n_arbitrary}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Build an RNA atomic model from registered phosphate coordinates"
    )
    parser.add_argument("--backbone", "-b", type=Path, required=True)
    parser.add_argument("--structure-2d", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--optimize-map-features", action="store_true", default=True)
    args = parser.parse_args()
    p_atoms, metadata = load_backbone_json(args.backbone)
    model = build_atomic_model(
        p_atoms,
        get_default_templates(),
        chain_id=args.chain_id,
        partner_map=load_2d_structure(args.structure_2d),
        optimize_map_features=args.optimize_map_features,
    )
    write_pdb(model, args.output)


if __name__ == "__main__":
    main()
