from __future__ import annotations
import argparse
import json
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from openmm import (
    app,
    unit,
    CustomExternalForce,
    CustomBondForce,
    Platform,
    LangevinMiddleIntegrator,
)
import openmm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
HYDROGEN_PLACEMENT_RANDOM_SEED = 20260811
OPENMM_DETERMINISTIC_FORCES = True
WC_HBOND_DISTANCES = {
    ("A", "U"): [("N6", "O4", 2.95), ("N1", "N3", 2.82)],
    ("U", "A"): [("O4", "N6", 2.95), ("N3", "N1", 2.82)],
    ("G", "C"): [("O6", "N4", 2.91), ("N1", "N3", 2.95), ("N2", "O2", 2.86)],
    ("C", "G"): [("N4", "O6", 2.91), ("N3", "N1", 2.95), ("O2", "N2", 2.86)],
    ("G", "U"): [("O6", "N3", 2.8), ("N1", "O2", 2.84)],
    ("U", "G"): [("N3", "O6", 2.8), ("O2", "N1", 2.84)],
}
DEFAULT_P_RESTRAINT_WEIGHT = 100000.0
DEFAULT_HBOND_RESTRAINT_WEIGHT = 100.0
DEFAULT_HBOND_TOLERANCE = 0.5
DEFAULT_DENSITY_RESTRAINT_WEIGHT = 500.0
DEFAULT_DENSITY_SEARCH_RADIUS = 6.0
DEFAULT_P5_CAP_BOND_LENGTH = 1.5
DENSITY_ATOM_INDICES = {
    "P": 1,
    "OP1": 2,
    "OP2": 3,
    "O5'": 4,
    "C5'": 5,
    "C4'": 6,
    "O4'": 7,
    "C3'": 8,
    "O3'": 9,
    "C2'": 10,
    "O2'": 11,
    "C1'": 12,
    "N9": 13,
    "N7": 14,
    "C8": 15,
    "N1": 16,
    "C2": 17,
    "C6": 18,
}
DENSITY_GROUP_INDICES = {
    "background": 0,
    "p_group": 1,
    "sugar_group": 2,
    "base_group": 3,
}
LEGACY_FOUR_CLASS_SCHEMA = "0bg_1p_2sugar_3base"
TYPED_FIVE_CLASS_SCHEMA = "0bg_1p_2sugar_3purine_4pyrimidine"
DENSITY_TYPED_GROUP_INDICES = {
    "background": 0,
    "p_group": 1,
    "sugar_group": 2,
    "purine_group": 3,
    "pyrimidine_group": 4,
}
ATOM_TO_GROUP = {
    "P": "p_group",
    "OP1": "p_group",
    "OP2": "p_group",
    "O5'": "sugar_group",
    "C5'": "sugar_group",
    "C4'": "sugar_group",
    "O4'": "sugar_group",
    "C3'": "sugar_group",
    "O3'": "sugar_group",
    "C2'": "sugar_group",
    "O2'": "sugar_group",
    "C1'": "sugar_group",
    "N9": "base_group",
    "N7": "base_group",
    "C8": "base_group",
    "N1": "base_group",
    "C2": "base_group",
    "C6": "base_group",
    "N3": "base_group",
    "C4": "base_group",
    "C5": "base_group",
    "N6": "base_group",
    "O6": "base_group",
    "N2": "base_group",
    "O2": "base_group",
    "N4": "base_group",
    "O4": "base_group",
}
PURINE_RESIDUE_NAMES = {"A", "G", "ADE", "GUA", "RA", "RG", "DA", "DG"}
PYRIMIDINE_RESIDUE_NAMES = {"C", "U", "CYT", "URA", "RC", "RU", "DC", "DU", "DT", "T"}
RNA_BASE_NAME_MAP = {
    "A": "A",
    "ADE": "A",
    "RA": "A",
    "DA": "A",
    "C": "C",
    "CYT": "C",
    "RC": "C",
    "DC": "C",
    "G": "G",
    "GUA": "G",
    "RG": "G",
    "DG": "G",
    "U": "U",
    "URA": "U",
    "RU": "U",
    "DU": "U",
    "DT": "U",
    "T": "U",
}
AUXILIARY_CAP_RESIDUE_NAME = "P5"
AUXILIARY_CAP_ATOM_NAME = "OP3"
AUXILIARY_CAP_METHOD = "p5_op3_tetrahedral"
CUSTOM_P5_TEMPLATE_XML = f'<ForceField>\n  <Residues>\n    <Residue name="{AUXILIARY_CAP_RESIDUE_NAME}">\n      <Atom name="{AUXILIARY_CAP_ATOM_NAME}" type="RNA-O2" charge="-0.776"/>\n      <ExternalBond atomName="{AUXILIARY_CAP_ATOM_NAME}"/>\n    </Residue>\n  </Residues>\n</ForceField>\n'


@dataclass
class RefinementResult:
    input_path: str
    output_path: str
    initial_energy: float
    final_energy: float
    minimization_steps: int
    converged: bool
    n_p_restraints: int
    n_hbond_restraints: int
    hydrogen_placement_random_seed: int = HYDROGEN_PLACEMENT_RANDOM_SEED
    openmm_deterministic_forces: bool = OPENMM_DETERMINISTIC_FORCES
    openmm_cpu_threads: int = 1
    energy_after_minimization: Optional[float] = None
    md_steps_requested: int = 0
    md_steps_completed: int = 0
    md_temperature_kelvin: Optional[float] = None
    md_timestep_femtoseconds: Optional[float] = None
    md_friction_per_picosecond: Optional[float] = None
    md_random_seed: Optional[int] = None
    md_potential_energy_before: Optional[float] = None
    md_potential_energy_after: Optional[float] = None
    post_md_minimization_iterations: int = 0
    post_md_minimization_converged: Optional[bool] = None
    n_density_restraints: int = 0
    n_auxiliary_caps: int = 0
    p_restraint_weight: float = 0.0
    hbond_restraint_weight: float = 0.0
    density_restraint_weight: float = 0.0
    auxiliary_cap_method: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "initial_energy_kJ_mol": self.initial_energy,
            "final_energy_kJ_mol": self.final_energy,
            "minimization_steps": self.minimization_steps,
            "converged": self.converged,
            "hydrogen_placement_random_seed": self.hydrogen_placement_random_seed,
            "openmm_deterministic_forces": self.openmm_deterministic_forces,
            "openmm_cpu_threads": self.openmm_cpu_threads,
            "energy_after_minimization_kJ_mol": self.energy_after_minimization,
            "md_steps_requested": self.md_steps_requested,
            "md_steps_completed": self.md_steps_completed,
            "md_temperature_kelvin": self.md_temperature_kelvin,
            "md_timestep_femtoseconds": self.md_timestep_femtoseconds,
            "md_friction_per_picosecond": self.md_friction_per_picosecond,
            "md_random_seed": self.md_random_seed,
            "md_potential_energy_before_kJ_mol": self.md_potential_energy_before,
            "md_potential_energy_after_kJ_mol": self.md_potential_energy_after,
            "post_md_minimization_iterations": self.post_md_minimization_iterations,
            "post_md_minimization_converged": self.post_md_minimization_converged,
            "n_p_restraints": self.n_p_restraints,
            "n_hbond_restraints": self.n_hbond_restraints,
            "n_density_restraints": self.n_density_restraints,
            "n_auxiliary_caps": self.n_auxiliary_caps,
            "p_restraint_weight": self.p_restraint_weight,
            "hbond_restraint_weight": self.hbond_restraint_weight,
            "density_restraint_weight": self.density_restraint_weight,
            "auxiliary_cap_method": self.auxiliary_cap_method,
        }


@dataclass
class BasePair:
    res_i: int
    res_j: int
    base_i: str
    base_j: str
    canonical: bool


def build_amber_forcefield() -> app.ForceField:
    temp_xml_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix="_p5_template.xml", delete=False
        ) as handle:
            handle.write(CUSTOM_P5_TEMPLATE_XML)
            temp_xml_path = handle.name
        return app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml", temp_xml_path)
    finally:
        if temp_xml_path is not None:
            try:
                Path(temp_xml_path).unlink()
            except OSError:
                pass


def configure_openmm_platform(platform_name: str):
    platform = Platform.getPlatformByName(platform_name)
    available = set(platform.getPropertyNames())
    properties: Dict[str, str] = {}
    if OPENMM_DETERMINISTIC_FORCES and "DeterministicForces" in available:
        properties["DeterministicForces"] = "true"
    if platform_name.upper() == "CPU" and "Threads" in available:
        properties["Threads"] = str(int(os.environ.get("OPENMM_CPU_THREADS", "1")))
    for name, value in properties.items():
        platform.setPropertyDefaultValue(name, value)
    return (platform, properties)


def _position_to_angstrom_array(position: Any) -> np.ndarray:
    coords = []
    for index in range(3):
        value = position[index]
        if hasattr(value, "value_in_unit"):
            coords.append(value.value_in_unit(unit.angstrom))
        else:
            coords.append(float(value))
    return np.array(coords, dtype=float)


def _canonical_rna_base_name(residue_name: str) -> Optional[str]:
    name = str(residue_name).upper()
    if len(name) == 2 and name[0] in "ACGU" and name[1] in "35":
        name = name[0]
    return RNA_BASE_NAME_MAP.get(name)


def _quantity_from_angstrom_array(position_xyz: np.ndarray):
    return openmm.Vec3(*[float(value) for value in position_xyz]) * unit.angstrom


def _positions_quantity_from_angstrom_arrays(position_arrays: List[np.ndarray]):
    return unit.Quantity(
        [
            openmm.Vec3(*[float(value) for value in position])
            for position in position_arrays
        ],
        unit.angstrom,
    )


def add_missing_phosphodiester_bonds(
    topology: app.Topology,
    positions: List,
    max_o3_p_distance: float = 4.0,
    verbose: bool = False,
) -> int:
    existing_bonds = {
        tuple(sorted((atom1.index, atom2.index))) for atom1, atom2 in topology.bonds()
    }
    added_bonds = 0
    for chain in topology.chains():
        residues = list(chain.residues())
        for residue_i, residue_j in zip(residues, residues[1:]):
            atoms_i = {atom.name: atom for atom in residue_i.atoms()}
            atoms_j = {atom.name: atom for atom in residue_j.atoms()}
            o3_atom = atoms_i.get("O3'")
            p_atom = atoms_j.get("P")
            if o3_atom is None or p_atom is None:
                continue
            bond_key = tuple(sorted((o3_atom.index, p_atom.index)))
            if bond_key in existing_bonds:
                continue
            distance = np.linalg.norm(
                _position_to_angstrom_array(positions[o3_atom.index])
                - _position_to_angstrom_array(positions[p_atom.index])
            )
            if distance > max_o3_p_distance:
                if verbose:
                    logger.warning(
                        "Skipping O3'-P bond %s:%s -> %s:%s (distance %.2f A)",
                        residue_i.id,
                        o3_atom.name,
                        residue_j.id,
                        p_atom.name,
                        distance,
                    )
                continue
            topology.addBond(o3_atom, p_atom)
            existing_bonds.add(bond_key)
            added_bonds += 1
    return added_bonds


def rename_final_rna_residue_to_3prime_template(
    topology: app.Topology, verbose: bool = False
) -> int:
    renamed = 0
    for chain in topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        residue = residues[-1]
        canonical = _canonical_rna_base_name(residue.name)
        target_name = f"{canonical}3" if canonical is not None else None
        if not target_name or residue.name == target_name:
            continue
        if verbose:
            logger.info(
                "  Renaming terminal residue %s:%s %s -> %s",
                chain.id,
                residue.id,
                residue.name,
                target_name,
            )
        residue.name = target_name
        renamed += 1
    return renamed


def compute_p5_cap_position(
    p_position: np.ndarray,
    op1_position: np.ndarray,
    op2_position: np.ndarray,
    o5_position: np.ndarray,
    bond_length: float = DEFAULT_P5_CAP_BOND_LENGTH,
) -> np.ndarray:
    unit_vectors = []
    for neighbor in (op1_position, op2_position, o5_position):
        vector = np.asarray(neighbor, dtype=float) - np.asarray(p_position, dtype=float)
        norm = np.linalg.norm(vector)
        if norm <= 1e-08:
            raise ValueError(
                "Cannot construct P5 cap direction from coincident phosphate geometry"
            )
        unit_vectors.append(vector / norm)
    direction = -(unit_vectors[0] + unit_vectors[1] + unit_vectors[2])
    norm = np.linalg.norm(direction)
    if norm <= 1e-08:
        raise ValueError(
            "Cannot construct P5 cap direction from degenerate phosphate geometry"
        )
    return np.asarray(p_position, dtype=float) + bond_length * direction / norm


def _copy_topology_and_positions(
    topology: app.Topology, positions: List
) -> Tuple[app.Topology, List, Dict[Any, Any], Dict[Any, Any]]:
    new_topology = app.Topology()
    try:
        vectors = topology.getPeriodicBoxVectors()
        if vectors is not None:
            new_topology.setPeriodicBoxVectors(vectors)
    except Exception:
        pass
    atom_map = {}
    residue_map = {}
    new_position_arrays = []
    for chain in topology.chains():
        new_chain = new_topology.addChain(chain.id)
        for residue in chain.residues():
            new_residue = new_topology.addResidue(
                residue.name,
                new_chain,
                id=residue.id,
                insertionCode=getattr(residue, "insertionCode", ""),
            )
            residue_map[residue] = new_residue
            for atom in residue.atoms():
                new_atom = new_topology.addAtom(
                    atom.name,
                    atom.element,
                    new_residue,
                    id=getattr(atom, "id", None),
                    formalCharge=getattr(atom, "formalCharge", None),
                )
                atom_map[atom] = new_atom
                new_position_arrays.append(
                    _position_to_angstrom_array(positions[atom.index])
                )
    for atom1, atom2 in topology.bonds():
        new_topology.addBond(atom_map[atom1], atom_map[atom2])
    return (
        new_topology,
        _positions_quantity_from_angstrom_arrays(new_position_arrays),
        atom_map,
        residue_map,
    )


def add_auxiliary_p5_caps(
    topology: app.Topology, positions: List, verbose: bool = False
) -> Tuple[app.Topology, List, int, set[Tuple[str, str]]]:
    new_topology, new_positions, atom_map, residue_map = _copy_topology_and_positions(
        topology, positions
    )
    new_position_arrays = [
        _position_to_angstrom_array(position) for position in new_positions
    ]
    cap_bearing_residue_keys: set[Tuple[str, str]] = set()
    n_caps = 0
    for chain in topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        first_residue = residues[0]
        if _canonical_rna_base_name(first_residue.name) is None:
            continue
        atoms = {atom.name: atom for atom in first_residue.atoms()}
        required_names = {"P", "OP1", "OP2", "O5'"}
        if not required_names.issubset(atoms):
            continue
        p_position = _position_to_angstrom_array(positions[atoms["P"].index])
        op1_position = _position_to_angstrom_array(positions[atoms["OP1"].index])
        op2_position = _position_to_angstrom_array(positions[atoms["OP2"].index])
        o5_position = _position_to_angstrom_array(positions[atoms["O5'"].index])
        cap_position = compute_p5_cap_position(
            p_position, op1_position, op2_position, o5_position
        )
        cap_chain = new_topology.addChain(f"{chain.id}_cap")
        cap_residue = new_topology.addResidue(
            AUXILIARY_CAP_RESIDUE_NAME, cap_chain, id=first_residue.id
        )
        cap_atom = new_topology.addAtom(
            AUXILIARY_CAP_ATOM_NAME, app.element.oxygen, cap_residue
        )
        new_topology.addBond(cap_atom, atom_map[atoms["P"]])
        new_position_arrays.append(cap_position)
        cap_bearing_residue_keys.add((chain.id, first_residue.id))
        n_caps += 1
        if verbose:
            logger.info(
                "  Added P5 cap to %s:%s at [%.3f, %.3f, %.3f] A",
                chain.id,
                first_residue.id,
                cap_position[0],
                cap_position[1],
                cap_position[2],
            )
    return (
        new_topology,
        _positions_quantity_from_angstrom_arrays(new_position_arrays),
        n_caps,
        cap_bearing_residue_keys,
    )


def delete_spurious_ho5_prime_hydrogens(
    modeller: app.Modeller,
    phosphate_residue_keys: set[Tuple[str, str]],
    verbose: bool = False,
) -> int:
    atoms_to_delete = []
    for atom in modeller.topology.atoms():
        key = (atom.residue.chain.id, atom.residue.id)
        if key in phosphate_residue_keys and atom.name == "HO5'":
            atoms_to_delete.append(atom)
    if atoms_to_delete:
        modeller.delete(atoms_to_delete)
        if verbose:
            logger.info(f"  Deleted {len(atoms_to_delete)} spurious HO5' atoms")
    return len(atoms_to_delete)


def strip_auxiliary_p5_caps(
    topology: app.Topology, positions: List
) -> Tuple[app.Topology, List, int]:
    new_topology = app.Topology()
    try:
        vectors = topology.getPeriodicBoxVectors()
        if vectors is not None:
            new_topology.setPeriodicBoxVectors(vectors)
    except Exception:
        pass
    atom_map = {}
    new_position_arrays = []
    removed_count = 0
    for chain in topology.chains():
        kept_residues = [
            residue
            for residue in chain.residues()
            if residue.name != AUXILIARY_CAP_RESIDUE_NAME
        ]
        if not kept_residues:
            removed_count += sum(
                (
                    1
                    for residue in chain.residues()
                    if residue.name == AUXILIARY_CAP_RESIDUE_NAME
                )
            )
            continue
        new_chain = new_topology.addChain(chain.id)
        for residue in chain.residues():
            if residue.name == AUXILIARY_CAP_RESIDUE_NAME:
                removed_count += 1
                continue
            new_residue = new_topology.addResidue(
                residue.name,
                new_chain,
                id=residue.id,
                insertionCode=getattr(residue, "insertionCode", ""),
            )
            for atom in residue.atoms():
                new_atom = new_topology.addAtom(
                    atom.name,
                    atom.element,
                    new_residue,
                    id=getattr(atom, "id", None),
                    formalCharge=getattr(atom, "formalCharge", None),
                )
                atom_map[atom] = new_atom
                new_position_arrays.append(
                    _position_to_angstrom_array(positions[atom.index])
                )
    for atom1, atom2 in topology.bonds():
        if atom1 in atom_map and atom2 in atom_map:
            new_topology.addBond(atom_map[atom1], atom_map[atom2])
    return (
        new_topology,
        _positions_quantity_from_angstrom_arrays(new_position_arrays),
        removed_count,
    )


def prepare_structure_with_openmm(
    pdb_path: Path, verbose: bool = False
) -> app.Modeller:
    logger.info(f"Preparing structure with OpenMM-only path: {pdb_path}")
    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    positions = pdb.positions
    n_added_bonds = add_missing_phosphodiester_bonds(
        topology, positions, verbose=verbose
    )
    if n_added_bonds > 0:
        logger.info(f"  Added {n_added_bonds} missing O3'-P backbone bonds")
    capped_topology, capped_positions, n_caps, cap_bearing_residue_keys = (
        add_auxiliary_p5_caps(topology, positions, verbose=verbose)
    )
    if n_caps > 0:
        logger.info(
            f"  Added {n_caps} auxiliary P5 caps for phosphate-bearing 5' termini"
        )
    modeller = app.Modeller(capped_topology, capped_positions)
    random_state = random.getstate()
    random.seed(HYDROGEN_PLACEMENT_RANDOM_SEED)
    try:
        hydrogen_platform, hydrogen_platform_properties = configure_openmm_platform(
            os.environ.get("OPENMM_PLATFORM", "CPU")
        )
        modeller.addHydrogens(pH=7.0, platform=hydrogen_platform)
    except Exception as exc:
        raise RuntimeError(
            f"OpenMM-only hydrogen addition failed for {pdb_path}: {exc}"
        ) from exc
    finally:
        random.setstate(random_state)
    logger.info(
        "  Added hydrogens with deterministic placement seed %d and platform properties %s",
        HYDROGEN_PLACEMENT_RANDOM_SEED,
        hydrogen_platform_properties,
    )
    delete_spurious_ho5_prime_hydrogens(
        modeller, cap_bearing_residue_keys, verbose=verbose
    )
    n_terminal_renames = rename_final_rna_residue_to_3prime_template(
        modeller.topology, verbose=verbose
    )
    if n_terminal_renames > 0:
        logger.info(
            f"  Renamed {n_terminal_renames} terminal residues to 3' AMBER variants"
        )
    n_atoms = modeller.topology.getNumAtoms()
    n_residues = modeller.topology.getNumResidues()
    logger.info(f"  Prepared structure: {n_residues} residues, {n_atoms} atoms")
    return modeller


def fix_rna_termini(pdb_path: Path, output_path: Path, verbose: bool = False) -> Path:
    logger.info(
        "Preserving terminal phosphates; terminal hydrogens will be handled during preparation"
    )
    with open(pdb_path) as f:
        lines = f.readlines()
    atom_lines = [
        line for line in lines if line.startswith("ATOM") or line.startswith("HETATM")
    ]
    if not atom_lines:
        raise ValueError(f"No atoms in {pdb_path}")
    if verbose:
        first_atom = atom_lines[0][12:16].strip()
        first_res = atom_lines[0][22:26].strip()
        last_res = atom_lines[-1][22:26].strip()
        logger.info(f"  First atom/residue preserved: {first_atom} / {first_res}")
        logger.info(f"  Last residue preserved: {last_res}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(lines)
    logger.info(f"  Coordinate-preserving copy written to: {output_path}")
    return output_path


def create_system_with_restraints(
    modeller: app.Modeller,
    p_restraint_weight: float = DEFAULT_P_RESTRAINT_WEIGHT,
    verbose: bool = False,
) -> Tuple[openmm.System, Dict[str, int]]:
    logger.info("Creating OpenMM system with AMBER14 force field")
    forcefield = build_amber_forcefield()
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=1.5 * unit.amu,
    )
    if verbose:
        logger.info(f"  Created system with {system.getNumParticles()} particles")
        logger.info(
            f"  Forces: {[system.getForce(i).__class__.__name__ for i in range(system.getNumForces())]}"
        )
    p_atom_indices = {}
    positions = modeller.positions
    for atom in modeller.topology.atoms():
        if atom.name == "P":
            res_num = atom.residue.id
            try:
                res_num = int(res_num)
            except (ValueError, TypeError):
                pass
            p_atom_indices[res_num] = atom.index
    logger.info(f"  Found {len(p_atom_indices)} P atoms for restraints")
    if p_restraint_weight > 0 and len(p_atom_indices) > 0:
        restraint_force = CustomExternalForce("k_p*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        restraint_force.addGlobalParameter("k_p", p_restraint_weight)
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")
        for res_num, atom_idx in p_atom_indices.items():
            pos = positions[atom_idx]
            x0 = pos[0].value_in_unit(unit.nanometer)
            y0 = pos[1].value_in_unit(unit.nanometer)
            z0 = pos[2].value_in_unit(unit.nanometer)
            restraint_force.addParticle(atom_idx, [x0, y0, z0])
        system.addForce(restraint_force)
        logger.info(
            f"  Added P atom position restraints (k={p_restraint_weight} kJ/mol/nm²)"
        )
    return (system, p_atom_indices)


def add_basepair_hbond_restraints(
    system: openmm.System,
    modeller: app.Modeller,
    base_pairs: List[BasePair],
    hbond_weight: float = DEFAULT_HBOND_RESTRAINT_WEIGHT,
    tolerance: float = DEFAULT_HBOND_TOLERANCE,
    verbose: bool = False,
) -> int:
    logger.info("Adding base pair H-bond restraints")
    atom_lookup = {}
    for atom in modeller.topology.atoms():
        res_num = atom.residue.id
        try:
            res_num = int(res_num)
        except (ValueError, TypeError):
            pass
        key = (res_num, atom.name)
        atom_lookup[key] = atom.index
    residue_bases = {}
    for residue in modeller.topology.residues():
        res_num = residue.id
        try:
            res_num = int(res_num)
        except (ValueError, TypeError):
            pass
        base = _canonical_rna_base_name(residue.name)
        if base is not None:
            residue_bases[res_num] = base
    tolerance_nm = tolerance / 10.0
    hbond_force = CustomBondForce(
        f"k_hbond * step(abs(r - r0) - tol) * (abs(r - r0) - tol)^2; tol={tolerance_nm}"
    )
    hbond_force.addGlobalParameter("k_hbond", hbond_weight)
    hbond_force.addPerBondParameter("r0")
    n_restraints = 0
    n_skipped = 0
    for bp in base_pairs:
        base_i = residue_bases.get(bp.res_i, bp.base_i)
        base_j = residue_bases.get(bp.res_j, bp.base_j)
        hbond_specs = WC_HBOND_DISTANCES.get((base_i, base_j))
        if hbond_specs is None:
            hbond_specs = WC_HBOND_DISTANCES.get((base_j, base_i))
            if hbond_specs is not None:
                bp.res_i, bp.res_j = (bp.res_j, bp.res_i)
                base_i, base_j = (base_j, base_i)
        if hbond_specs is None:
            if verbose:
                logger.debug(
                    f"  No H-bond specs for {base_i}-{base_j} pair (res {bp.res_i}-{bp.res_j})"
                )
            n_skipped += 1
            continue
        for atom_name_i, atom_name_j, target_dist in hbond_specs:
            atom_i = atom_lookup.get((bp.res_i, atom_name_i))
            atom_j = atom_lookup.get((bp.res_j, atom_name_j))
            if atom_i is None or atom_j is None:
                if verbose:
                    logger.debug(
                        f"  Missing atoms: {bp.res_i}:{atom_name_i} or {bp.res_j}:{atom_name_j}"
                    )
                continue
            target_nm = target_dist / 10.0
            hbond_force.addBond(atom_i, atom_j, [target_nm])
            n_restraints += 1
            if verbose:
                logger.debug(
                    f"  H-bond: {bp.res_i}:{atom_name_i} - {bp.res_j}:{atom_name_j} = {target_dist:.2f} Å"
                )
    if n_restraints > 0:
        system.addForce(hbond_force)
        logger.info(
            f"  Added {n_restraints} H-bond restraints from {len(base_pairs)} base pairs"
        )
        if n_skipped > 0:
            logger.info(f"  Skipped {n_skipped} non-canonical/unsupported base pairs")
    else:
        logger.warning("  No H-bond restraints could be added")
    return n_restraints


def load_density_predictions(
    predictions_path: Path, meta_path: Path
) -> Tuple[np.ndarray, Dict[str, Any]]:
    predictions = np.load(predictions_path)
    with open(meta_path) as f:
        meta = json.load(f)
    logger.info(f"Loaded density predictions: shape={predictions.shape}")
    logger.info(f"  Voxel size: {meta.get('voxel_size_xyz', meta.get('voxel_size'))} Å")
    logger.info(f"  Origin: {meta.get('origin_xyz', meta.get('origin'))}")
    return (predictions, meta)


def world_to_voxel(
    position: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray
) -> np.ndarray:
    return ((position - origin) / voxel_size).astype(int)


def find_density_peak(
    predictions: np.ndarray,
    center_world: np.ndarray,
    atom_class: int | List[int] | Tuple[int, ...],
    origin: np.ndarray,
    voxel_size: np.ndarray,
    search_radius: float = 6.0,
) -> Optional[np.ndarray]:
    if voxel_size.ndim == 0:
        voxel_size = np.array([float(voxel_size)] * 3)
    center_voxel = world_to_voxel(center_world, origin, voxel_size)
    radius_voxels = (search_radius / voxel_size).astype(int) + 1
    shape_xyz = np.array(predictions.shape[::-1])
    x_min, y_min, z_min = np.clip(center_voxel - radius_voxels, 0, shape_xyz)
    x_max, y_max, z_max = np.clip(center_voxel + radius_voxels + 1, 0, shape_xyz)
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        return None
    region = predictions[z_min:z_max, y_min:y_max, x_min:x_max]
    atom_classes = np.atleast_1d(np.asarray(atom_class, dtype=int))
    mask = np.isin(region, atom_classes)
    if not np.any(mask):
        return None
    z_idx, y_idx, x_idx = np.where(mask)
    if len(z_idx) == 0:
        return None
    z_world = (z_idx + z_min) * voxel_size[2] + origin[2]
    y_world = (y_idx + y_min) * voxel_size[1] + origin[1]
    x_world = (x_idx + x_min) * voxel_size[0] + origin[0]
    return np.array([np.mean(x_world), np.mean(y_world), np.mean(z_world)])


def infer_prediction_layout(
    predictions: np.ndarray, meta: Optional[Dict[str, Any]] = None
) -> str:
    meta = meta or {}
    explicit_schema = meta.get("prediction_schema") or meta.get("label_schema")
    if explicit_schema in {LEGACY_FOUR_CLASS_SCHEMA, TYPED_FIVE_CLASS_SCHEMA, "atom"}:
        return explicit_schema
    class_names = meta.get("class_names")
    if isinstance(class_names, list):
        normalized_names = {str(name).strip().lower() for name in class_names}
        if {"purine", "pyrimidine"}.issubset(normalized_names):
            return TYPED_FIVE_CLASS_SCHEMA
        if {"p", "sugar", "base"}.issubset(normalized_names):
            return LEGACY_FOUR_CLASS_SCHEMA
        if len(normalized_names) > 5:
            return "atom"
    n_classes = meta.get("n_classes")
    if n_classes is not None:
        try:
            n_classes = int(n_classes)
        except (TypeError, ValueError):
            n_classes = None
        if n_classes is not None:
            if n_classes > 5:
                return "atom"
            if n_classes == 5:
                return TYPED_FIVE_CLASS_SCHEMA
            if n_classes == 4:
                return LEGACY_FOUR_CLASS_SCHEMA
    unique_labels = set(np.unique(predictions).astype(int).tolist())
    if any((label > 4 for label in unique_labels)):
        return "atom"
    if 4 in unique_labels:
        return TYPED_FIVE_CLASS_SCHEMA
    logger.warning(
        "Ambiguous prediction labels %s without explicit schema metadata; defaulting to legacy 4-label group interpretation",
        sorted(unique_labels),
    )
    return LEGACY_FOUR_CLASS_SCHEMA


def resolve_density_group_classes(
    atom: Any, prediction_layout: str
) -> Optional[int | List[int]]:
    atom_name = atom.name
    if atom_name not in ATOM_TO_GROUP:
        return None
    group_name = ATOM_TO_GROUP[atom_name]
    if prediction_layout == LEGACY_FOUR_CLASS_SCHEMA:
        return DENSITY_GROUP_INDICES[group_name]
    if prediction_layout != TYPED_FIVE_CLASS_SCHEMA:
        raise ValueError(f"Unsupported group prediction layout: {prediction_layout}")
    if group_name != "base_group":
        return DENSITY_TYPED_GROUP_INDICES[group_name]
    residue_name = _canonical_rna_base_name(getattr(atom.residue, "name", ""))
    if residue_name in PURINE_RESIDUE_NAMES:
        return DENSITY_TYPED_GROUP_INDICES["purine_group"]
    if residue_name in PYRIMIDINE_RESIDUE_NAMES:
        return DENSITY_TYPED_GROUP_INDICES["pyrimidine_group"]
    return [
        DENSITY_TYPED_GROUP_INDICES["purine_group"],
        DENSITY_TYPED_GROUP_INDICES["pyrimidine_group"],
    ]


def add_density_restraints(
    system: openmm.System,
    modeller: app.Modeller,
    predictions: np.ndarray,
    meta: Dict[str, Any],
    density_weight: float = DEFAULT_DENSITY_RESTRAINT_WEIGHT,
    search_radius: float = DEFAULT_DENSITY_SEARCH_RADIUS,
    restrain_backbone: bool = True,
    restrain_bases: bool = True,
    prediction_type: Optional[str] = None,
    verbose: bool = False,
) -> int:
    logger.info("Adding density-based position restraints (Phase 5c)")
    origin = np.array(meta.get("origin_xyz", meta.get("origin")))
    voxel_size_raw = meta.get("voxel_size_xyz", meta.get("voxel_size"))
    if isinstance(voxel_size_raw, (int, float)):
        voxel_size = np.array([voxel_size_raw, voxel_size_raw, voxel_size_raw])
    else:
        voxel_size = np.array(voxel_size_raw)
    if prediction_type == "atom":
        prediction_layout = "atom"
    elif prediction_type in {LEGACY_FOUR_CLASS_SCHEMA, TYPED_FIVE_CLASS_SCHEMA}:
        prediction_layout = prediction_type
    else:
        prediction_layout = infer_prediction_layout(predictions, meta)
    logger.info(f"  Prediction layout: {prediction_layout}")
    backbone_atoms = {
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
    }
    base_atoms = {
        "N9",
        "N7",
        "C8",
        "N1",
        "C2",
        "C6",
        "N3",
        "C4",
        "C5",
        "N6",
        "O6",
        "N2",
        "O2",
        "N4",
        "O4",
    }
    density_force = CustomExternalForce("k_density*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    density_force.addGlobalParameter("k_density", density_weight)
    density_force.addPerParticleParameter("x0")
    density_force.addPerParticleParameter("y0")
    density_force.addPerParticleParameter("z0")
    n_restraints = 0
    n_not_found = 0
    positions = modeller.positions
    for atom in modeller.topology.atoms():
        atom_name = atom.name
        if atom_name == "P":
            continue
        if atom_name in backbone_atoms and (not restrain_backbone):
            continue
        if atom_name in base_atoms and (not restrain_bases):
            continue
        if prediction_layout == "atom":
            if atom_name not in DENSITY_ATOM_INDICES:
                continue
            search_class = DENSITY_ATOM_INDICES[atom_name]
        else:
            search_class = resolve_density_group_classes(atom, prediction_layout)
            if search_class is None:
                continue
        pos = positions[atom.index]
        current_pos = np.array(
            [
                pos[0].value_in_unit(unit.angstrom),
                pos[1].value_in_unit(unit.angstrom),
                pos[2].value_in_unit(unit.angstrom),
            ]
        )
        predicted_pos = find_density_peak(
            predictions, current_pos, search_class, origin, voxel_size, search_radius
        )
        if predicted_pos is None:
            n_not_found += 1
            if verbose:
                logger.debug(
                    f"  No density peak for {atom.residue.id}:{atom_name} (class {search_class})"
                )
            continue
        x0_nm = predicted_pos[0] / 10.0
        y0_nm = predicted_pos[1] / 10.0
        z0_nm = predicted_pos[2] / 10.0
        density_force.addParticle(atom.index, [x0_nm, y0_nm, z0_nm])
        n_restraints += 1
        if verbose:
            dist = np.linalg.norm(predicted_pos - current_pos)
            logger.debug(
                f"  Restraint: {atom.residue.id}:{atom_name} -> density peak ({dist:.2f} Å away)"
            )
    if n_restraints > 0:
        system.addForce(density_force)
        logger.info(
            f"  Added {n_restraints} density restraints (k={density_weight} kJ/mol/nm²)"
        )
        if n_not_found > 0:
            logger.info(f"  Skipped {n_not_found} atoms (no density peak found)")
    else:
        raise ValueError(
            "No density restraints could be added; check map labels and coordinates"
        )
    return n_restraints


def run_energy_minimization(
    system: openmm.System,
    modeller: app.Modeller,
    max_iterations: int = 1000,
    tolerance: float = 10.0,
    preferred_platform: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List, float, float, int, bool]:
    logger.info("Running energy minimization")
    env_platform = os.environ.get("OPENMM_PLATFORM")
    platform_candidates: List[str] = []
    for candidate in (preferred_platform, env_platform, "CUDA", "OpenCL", "CPU"):
        if candidate is None:
            continue
        normalized = str(candidate).strip()
        if not normalized:
            continue
        if normalized not in platform_candidates:
            platform_candidates.append(normalized)
    simulation = None
    platform_errors = []
    for platform_name in platform_candidates:
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        try:
            platform, platform_properties = configure_openmm_platform(platform_name)
            candidate_simulation = app.Simulation(
                modeller.topology,
                system,
                integrator,
                platform,
                platformProperties=platform_properties,
            )
            candidate_simulation.context.setPositions(modeller.positions)
            candidate_simulation.context.getState(getEnergy=True)
            simulation = candidate_simulation
            logger.info(
                "  Using %s platform with properties %s",
                platform_name,
                platform_properties,
            )
            break
        except Exception as exc:
            platform_errors.append(f"{platform_name}: {exc}")
            logger.warning(f"  Platform {platform_name} failed: {exc}")
    if simulation is None:
        raise RuntimeError(
            "OpenMM platform initialization failed: " + " | ".join(platform_errors)
        )
    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    logger.info(f"  Initial energy: {initial_energy:.2f} kJ/mol")
    try:
        simulation.minimizeEnergy(
            tolerance=tolerance * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=max_iterations,
        )
        converged = True
    except Exception as e:
        raise RuntimeError(f"Energy minimization failed: {e}") from e
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    final_positions = state.getPositions()
    if not np.isfinite([initial_energy, final_energy]).all():
        raise RuntimeError("Non-finite energy during minimization")
    logger.info(f"  Final energy: {final_energy:.2f} kJ/mol")
    logger.info(f"  Energy reduction: {initial_energy - final_energy:.2f} kJ/mol")
    steps = max_iterations
    return (final_positions, initial_energy, final_energy, steps, converged)


def run_restrained_molecular_dynamics(
    system: openmm.System,
    modeller: app.Modeller,
    md_steps: int,
    temperature_kelvin: float = 300.0,
    timestep_femtoseconds: float = 1.0,
    friction_per_picosecond: float = 1.0,
    random_seed: int = 20260811,
    post_md_minimization_iterations: int = 250,
    tolerance: float = 10.0,
    preferred_platform: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List, float, float, float, int, bool]:
    if md_steps <= 0:
        raise ValueError(f"md_steps must be positive, got {md_steps}")
    if temperature_kelvin <= 0:
        raise ValueError("temperature_kelvin must be positive")
    if timestep_femtoseconds <= 0:
        raise ValueError("timestep_femtoseconds must be positive")
    if friction_per_picosecond <= 0:
        raise ValueError("friction_per_picosecond must be positive")
    if post_md_minimization_iterations < 0:
        raise ValueError("post_md_minimization_iterations cannot be negative")
    logger.info(
        "Running restrained molecular dynamics: %d steps, %.3f fs, %.1f K, seed=%d",
        md_steps,
        timestep_femtoseconds,
        temperature_kelvin,
        random_seed,
    )
    env_platform = os.environ.get("OPENMM_PLATFORM")
    platform_candidates: List[str] = []
    for candidate in (preferred_platform, env_platform, "CUDA", "OpenCL", "CPU"):
        if candidate is None:
            continue
        normalized = str(candidate).strip()
        if normalized and normalized not in platform_candidates:
            platform_candidates.append(normalized)
    simulation = None
    platform_errors = []
    for platform_name in platform_candidates:
        integrator = LangevinMiddleIntegrator(
            temperature_kelvin * unit.kelvin,
            friction_per_picosecond / unit.picosecond,
            timestep_femtoseconds / 1000.0 * unit.picoseconds,
        )
        integrator.setRandomNumberSeed(int(random_seed))
        try:
            platform, platform_properties = configure_openmm_platform(platform_name)
            candidate_simulation = app.Simulation(
                modeller.topology,
                system,
                integrator,
                platform,
                platformProperties=platform_properties,
            )
            candidate_simulation.context.setPositions(modeller.positions)
            candidate_simulation.context.getState(getEnergy=True)
            simulation = candidate_simulation
            logger.info(
                "  Using %s platform for dynamics with properties %s",
                platform_name,
                platform_properties,
            )
            break
        except Exception as exc:
            platform_errors.append(f"{platform_name}: {exc}")
            logger.warning("  Dynamics platform %s failed: %s", platform_name, exc)
    if simulation is None:
        raise RuntimeError(
            "OpenMM dynamics platform initialization failed: "
            + " | ".join(platform_errors)
        )
    state = simulation.context.getState(getEnergy=True)
    energy_before_md = state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    simulation.context.setVelocitiesToTemperature(
        temperature_kelvin * unit.kelvin, int(random_seed)
    )
    simulation.step(int(md_steps))
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy_after_md = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    final_positions = state.getPositions()
    post_md_converged = True
    if post_md_minimization_iterations > 0:
        try:
            simulation.minimizeEnergy(
                tolerance=tolerance * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=post_md_minimization_iterations,
            )
        except Exception as exc:
            raise RuntimeError(f"Post-MD minimization failed: {exc}") from exc
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        final_positions = state.getPositions()
    final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    if not np.isfinite([energy_before_md, energy_after_md, final_energy]).all():
        raise RuntimeError(
            f"Non-finite energy during restrained dynamics: before={energy_before_md}, after={energy_after_md}, final={final_energy}"
        )
    logger.info("  Potential energy before MD: %.2f kJ/mol", energy_before_md)
    logger.info("  Potential energy after MD: %.2f kJ/mol", energy_after_md)
    logger.info("  Final post-MD energy: %.2f kJ/mol", final_energy)
    return (
        final_positions,
        energy_before_md,
        energy_after_md,
        final_energy,
        int(md_steps),
        post_md_converged,
    )


def load_2d_structure(structure_path: Path) -> List[BasePair]:
    logger.info(f"Loading 2D structure: {structure_path}")
    with open(structure_path) as f:
        data = json.load(f)
    base_pairs = []
    sequence = data.get("sequence", "")
    index_base = int(data.get("index_base", 0))
    first_residue = int(data.get("first_residue", 1))
    if index_base not in (0, 1):
        raise ValueError(
            f"Unsupported 2D structure index_base={index_base} in {structure_path}; expected 0 or 1"
        )
    for bp_data in data.get("base_pairs", []):
        i_local = int(bp_data["i"]) - index_base
        j_local = int(bp_data["j"]) - index_base
        i = first_residue + i_local
        j = first_residue + j_local
        base_i = sequence[i_local] if 0 <= i_local < len(sequence) else "N"
        base_j = sequence[j_local] if 0 <= j_local < len(sequence) else "N"
        canonical = bp_data.get("canonical", True)
        base_pairs.append(
            BasePair(
                res_i=i, res_j=j, base_i=base_i, base_j=base_j, canonical=canonical
            )
        )
        logger.info(
            f"  Loaded {len(base_pairs)} base pairs ({sum((1 for bp in base_pairs if bp.canonical))} canonical, index_base={index_base}, first_residue={first_residue})"
        )
    return base_pairs


def write_pdb(
    topology: app.Topology,
    positions: List,
    output_path: Path,
    remarks: Optional[List[str]] = None,
):
    logger.info(f"Writing refined structure: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_topology, output_positions, removed_caps = strip_auxiliary_p5_caps(
        topology, positions
    )
    if removed_caps > 0:
        logger.info(
            f"  Removed {removed_caps} auxiliary P5 cap residues before PDB output"
        )
    with open(output_path, "w") as f:
        if remarks:
            for remark in remarks:
                f.write(f"REMARK   {remark}\n")
        app.PDBFile.writeFile(output_topology, output_positions, f, keepIds=True)
    logger.info(f"  Written: {output_path}")


def write_mmcif(topology: app.Topology, positions: List, output_path: Path):
    logger.info(f"Writing mmCIF: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_topology, output_positions, removed_caps = strip_auxiliary_p5_caps(
        topology, positions
    )
    if removed_caps > 0:
        logger.info(
            f"  Removed {removed_caps} auxiliary P5 cap residues before mmCIF output"
        )
    with open(output_path, "w") as f:
        app.PDBxFile.writeFile(output_topology, output_positions, f, keepIds=True)
    logger.info(f"  Written: {output_path}")


def refine_geometry(
    input_path: Path,
    output_path: Path,
    structure_2d_path: Optional[Path] = None,
    add_basepair_restraints: bool = False,
    predictions_path: Optional[Path] = None,
    predictions_meta_path: Optional[Path] = None,
    add_density_restraints_flag: bool = False,
    p_restraint_weight: float = DEFAULT_P_RESTRAINT_WEIGHT,
    hbond_restraint_weight: float = DEFAULT_HBOND_RESTRAINT_WEIGHT,
    hbond_tolerance: float = DEFAULT_HBOND_TOLERANCE,
    density_restraint_weight: float = DEFAULT_DENSITY_RESTRAINT_WEIGHT,
    density_search_radius: float = DEFAULT_DENSITY_SEARCH_RADIUS,
    max_iterations: int = 1000,
    md_steps: int = 2000,
    md_temperature_kelvin: float = 300.0,
    md_timestep_femtoseconds: float = 1.0,
    md_friction_per_picosecond: float = 1.0,
    md_random_seed: int = 20260811,
    post_md_minimization_iterations: int = 250,
    openmm_platform: Optional[str] = None,
    output_cif: Optional[Path] = None,
    verbose: bool = False,
    skip_terminal_fix: bool = False,
) -> RefinementResult:
    logger.info("=" * 60)
    logger.info("Phase 5: Geometry Refinement")
    logger.info("=" * 60)
    if skip_terminal_fix:
        logger.info("Skipping terminal fix (already processed)")
        working_pdb_path = input_path
    else:
        fixed_pdb_path = output_path.parent / f"{input_path.stem}_fixed_termini.pdb"
        working_pdb_path = fix_rna_termini(input_path, fixed_pdb_path, verbose=verbose)
    modeller = prepare_structure_with_openmm(working_pdb_path, verbose=verbose)
    system, p_atom_indices = create_system_with_restraints(
        modeller, p_restraint_weight=p_restraint_weight, verbose=verbose
    )
    n_hbond_restraints = 0
    if add_basepair_restraints and structure_2d_path is not None:
        base_pairs = load_2d_structure(structure_2d_path)
        n_hbond_restraints = add_basepair_hbond_restraints(
            system,
            modeller,
            base_pairs,
            hbond_weight=hbond_restraint_weight,
            tolerance=hbond_tolerance,
            verbose=verbose,
        )
    elif add_basepair_restraints:
        raise ValueError("Base-pair restraints require a secondary structure JSON")
    n_density_restraints = 0
    if (
        add_density_restraints_flag
        and predictions_path is not None
        and (predictions_meta_path is not None)
    ):
        predictions, density_meta = load_density_predictions(
            predictions_path, predictions_meta_path
        )
        n_density_restraints = add_density_restraints(
            system,
            modeller,
            predictions,
            density_meta,
            density_weight=density_restraint_weight,
            search_radius=density_search_radius,
            verbose=verbose,
        )
    elif add_density_restraints_flag:
        raise ValueError("Density restraints require predictions and metadata")
    minimized_positions, initial_energy, minimized_energy, steps, converged = (
        run_energy_minimization(
            system,
            modeller,
            max_iterations=max_iterations,
            preferred_platform=openmm_platform,
            verbose=verbose,
        )
    )
    final_positions = minimized_positions
    final_energy = minimized_energy
    md_steps_completed = 0
    md_energy_before = None
    md_energy_after = None
    post_md_converged = None
    if md_steps > 0:
        md_modeller = app.Modeller(modeller.topology, minimized_positions)
        (
            final_positions,
            md_energy_before,
            md_energy_after,
            final_energy,
            md_steps_completed,
            post_md_converged,
        ) = run_restrained_molecular_dynamics(
            system,
            md_modeller,
            md_steps=md_steps,
            temperature_kelvin=md_temperature_kelvin,
            timestep_femtoseconds=md_timestep_femtoseconds,
            friction_per_picosecond=md_friction_per_picosecond,
            random_seed=md_random_seed,
            post_md_minimization_iterations=post_md_minimization_iterations,
            preferred_platform=openmm_platform,
            verbose=verbose,
        )
        converged = converged and bool(post_md_converged)
    n_auxiliary_caps = sum(
        (
            1
            for chain in modeller.topology.chains()
            for residue in chain.residues()
            if residue.name == AUXILIARY_CAP_RESIDUE_NAME
        )
    )
    remarks = [
        "Generated by RNA Cryo-EM Pipeline Phase 5",
        f"Initial energy: {initial_energy:.2f} kJ/mol",
        f"Energy after minimization: {minimized_energy:.2f} kJ/mol",
        f"Final energy: {final_energy:.2f} kJ/mol",
        f"Restrained MD steps: {md_steps_completed}/{md_steps}",
        f"Hydrogen placement random seed: {HYDROGEN_PLACEMENT_RANDOM_SEED}",
        f"OpenMM deterministic forces: {OPENMM_DETERMINISTIC_FORCES}",
        f"OpenMM CPU threads: {int(os.environ.get('OPENMM_CPU_THREADS', '1'))}",
        f"P atom restraints: {len(p_atom_indices)} (k={p_restraint_weight})",
        f"H-bond restraints: {n_hbond_restraints}",
        f"Density restraints: {n_density_restraints} (k={density_restraint_weight})",
        f"Auxiliary caps: {n_auxiliary_caps} ({AUXILIARY_CAP_METHOD})",
    ]
    write_pdb(modeller.topology, final_positions, output_path, remarks=remarks)
    if output_cif is not None:
        write_mmcif(modeller.topology, final_positions, output_cif)
    result = RefinementResult(
        input_path=str(input_path),
        output_path=str(output_path),
        initial_energy=initial_energy,
        final_energy=final_energy,
        minimization_steps=steps,
        converged=converged,
        n_p_restraints=len(p_atom_indices),
        n_hbond_restraints=n_hbond_restraints,
        hydrogen_placement_random_seed=HYDROGEN_PLACEMENT_RANDOM_SEED,
        openmm_deterministic_forces=OPENMM_DETERMINISTIC_FORCES,
        openmm_cpu_threads=int(os.environ.get("OPENMM_CPU_THREADS", "1")),
        energy_after_minimization=minimized_energy,
        md_steps_requested=int(md_steps),
        md_steps_completed=md_steps_completed,
        md_temperature_kelvin=md_temperature_kelvin if md_steps > 0 else None,
        md_timestep_femtoseconds=md_timestep_femtoseconds if md_steps > 0 else None,
        md_friction_per_picosecond=md_friction_per_picosecond if md_steps > 0 else None,
        md_random_seed=md_random_seed if md_steps > 0 else None,
        md_potential_energy_before=md_energy_before,
        md_potential_energy_after=md_energy_after,
        post_md_minimization_iterations=post_md_minimization_iterations
        if md_steps > 0
        else 0,
        post_md_minimization_converged=post_md_converged,
        p_restraint_weight=p_restraint_weight,
        hbond_restraint_weight=hbond_restraint_weight,
        n_density_restraints=n_density_restraints,
        density_restraint_weight=density_restraint_weight,
        n_auxiliary_caps=n_auxiliary_caps,
        auxiliary_cap_method=AUXILIARY_CAP_METHOD if n_auxiliary_caps > 0 else None,
    )
    logger.info("=" * 60)
    logger.info("Refinement Summary")
    logger.info("=" * 60)
    logger.info(f"  Energy: {initial_energy:.2f} → {final_energy:.2f} kJ/mol")
    if md_steps > 0:
        logger.info(
            "  Restrained MD: %d steps (%.3f fs; %.3f ps total; seed=%d)",
            md_steps_completed,
            md_timestep_femtoseconds,
            md_steps_completed * md_timestep_femtoseconds / 1000.0,
            md_random_seed,
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Refine an RNA model with restrained AMBER14 RNA.OL3 dynamics"
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--structure-2d", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-meta", type=Path, required=True)
    parser.add_argument("--output-cif", type=Path)
    parser.add_argument("--openmm-platform")
    args = parser.parse_args()
    refine_geometry(
        input_path=args.input,
        output_path=args.output,
        structure_2d_path=args.structure_2d,
        add_basepair_restraints=True,
        predictions_path=args.predictions,
        predictions_meta_path=args.predictions_meta,
        add_density_restraints_flag=True,
        output_cif=args.output_cif,
        openmm_platform=args.openmm_platform,
    )


if __name__ == "__main__":
    main()
