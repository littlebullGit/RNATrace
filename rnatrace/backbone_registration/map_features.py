from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy import ndimage
SUGAR_TARGET_DISTANCE_A = 3.8
BASE_TARGET_DISTANCE_A = 8.0
SUGAR_DISTANCE_TOLERANCE_A = 2.5
BASE_DISTANCE_TOLERANCE_A = 3.5

@dataclass(frozen=True)
class FeatureComponent:
    label: str
    centroid: tuple[float, float, float]
    distance_to_p: float
    n_voxels: int

    def to_dict(self) -> dict:
        return {'label': self.label, 'centroid': [round(value, 3) for value in self.centroid], 'distance_to_p': round(self.distance_to_p, 3), 'n_voxels': int(self.n_voxels)}

@dataclass(frozen=True)
class MapGridTransform:
    origin_xyz: tuple[float, float, float]
    voxel_size: np.ndarray
    shape_zyx: tuple[int, int, int] | None = None

    def world_to_voxel_zyx(self, world_xyz: np.ndarray) -> np.ndarray:
        origin = np.asarray(self.origin_xyz, dtype=np.float64)
        voxel_xyz = (np.asarray(world_xyz, dtype=np.float64) - origin) / self.voxel_size
        return voxel_xyz[::-1]

    def voxel_zyx_to_world_xyz(self, voxel_zyx: np.ndarray) -> np.ndarray:
        origin = np.asarray(self.origin_xyz, dtype=np.float64)
        voxel_xyz = np.asarray(voxel_zyx, dtype=np.float64)[::-1]
        return origin + voxel_xyz * self.voxel_size

def _window_slices(center_vox: np.ndarray, radius_vox: int, shape: Sequence[int]) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    starts = np.clip(center_vox - radius_vox, 0, np.asarray(shape))
    stops = np.clip(center_vox + radius_vox + 1, 0, np.asarray(shape))
    slices = tuple((slice(int(start), int(stop)) for start, stop in zip(starts, stops)))
    return (slices, starts)

def _sphere_mask(shape: Sequence[int], starts: np.ndarray, center_float_vox: np.ndarray, radius_vox: float) -> np.ndarray:
    grids = np.ogrid[tuple((slice(0, int(size)) for size in shape))]
    dist2 = np.zeros(tuple((int(size) for size in shape)), dtype=np.float64)
    radii = np.broadcast_to(np.asarray(radius_vox), (3,))
    for axis, grid in enumerate(grids):
        coords = grid + starts[axis]
        delta = coords - center_float_vox[axis]
        dist2 += (delta / radii[axis]) ** 2
    return dist2 <= 1.0

def _components_for_labels(labels: np.ndarray, p_position: np.ndarray, *, label_values: set[int], label_name: str, apix: float, search_radius_a: float, min_voxels: int, map_grid: MapGridTransform | None=None) -> list[FeatureComponent]:
    if map_grid is not None:
        center_float_vox = map_grid.world_to_voxel_zyx(p_position)
        radius_float_vox = float(search_radius_a) / np.asarray(map_grid.voxel_size)[::-1]
    else:
        center_float_vox = p_position / float(apix)
        radius_float_vox = float(search_radius_a) / float(apix)
    center_vox = np.rint(center_float_vox).astype(np.int64)
    radius_vox = np.ceil(radius_float_vox).astype(np.int64)
    slices, starts = _window_slices(center_vox, radius_vox, labels.shape)
    window = labels[slices]
    if window.size == 0:
        return []
    mask = np.isin(window, list(label_values))
    mask &= _sphere_mask(window.shape, starts, center_float_vox, radius_float_vox)
    if not np.any(mask):
        return []
    structure = np.ones((3, 3, 3), dtype=np.int8)
    labeled, n_components = ndimage.label(mask, structure=structure)
    components: list[FeatureComponent] = []
    for component_id in range(1, n_components + 1):
        local_coords = np.argwhere(labeled == component_id)
        n_voxels = int(len(local_coords))
        if n_voxels < min_voxels:
            continue
        global_coords = local_coords.astype(np.float64) + starts[None, :]
        if map_grid is not None:
            centroid_array = map_grid.voxel_zyx_to_world_xyz(global_coords.mean(axis=0))
        else:
            centroid_array = global_coords.mean(axis=0) * float(apix)
        centroid = tuple((float(value) for value in centroid_array))
        distance = float(np.linalg.norm(centroid_array - p_position))
        components.append(FeatureComponent(label=label_name, centroid=centroid, distance_to_p=distance, n_voxels=n_voxels))
    components.sort(key=lambda item: (-item.n_voxels, item.distance_to_p))
    return components

def _choose_component(components: Sequence[FeatureComponent], *, target_distance_a: float) -> FeatureComponent | None:
    if not components:
        return None
    return max(components, key=lambda item: (-abs(item.distance_to_p - target_distance_a), np.log1p(item.n_voxels), -item.distance_to_p))

def _distance_score(distance: float | None, target: float, tolerance: float) -> float:
    if distance is None:
        return 0.0
    scaled = (float(distance) - float(target)) / float(tolerance)
    return float(np.exp(-(scaled * scaled)))

def feature_confidence(atom_features: dict) -> float:
    sugar = atom_features.get('sugar') or {}
    base = atom_features.get('base') or {}
    sugar_score = _distance_score(sugar.get('distance_to_p'), SUGAR_TARGET_DISTANCE_A, SUGAR_DISTANCE_TOLERANCE_A)
    base_score = _distance_score(base.get('distance_to_p'), BASE_TARGET_DISTANCE_A, BASE_DISTANCE_TOLERANCE_A)
    return round(0.5 * sugar_score + 0.5 * base_score, 4)
