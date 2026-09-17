from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from scipy import ndimage
from scipy.ndimage import watershed_ift
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
logger = logging.getLogger(__name__)
P_GROUP_LABEL = 1
SUGAR_GROUP_LABEL = 2
BASE_GROUP_LABEL = 3
IDEAL_PP_DISTANCE = 5.9
MIN_PP_DISTANCE = 3.5
MAX_PP_DISTANCE = 8.5
BACKBONE_RANGE = (4.0, 8.0)
MIN_CLUSTER_VOXELS = 10
SUGAR_CAP_RATIO = 2.0
SUGAR_CAP_MIN_PEAKS = 2

def voxel_to_world(voxel_zyx: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    return origin + voxel_zyx[::-1] * voxel_size

def world_to_voxel(world_xyz: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    vox_xyz = (world_xyz - origin) / voxel_size
    return vox_xyz[::-1].astype(int)

def find_p_clusters(labels: np.ndarray, min_cluster_voxels: int=MIN_CLUSTER_VOXELS) -> Tuple[np.ndarray, int, np.ndarray]:
    p_mask = labels == P_GROUP_LABEL
    n_p_voxels = int(np.sum(p_mask))
    logger.info(f'P-group voxels: {n_p_voxels}')
    if n_p_voxels == 0:
        return (np.zeros_like(labels, dtype=np.int32), 0, np.array([]))
    structure = ndimage.generate_binary_structure(3, 2)
    labeled_clusters, n_raw = ndimage.label(p_mask, structure=structure)
    cluster_volumes = ndimage.sum_labels(p_mask.astype(np.int32), labeled_clusters, range(1, n_raw + 1)).astype(np.int32)
    noise_ids = np.where(cluster_volumes < min_cluster_voxels)[0]
    for idx in noise_ids:
        cid = idx + 1
        labeled_clusters[labeled_clusters == cid] = 0
        cluster_volumes[idx] = 0
    n_valid = int(np.sum(cluster_volumes >= min_cluster_voxels))
    n_noise = len(noise_ids)
    logger.info(f'Clusters: {n_raw} raw, {n_noise} noise (<{min_cluster_voxels} vox), {n_valid} valid')
    return (labeled_clusters, n_valid, cluster_volumes)

def estimate_single_atom_volume(cluster_volumes: np.ndarray) -> float:
    valid = cluster_volumes[cluster_volumes >= MIN_CLUSTER_VOXELS]
    if len(valid) == 0:
        return 400.0
    vmin, vmax = (valid.min(), valid.max())
    n_bins = max(10, int((vmax - vmin) / 50))
    counts, edges = np.histogram(valid, bins=n_bins)
    peak_idx = np.argmax(counts)
    peak_centre = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0
    lower = peak_centre * 0.7
    upper = peak_centre * 1.3
    near_peak = valid[(valid >= lower) & (valid <= upper)]
    if len(near_peak) > 0:
        single_vol = float(np.median(near_peak))
    else:
        single_vol = float(peak_centre)
    logger.info(f'Estimated single-atom volume: {single_vol:.0f} voxels (histogram peak at {peak_centre:.0f}, {len(near_peak)} clusters in ±30% window)')
    return single_vol

def count_group_peaks_per_cluster(labels: np.ndarray, density: np.ndarray, origin: np.ndarray, voxel_size: float, labeled_p_clusters: np.ndarray, cluster_volumes: np.ndarray, group_label: int=SUGAR_GROUP_LABEL, group_name: str='sugar', min_separation: float=2.5) -> np.ndarray:
    n_clusters = len(cluster_volumes)
    group_peaks = np.zeros(n_clusters, dtype=int)
    group_mask = labels == group_label
    n_group_vox = int(np.sum(group_mask))
    if n_group_vox == 0:
        logger.info(f'No {group_name}-group voxels — skipping correction')
        return group_peaks
    group_density = np.where(group_mask, density, -np.inf)
    sep_vox = max(1, int(min_separation / voxel_size))
    filter_size = 2 * sep_vox + 1
    group_lm = ndimage.maximum_filter(group_density, size=filter_size)
    group_pk_mask = (group_density == group_lm) & group_mask & (density > 0)
    group_pk_coords = np.array(np.where(group_pk_mask)).T
    n_group_peaks = len(group_pk_coords)
    logger.info(f'{group_name.capitalize()} density maxima (min_sep={min_separation:.1f}Å, filter={filter_size}): {n_group_peaks}')
    if n_group_peaks == 0:
        return group_peaks
    group_pk_world = np.array([voxel_to_world(v, origin, voxel_size) for v in group_pk_coords])
    valid_cids = []
    centroids = []
    for cid_0 in range(n_clusters):
        if cluster_volumes[cid_0] < MIN_CLUSTER_VOXELS:
            continue
        cid = cid_0 + 1
        voxels = np.array(np.where(labeled_p_clusters == cid)).T
        dens = np.array([density[v[0], v[1], v[2]] for v in voxels])
        w = np.maximum(dens, 0.0)
        ws = w.sum()
        if ws > 0:
            c = (voxels * w[:, None]).sum(axis=0) / ws
        else:
            c = voxels.mean(axis=0)
        valid_cids.append(cid_0)
        centroids.append(voxel_to_world(c, origin, voxel_size))
    if len(centroids) == 0:
        return group_peaks
    centroid_arr = np.array(centroids)
    dists = cdist(group_pk_world, centroid_arr)
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(len(group_pk_world)), nearest_idx]
    for i, (ni, nd) in enumerate(zip(nearest_idx, nearest_dist)):
        if nd <= 10.0:
            group_peaks[valid_cids[ni]] += 1
    assigned = int(np.sum(nearest_dist <= 10.0))
    logger.info(f'{group_name.capitalize()} peaks assigned to P-clusters: {assigned}/{n_group_peaks}')
    return group_peaks

def allocate_atoms_to_clusters(cluster_volumes: np.ndarray, single_atom_volume: float, expected_atoms: int, sugar_peaks_per_cluster: Optional[np.ndarray]=None, base_peaks_per_cluster: Optional[np.ndarray]=None) -> np.ndarray:
    n_clusters = len(cluster_volumes)
    raw_alloc = np.zeros(n_clusters, dtype=np.float64)
    for i in range(n_clusters):
        vol = cluster_volumes[i]
        if vol < MIN_CLUSTER_VOXELS:
            raw_alloc[i] = 0.0
        else:
            raw_alloc[i] = vol / single_atom_volume
    allocation = np.round(raw_alloc).astype(int)
    for i in range(n_clusters):
        if cluster_volumes[i] >= MIN_CLUSTER_VOXELS and allocation[i] < 1:
            allocation[i] = 1
    total_before = int(allocation.sum())
    logger.info(f'Volume-only allocation: {total_before} atoms across {np.sum(allocation > 0)} clusters (target: {expected_atoms})')
    if sugar_peaks_per_cluster is not None:
        has_base = base_peaks_per_cluster is not None
        n_corrected = 0
        for i in range(n_clusters):
            if cluster_volumes[i] < MIN_CLUSTER_VOXELS:
                continue
            vol_est = allocation[i]
            sugar = int(sugar_peaks_per_cluster[i])
            base = int(base_peaks_per_cluster[i]) if has_base else -1
            if sugar >= SUGAR_CAP_MIN_PEAKS and vol_est >= SUGAR_CAP_RATIO * sugar:
                logger.debug(f'Cluster {i + 1}: vol_est={vol_est}, sugar={sugar}, base={base} → capped to {sugar}')
                allocation[i] = sugar
                n_corrected += 1
        total_after = int(allocation.sum())
        logger.info(f'Sugar+base corrected allocation: {total_after} atoms ({n_corrected} clusters adjusted, delta={total_after - total_before:+d})')
    allocation = _adjust_allocation(allocation, cluster_volumes, single_atom_volume, expected_atoms)
    logger.info(f'Final allocation: {allocation.sum()} atoms (max per cluster: {allocation.max()})')
    return allocation

def _adjust_allocation(allocation: np.ndarray, cluster_volumes: np.ndarray, single_atom_volume: float, expected_atoms: int) -> np.ndarray:
    alloc = allocation.copy()
    total = alloc.sum()
    if total == expected_atoms:
        return alloc
    frac = np.zeros(len(alloc), dtype=np.float64)
    for i in range(len(alloc)):
        if cluster_volumes[i] >= MIN_CLUSTER_VOXELS:
            exact = cluster_volumes[i] / single_atom_volume
            frac[i] = exact - int(exact)
    if total > expected_atoms:
        deficit = int(total - expected_atoms)
        candidates = np.where(alloc > 0)[0]
        order = sorted(candidates, key=lambda i: (frac[i], -cluster_volumes[i]))
        while deficit > 0:
            changed = False
            for idx in order:
                if deficit <= 0:
                    break
                if alloc[idx] > 1:
                    alloc[idx] -= 1
                    deficit -= 1
                    changed = True
            if not changed:
                break
        if deficit > 0:
            order2 = sorted(candidates, key=lambda i: cluster_volumes[i])
            for idx in order2:
                if deficit <= 0:
                    break
                if alloc[idx] > 0:
                    alloc[idx] -= 1
                    deficit -= 1
    elif total < expected_atoms:
        surplus = int(expected_atoms - total)
        candidates = np.where(alloc > 0)[0]
        if len(candidates) == 0:
            return alloc
        order = sorted(candidates, key=lambda i: (-frac[i], -cluster_volumes[i]))
        while surplus > 0:
            for idx in order:
                if surplus <= 0:
                    break
                alloc[idx] += 1
                surplus -= 1
        if surplus > 0:
            order2 = sorted(candidates, key=lambda i: -cluster_volumes[i])
            for idx in order2:
                if surplus <= 0:
                    break
                alloc[idx] += 1
                surplus -= 1
    return alloc

def find_peaks_in_cluster(cluster_mask: np.ndarray, density: np.ndarray, origin: np.ndarray, voxel_size: float, n_atoms: int, method: str='auto') -> List[np.ndarray]:
    voxels = np.array(np.where(cluster_mask)).T
    if len(voxels) == 0:
        return []
    densities = np.array([density[v[0], v[1], v[2]] for v in voxels])
    if n_atoms == 1:
        weights = np.maximum(densities, 0)
        w_sum = weights.sum()
        if w_sum > 0:
            centroid_zyx = (voxels * weights[:, None]).sum(axis=0) / w_sum
        else:
            centroid_zyx = voxels.mean(axis=0)
        world = voxel_to_world(centroid_zyx, origin, voxel_size)
        return [world]
    if method in ('auto', 'watershed'):
        try:
            ws_result = _watershed_peaks(cluster_mask, density, origin, voxel_size, n_atoms)
            if len(ws_result) >= n_atoms:
                return ws_result[:n_atoms]
            logger.debug(f'Watershed got {len(ws_result)}/{n_atoms}, falling back to iterative')
        except Exception as e:
            logger.debug(f'Watershed failed ({e}), falling back to iterative')
    positions = []
    exclusion_radius_vox = max(2, int(IDEAL_PP_DISTANCE * 0.45 / voxel_size))
    remaining_density = densities.copy()
    remaining_mask = np.ones(len(voxels), dtype=bool)
    for _ in range(n_atoms):
        active = np.where(remaining_mask)[0]
        if len(active) == 0:
            break
        best_local = active[np.argmax(remaining_density[active])]
        peak_zyx = voxels[best_local].astype(float)
        positions.append(voxel_to_world(peak_zyx, origin, voxel_size))
        dists_vox = np.sqrt(np.sum((voxels - peak_zyx) ** 2, axis=1))
        remaining_mask[dists_vox < exclusion_radius_vox] = False
    if len(positions) >= n_atoms:
        return positions[:n_atoms]
    logger.debug(f'Iterative peak finding got {len(positions)}/{n_atoms}, falling back to k-means')
    return _kmeans_peaks(voxels, densities, origin, voxel_size, n_atoms)

def _kmeans_peaks(voxels: np.ndarray, densities: np.ndarray, origin: np.ndarray, voxel_size: float, n_atoms: int, max_iter: int=50) -> List[np.ndarray]:
    weights = np.maximum(densities, 0.0)
    w_sum = weights.sum()
    if w_sum == 0:
        weights = np.ones(len(densities))
        w_sum = weights.sum()
    voxels_f = voxels.astype(np.float64)
    order = np.argsort(-densities)
    centres = [voxels_f[order[0]]]
    min_sep_vox = max(2, int(MIN_PP_DISTANCE / voxel_size))
    for idx in order[1:]:
        if len(centres) >= n_atoms:
            break
        candidate = voxels_f[idx]
        dists = [np.linalg.norm(candidate - c) for c in centres]
        if min(dists) >= min_sep_vox:
            centres.append(candidate)
    while len(centres) < n_atoms:
        next_idx = len(centres)
        if next_idx < len(order):
            centres.append(voxels_f[order[next_idx]])
        else:
            centres.append(centres[-1] + np.random.randn(3) * 0.5)
    centres = np.array(centres)
    for _ in range(max_iter):
        dists = cdist(voxels_f, centres)
        assignments = np.argmin(dists, axis=1)
        new_centres = np.zeros_like(centres)
        for k in range(n_atoms):
            mask = assignments == k
            if mask.sum() == 0:
                new_centres[k] = centres[k]
            else:
                w = weights[mask]
                new_centres[k] = (voxels_f[mask] * w[:, None]).sum(0) / w.sum()
        shift = np.linalg.norm(new_centres - centres, axis=1).max()
        centres = new_centres
        if shift < 0.1:
            break
    return [voxel_to_world(c, origin, voxel_size) for c in centres]

def _watershed_peaks(cluster_mask: np.ndarray, density: np.ndarray, origin: np.ndarray, voxel_size: float, n_atoms: int) -> List[np.ndarray]:
    coords = np.array(np.where(cluster_mask))
    z_min, z_max = (int(coords[0].min()), int(coords[0].max()))
    y_min, y_max = (int(coords[1].min()), int(coords[1].max()))
    x_min, x_max = (int(coords[2].min()), int(coords[2].max()))
    pad = 1
    sl = (slice(max(0, z_min - pad), min(density.shape[0], z_max + pad + 1)), slice(max(0, y_min - pad), min(density.shape[1], y_max + pad + 1)), slice(max(0, x_min - pad), min(density.shape[2], x_max + pad + 1)))
    offset = np.array([sl[0].start, sl[1].start, sl[2].start])
    sub_density = density[sl].copy()
    sub_mask = cluster_mask[sl].copy()
    sub_voxels = np.array(np.where(sub_mask)).T
    if len(sub_voxels) < n_atoms:
        return []
    sub_dens = np.array([sub_density[v[0], v[1], v[2]] for v in sub_voxels])
    min_sep_vox = max(3, int(MIN_PP_DISTANCE / voxel_size))
    remaining_mask = np.ones(len(sub_voxels), dtype=bool)
    seeds: List[np.ndarray] = []
    for _ in range(n_atoms):
        active = np.where(remaining_mask)[0]
        if len(active) == 0:
            break
        best = active[np.argmax(sub_dens[active])]
        seed = sub_voxels[best].copy()
        seeds.append(seed)
        dists = np.sqrt(np.sum((sub_voxels - seed) ** 2, axis=1))
        remaining_mask[dists < min_sep_vox] = False
    if len(seeds) < n_atoms:
        return []
    markers = np.zeros(sub_density.shape, dtype=np.int32)
    for i, seed in enumerate(seeds):
        markers[seed[0], seed[1], seed[2]] = i + 1
    max_d = float(sub_density[sub_mask].max())
    if max_d <= 0:
        return []
    inv = np.full(sub_density.shape, np.uint16(65535), dtype=np.uint16)
    mask_vals = sub_density[sub_mask].astype(np.float64)
    inv[sub_mask] = ((max_d - mask_vals) / max_d * 60000).astype(np.uint16)
    structure = ndimage.generate_binary_structure(3, 2)
    ws_labels = watershed_ift(inv, markers, structure=structure)
    ws_labels[~sub_mask] = 0
    positions: List[np.ndarray] = []
    for label_id in range(1, n_atoms + 1):
        basin = (ws_labels == label_id) & sub_mask
        bv = np.array(np.where(basin)).T
        if len(bv) == 0:
            seed_global = seeds[label_id - 1].astype(float) + offset
            positions.append(voxel_to_world(seed_global, origin, voxel_size))
            continue
        bd = np.array([sub_density[v[0], v[1], v[2]] for v in bv])
        w = np.maximum(bd, 0.0)
        w_sum = w.sum()
        if w_sum > 0:
            centroid = (bv.astype(float) * w[:, None]).sum(0) / w_sum
        else:
            centroid = bv.astype(float).mean(0)
        centroid_global = centroid + offset
        positions.append(voxel_to_world(centroid_global, origin, voxel_size))
    if len(positions) >= 2:
        pos_arr = np.array(positions)
        pair_D = cdist(pos_arr, pos_arr)
        np.fill_diagonal(pair_D, np.inf)
        if pair_D.min() < MIN_PP_DISTANCE:
            logger.debug(f'Watershed centroids too close ({pair_D.min():.1f} Å < {MIN_PP_DISTANCE} Å) — falling back to iterative')
            return []
    return positions

def extract_p_atoms_volume_guided(labels: np.ndarray, density: np.ndarray, origin: np.ndarray, voxel_size: float, expected_atoms: int, use_watershed: bool=True, refine: bool=True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    labeled_clusters, n_valid, cluster_volumes = find_p_clusters(labels)
    if n_valid == 0:
        logger.error('No valid P-group clusters found!')
        return ([], {'error': 'no_clusters'})
    histogram_single_vol = estimate_single_atom_volume(cluster_volumes)
    valid_p_voxels = int(cluster_volumes.sum())
    single_vol = float(valid_p_voxels) / float(expected_atoms) if expected_atoms > 0 and valid_p_voxels > 0 else histogram_single_vol
    logger.info('Sequence-calibrated single-atom volume: %.1f voxels (histogram estimate %.1f, valid P voxels=%d, expected atoms=%d)', single_vol, histogram_single_vol, valid_p_voxels, expected_atoms)
    sugar_peaks = count_group_peaks_per_cluster(labels, density, origin, voxel_size, labeled_clusters, cluster_volumes, group_label=SUGAR_GROUP_LABEL, group_name='sugar')
    base_peaks = count_group_peaks_per_cluster(labels, density, origin, voxel_size, labeled_clusters, cluster_volumes, group_label=BASE_GROUP_LABEL, group_name='base')
    allocation = allocate_atoms_to_clusters(cluster_volumes, single_vol, expected_atoms, sugar_peaks_per_cluster=sugar_peaks, base_peaks_per_cluster=base_peaks)
    all_atoms: List[Dict[str, Any]] = []
    n_clusters = len(cluster_volumes)
    for cid_0 in range(n_clusters):
        cid = cid_0 + 1
        n_atoms = allocation[cid_0]
        if n_atoms <= 0:
            continue
        cluster_mask = labeled_clusters == cid
        peak_method = 'auto' if use_watershed else 'iterative'
        peaks = find_peaks_in_cluster(cluster_mask, density, origin, voxel_size, n_atoms, method=peak_method)
        for pos in peaks:
            vox_zyx = world_to_voxel(pos, origin, voxel_size)
            z, y, x = vox_zyx
            dens_val = 0.0
            if 0 <= z < density.shape[0] and 0 <= y < density.shape[1] and (0 <= x < density.shape[2]):
                dens_val = float(density[z, y, x])
            all_atoms.append({'position': pos.tolist(), 'density': dens_val, 'cluster_id': int(cid), 'cluster_volume': int(cluster_volumes[cid_0]), 'cluster_allocation': int(n_atoms)})
    logger.info(f'Extracted {len(all_atoms)} P atoms from {n_valid} clusters')
    fallback_info: Dict[str, Any] = {'selected': False}
    if len(all_atoms) == expected_atoms and expected_atoms > 1:
        initial_positions = np.asarray([atom['position'] for atom in all_atoms], dtype=np.float64)
        initial_distances = cdist(initial_positions, initial_positions)
        np.fill_diagonal(initial_distances, np.inf)
        initial_min_nn = float(initial_distances.min())
        if initial_min_nn < 2.5:
            candidates, candidate_densities = _find_p_mask_candidates(labels, density, origin, voxel_size, min_separation=2.5)
            if len(candidates) >= expected_atoms:
                selected_positions = candidates[:expected_atoms]
                selected_densities = candidate_densities[:expected_atoms]
                selected_distances = cdist(selected_positions, selected_positions)
                np.fill_diagonal(selected_distances, np.inf)
                selected_min_nn = float(selected_distances.min())
                if selected_min_nn >= 2.5 - 1e-06:
                    replacement: List[Dict[str, Any]] = []
                    for position, density_value in zip(selected_positions, selected_densities):
                        voxel = world_to_voxel(position, origin, voxel_size)
                        z, y, x = (int(value) for value in voxel)
                        cluster_id = int(labeled_clusters[z, y, x]) if 0 <= z < labeled_clusters.shape[0] and 0 <= y < labeled_clusters.shape[1] and (0 <= x < labeled_clusters.shape[2]) else 0
                        replacement.append({'position': position.tolist(), 'density': float(density_value), 'cluster_id': cluster_id, 'cluster_volume': int(cluster_volumes[cluster_id - 1]) if cluster_id > 0 else 0, 'cluster_allocation': 1, 'strict_global_nms': True})
                    all_atoms = replacement
                    fallback_info = {'selected': True, 'reason': 'allocated_cloud_min_separation_below_2.5A', 'initial_min_nn_distance': initial_min_nn, 'candidate_count': int(len(candidates)), 'selected_count': int(expected_atoms), 'selected_min_nn_distance': selected_min_nn, 'min_separation': 2.5}
                    logger.warning('Replaced colliding allocated P cloud with %d strict global-NMS candidates (min NN %.3f -> %.3f Å)', expected_atoms, initial_min_nn, selected_min_nn)
            if not fallback_info['selected']:
                fallback_info = {'selected': False, 'reason': 'insufficient_strict_global_nms_candidates', 'initial_min_nn_distance': initial_min_nn, 'candidate_count': int(len(candidates)), 'required_count': int(expected_atoms), 'min_separation': 2.5}
    all_atoms, connectivity_info = _ensure_connectivity(all_atoms, expected_atoms)
    refine_info: Dict[str, Any] = {}
    if refine:
        all_atoms, refine_info = _refine_atom_positions(all_atoms, labels, density, origin, voxel_size)
    all_atoms, connectivity_info = _ensure_connectivity(all_atoms, expected_atoms)
    metadata = _compute_stats(all_atoms, single_vol, allocation, cluster_volumes, connectivity_info)
    if refine_info:
        metadata['refinement'] = refine_info
    metadata['strict_global_nms_fallback'] = fallback_info
    metadata['histogram_single_atom_volume'] = histogram_single_vol
    metadata['sequence_calibrated_single_atom_volume'] = single_vol
    return (all_atoms, metadata)

def _ensure_connectivity(atoms: List[Dict[str, Any]], expected_atoms: int, max_threshold: float=10.0) -> Tuple[List[Dict[str, Any]], Dict]:
    if len(atoms) < 2:
        return (atoms, {'connected': True, 'threshold': MAX_PP_DISTANCE})
    positions = np.array([a['position'] for a in atoms])
    D = cdist(positions, positions)
    for threshold in np.arange(MAX_PP_DISTANCE, max_threshold + 0.5, 0.5):
        adj = D <= threshold
        n = len(positions)
        visited = np.zeros(n, dtype=bool)
        components = []
        for start in range(n):
            if visited[start]:
                continue
            comp = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                comp.append(node)
                for nb in range(n):
                    if not visited[nb] and adj[node, nb]:
                        visited[nb] = True
                        queue.append(nb)
            components.append(comp)
        if len(components) == 1:
            logger.info(f'All {len(atoms)} atoms connected at {threshold:.1f}Å')
            return (atoms, {'connected': True, 'threshold': float(threshold), 'n_components': 1})
    comp_sizes = sorted([len(c) for c in components], reverse=True)
    logger.warning(f'Not fully connected at {max_threshold:.1f}Å: {len(components)} components, sizes={comp_sizes[:5]}')
    return (atoms, {'connected': False, 'threshold': float(max_threshold), 'n_components': len(components), 'component_sizes': comp_sizes})

def _find_p_mask_candidates(labels: np.ndarray, density: np.ndarray, origin: np.ndarray, voxel_size: float, min_separation: float=3.0) -> Tuple[np.ndarray, np.ndarray]:
    p_mask = labels == P_GROUP_LABEL
    p_density = np.where(p_mask, density, -np.inf)
    sep_vox = max(1, int(min_separation / voxel_size))
    filter_size = 2 * sep_vox + 1
    local_max = ndimage.maximum_filter(p_density, size=filter_size)
    peak_mask = (p_density == local_max) & p_mask
    peak_coords = np.array(np.where(peak_mask)).T
    if len(peak_coords) == 0:
        return (np.empty((0, 3)), np.empty(0))
    world_pos = np.array([voxel_to_world(v, origin, voxel_size) for v in peak_coords])
    peak_dens = np.array([density[v[0], v[1], v[2]] for v in peak_coords])
    order = np.lexsort((peak_coords[:, 2], peak_coords[:, 1], peak_coords[:, 0], -peak_dens))
    world_pos = world_pos[order]
    peak_dens = peak_dens[order]
    selected_indices: List[int] = []
    min_separation_sq = float(min_separation) ** 2
    for index, position in enumerate(world_pos):
        if selected_indices:
            selected = world_pos[np.asarray(selected_indices, dtype=int)]
            distance_sq = np.sum((selected - position) ** 2, axis=1)
            if np.any(distance_sq < min_separation_sq - 1e-09):
                continue
        selected_indices.append(index)
    selected = np.asarray(selected_indices, dtype=int)
    return (world_pos[selected], peak_dens[selected])

def _connected_components_for_positions(positions: np.ndarray, threshold: float) -> List[List[int]]:
    if len(positions) == 0:
        return []
    tree = cKDTree(positions)
    neighbours = tree.query_ball_tree(tree, r=float(threshold))
    seen: set[int] = set()
    components: List[List[int]] = []
    for start in range(len(positions)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: List[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in neighbours[current]:
                neighbour = int(neighbour)
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return sorted(components, key=len, reverse=True)

def _refine_atom_positions(atoms: List[Dict[str, Any]], labels: np.ndarray, density: np.ndarray, origin: np.ndarray, voxel_size: float, max_swaps: int=40) -> Tuple[List[Dict[str, Any]], Dict]:
    if len(atoms) < 10:
        return (atoms, {'refinement': 'skipped', 'reason': 'too_few_atoms'})
    candidates, cand_dens = _find_p_mask_candidates(labels, density, origin, voxel_size, min_separation=3.0)
    if len(candidates) == 0:
        return (atoms, {'refinement': 'skipped', 'reason': 'no_candidates'})
    positions = np.array([a['position'] for a in atoms])
    n_atoms = len(positions)
    dist_c2a = cdist(candidates, positions)
    vacancy_mask = dist_c2a.min(axis=1) > 3.0
    vac_pos = candidates[vacancy_mask]
    vac_dens = cand_dens[vacancy_mask]
    logger.info(f'Refinement: {len(candidates)} P-mask candidates, {len(vac_pos)} vacancies (>3 Å from atoms)')
    if len(vac_pos) == 0:
        return (atoms, {'refinement': 'skipped', 'reason': 'no_vacancies'})
    vac_to_atoms = cdist(vac_pos, positions)
    vac_bb = np.array([int(np.sum((vac_to_atoms[i] >= 4.0) & (vac_to_atoms[i] <= 9.5))) for i in range(len(vac_pos))])
    good_vac = vac_bb >= 1
    if not good_vac.any():
        logger.info('No vacancies with backbone potential — skipping refinement')
        return (atoms, {'refinement': 'skipped', 'reason': 'no_backbone_vacancies'})
    D = cdist(positions, positions)
    np.fill_diagonal(D, np.inf)
    adj = D <= 9.5
    comp_label = np.full(n_atoms, -1, dtype=int)
    comp_id = 0
    comp_sizes: Dict[int, int] = {}
    for start in range(n_atoms):
        if comp_label[start] >= 0:
            continue
        queue = [start]
        comp_label[start] = comp_id
        cnt = 0
        while queue:
            node = queue.pop(0)
            cnt += 1
            for nb in range(n_atoms):
                if comp_label[nb] < 0 and adj[node, nb]:
                    comp_label[nb] = comp_id
                    queue.append(nb)
        comp_sizes[comp_id] = cnt
        comp_id += 1
    main_comp = max(comp_sizes, key=comp_sizes.get)
    main_size = comp_sizes[main_comp]
    small_atoms = [i for i in range(n_atoms) if comp_label[i] != main_comp]
    logger.info(f'Refinement: {len(comp_sizes)} components, main={main_size}, small={len(small_atoms)} atoms')
    atoms_list = list(atoms)
    cur_pos = positions.copy()
    n_swapped = 0
    vac_used = np.zeros(len(vac_pos), dtype=bool)
    phase0 = 0
    while n_swapped < max_swaps:
        overlap_distances = cdist(cur_pos, cur_pos)
        np.fill_diagonal(overlap_distances, np.inf)
        overlap_pairs = np.argwhere(overlap_distances < 2.5 - 1e-09)
        overlap_pairs = overlap_pairs[overlap_pairs[:, 0] < overlap_pairs[:, 1]]
        if len(overlap_pairs) == 0:
            break
        left, right = (int(value) for value in overlap_pairs[0])
        replace_order = sorted((left, right), key=lambda index: (float(atoms_list[index].get('density', 0.0)), -index))
        components_before = len(_connected_components_for_positions(cur_pos, 9.5))
        best_ai, best_vi, best_score = (-1, -1, -np.inf)
        for ai in replace_order:
            for vi in range(len(vac_pos)):
                if vac_used[vi] or not good_vac[vi]:
                    continue
                d_all = cdist(vac_pos[vi:vi + 1], cur_pos)[0]
                d_all[ai] = np.inf
                if d_all.min() < 3.0 - 1e-09:
                    continue
                trial_positions = cur_pos.copy()
                trial_positions[ai] = vac_pos[vi]
                if len(_connected_components_for_positions(trial_positions, 9.5)) > components_before:
                    continue
                score = float(vac_dens[vi]) * (1.0 + 0.3 * vac_bb[vi])
                if score > best_score:
                    best_ai, best_vi, best_score = (ai, vi, score)
        if best_ai < 0:
            logger.warning('Could not repair overlapping P pair (%d, %d; %.3f A)', left, right, overlap_distances[left, right])
            break
        atoms_list[best_ai] = {**atoms_list[best_ai], 'position': vac_pos[best_vi].tolist(), 'density': float(vac_dens[best_vi]), 'refined': True, 'refine_phase': 0, 'refinement_reason': 'overlap_repair'}
        cur_pos[best_ai] = vac_pos[best_vi]
        vac_used[best_vi] = True
        n_swapped += 1
        phase0 += 1
    logger.info(f'Phase 0 (overlap repair): {phase0} atoms relocated')
    small_sorted = sorted(small_atoms, key=lambda i: atoms_list[i].get('density', 0))
    for ai in small_sorted:
        if n_swapped >= max_swaps:
            break
        best_vi, best_score = (-1, -np.inf)
        for vi in range(len(vac_pos)):
            if vac_used[vi] or not good_vac[vi]:
                continue
            d_to_main = cdist(vac_pos[vi:vi + 1], cur_pos[comp_label == main_comp])[0]
            if d_to_main.min() > 9.5:
                continue
            d_all = cdist(vac_pos[vi:vi + 1], cur_pos)[0]
            d_all[ai] = np.inf
            if d_all.min() < 3.0:
                continue
            score = float(vac_dens[vi]) * (1.0 + 0.3 * vac_bb[vi])
            if score > best_score:
                best_score = score
                best_vi = vi
        if best_vi >= 0:
            new_pos = vac_pos[best_vi].tolist()
            new_d = float(vac_dens[best_vi])
            logger.debug(f"Phase1: atom {ai} (dens={atoms_list[ai].get('density', 0):.3f}, comp={comp_label[ai]}) → vacancy (dens={new_d:.3f}, bb={vac_bb[best_vi]})")
            atoms_list[ai] = {**atoms_list[ai], 'position': new_pos, 'density': new_d, 'refined': True, 'refine_phase': 1}
            cur_pos[ai] = vac_pos[best_vi]
            vac_used[best_vi] = True
            n_swapped += 1
            comp_label[ai] = main_comp
    phase1 = n_swapped - phase0
    logger.info(f'Phase 1 (component repair): {phase1} atoms relocated')
    D2 = cdist(cur_pos, cur_pos)
    np.fill_diagonal(D2, np.inf)
    atom_dens = np.array([a.get('density', 0) for a in atoms_list])
    inter_bb = np.zeros(n_atoms, dtype=int)
    for i in range(n_atoms):
        my_cid = atoms_list[i].get('cluster_id', -1)
        for j in range(n_atoms):
            if i != j and atoms_list[j].get('cluster_id', -1) != my_cid:
                if 4.0 <= D2[i, j] <= 9.5:
                    inter_bb[i] += 1
    atom_scores = atom_dens * (0.5 + 0.5 * np.minimum(inter_bb, 3).astype(float) / 3.0)
    remaining_vac = np.where(~vac_used & good_vac)[0]
    if len(remaining_vac) > 0:
        rv_to_cur = cdist(vac_pos[remaining_vac], cur_pos)
        rv_bb = np.array([int(np.sum((rv_to_cur[k] >= 4.0) & (rv_to_cur[k] <= 9.5))) for k in range(len(remaining_vac))])
        rv_scores = vac_dens[remaining_vac] * (1.0 + 0.3 * rv_bb)
        vac_order = np.argsort(-rv_scores)
        atom_order = np.argsort(atom_scores)
        max_phase2 = 5
        phase2_done = 0
        a_ptr, v_ptr = (0, 0)
        while n_swapped < max_swaps and phase2_done < max_phase2 and (a_ptr < len(atom_order)) and (v_ptr < len(vac_order)):
            ai = atom_order[a_ptr]
            vi_loc = vac_order[v_ptr]
            vi = remaining_vac[vi_loc]
            if atoms_list[ai].get('refined', False):
                a_ptr += 1
                continue
            if inter_bb[ai] >= 1:
                a_ptr += 1
                continue
            if rv_scores[vi_loc] <= atom_scores[ai] * 2.0:
                break
            d_all = cdist(vac_pos[vi:vi + 1], cur_pos)[0]
            d_all[ai] = np.inf
            if d_all.min() < 3.0:
                v_ptr += 1
                continue
            new_d = float(vac_dens[vi])
            logger.debug(f'Phase2: atom {ai} (dens={atom_dens[ai]:.3f}, inter_bb={inter_bb[ai]}, score={atom_scores[ai]:.3f}) → vacancy (dens={new_d:.3f}, bb={rv_bb[vi_loc]}, score={rv_scores[vi_loc]:.3f})')
            atoms_list[ai] = {**atoms_list[ai], 'position': vac_pos[vi].tolist(), 'density': new_d, 'refined': True, 'refine_phase': 2}
            cur_pos[ai] = vac_pos[vi]
            n_swapped += 1
            phase2_done += 1
            a_ptr += 1
            v_ptr += 1
    phase2 = n_swapped - phase0 - phase1
    logger.info(f'Phase 2 (confidence swap): {phase2} atoms swapped')
    logger.info(f'Refinement total: {n_swapped} atoms modified')
    info = {'refinement': 'completed', 'phase0_swaps': phase0, 'phase1_swaps': phase1, 'phase2_swaps': phase2, 'total_swaps': n_swapped, 'vacancies_found': int(len(vac_pos)), 'vacancies_with_backbone': int(good_vac.sum()), 'candidates_total': len(candidates)}
    return (atoms_list, info)

def _compute_stats(atoms: List[Dict[str, Any]], single_vol: float, allocation: np.ndarray, cluster_volumes: np.ndarray, connectivity_info: Dict) -> Dict[str, Any]:
    if len(atoms) < 2:
        return {'n_atoms': len(atoms)}
    positions = np.array([a['position'] for a in atoms])
    D = cdist(positions, positions)
    np.fill_diagonal(D, np.inf)
    nn_dists = D.min(axis=1)
    return {'n_atoms': len(atoms), 'method': 'volume_guided', 'single_atom_volume': float(single_vol), 'n_clusters_used': int(np.sum(allocation > 0)), 'max_allocation': int(allocation.max()), 'connectivity': connectivity_info, 'mean_nn_distance': float(np.mean(nn_dists)), 'min_nn_distance': float(np.min(nn_dists)), 'max_nn_distance': float(np.max(nn_dists)), 'nn_in_backbone_range': int(np.sum((nn_dists >= BACKBONE_RANGE[0]) & (nn_dists <= BACKBONE_RANGE[1]))), 'nn_below_2A': int(np.sum(nn_dists < 2.0))}

def extract(labels: np.ndarray, density: np.ndarray, origin: np.ndarray, voxel_size: float, expected_atoms: int) -> np.ndarray:
    labels = np.asarray(labels)
    density = np.asarray(density, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float64)
    if labels.ndim != 3 or not labels.size or density.shape != labels.shape:
        raise ValueError('Labels and density must be matching nonempty three-dimensional arrays')
    if not np.isfinite(density).all() or not np.isin(labels, [0, 1, 2, 3, 4]).all():
        raise ValueError('Density must be finite and labels must be integers from zero to four')
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Origin must contain three finite XYZ coordinates')
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('Voxel size must be positive and finite')
    if isinstance(expected_atoms, bool) or int(expected_atoms) != expected_atoms or expected_atoms < 1:
        raise ValueError('Expected phosphate count must be a positive integer')
    atoms, metadata = extract_p_atoms_volume_guided(labels, density, origin, float(voxel_size), int(expected_atoms))
    if metadata.get('error') == 'no_clusters':
        raise ValueError('No phosphate component contains at least ten voxels')
    positions = np.asarray([atom['position'] for atom in atoms], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or (not len(positions)) or (not np.isfinite(positions).all()):
        raise ValueError('Extraction produced no finite phosphate coordinates')
    if len(positions) > 1:
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)
        if distances.min() < 2.5 - 1e-06:
            raise ValueError('Predicted phosphate support cannot produce the required sites at 2.5 A separation')
    return positions
