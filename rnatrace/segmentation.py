from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import SwinUNETR
NUCLEOTIDE_GROUP_N_CLASSES = 4
DEFAULT_FEATURE_SIZE = 48
DEFAULT_BINARY_THRESHOLD = 0.5

def iter_patch_indices(shape_zyx: Tuple[int, int, int], patch_size: int, overlap: int):
    step = int(patch_size - overlap)
    if step <= 0:
        raise ValueError(f'overlap must be smaller than patch_size, got overlap={overlap} patch_size={patch_size}')
    z, y, x = shape_zyx
    nz = int(np.ceil((z - patch_size) / step)) + 1 if z > patch_size else 1
    ny = int(np.ceil((y - patch_size) / step)) + 1 if y > patch_size else 1
    nx = int(np.ceil((x - patch_size) / step)) + 1 if x > patch_size else 1
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                ib = step * i
                ie = min(ib + patch_size, z)
                jb = step * j
                je = min(jb + patch_size, y)
                kb = step * k
                ke = min(kb + patch_size, x)
                yield (ib, ie, jb, je, kb, ke)

def _pad_patch_3d(patch: np.ndarray, patch_size: int) -> np.ndarray:
    dz, dy, dx = patch.shape
    if dz > patch_size or dy > patch_size or dx > patch_size:
        patch = patch[:patch_size, :patch_size, :patch_size]
        dz, dy, dx = patch.shape
    pad_z = patch_size - dz
    pad_y = patch_size - dy
    pad_x = patch_size - dx
    if pad_z == 0 and pad_y == 0 and (pad_x == 0):
        return patch
    return np.pad(patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

def extract_patch(data_zyx: np.ndarray, index_zyx: Tuple[int, int, int, int, int, int], patch_size: int) -> np.ndarray:
    ib, ie, jb, je, kb, ke = index_zyx
    patch = data_zyx[ib:ie, jb:je, kb:ke]
    return _pad_patch_3d(patch, patch_size)

@torch.no_grad()
def predict_binary_probability_full_map(model: nn.Module, data_zyx: np.ndarray, *, patch_size: int, overlap: int, device: torch.device) -> np.ndarray:
    shape = data_zyx.shape
    best_certainty = np.full(shape, -np.inf, dtype=np.float32)
    best_prob = np.zeros(shape, dtype=np.float32)
    model.eval()
    for idx in iter_patch_indices(shape, int(patch_size), int(overlap)):
        ib, ie, jb, je, kb, ke = idx
        dz, dy, dx = (ie - ib, je - jb, ke - kb)
        patch = extract_patch(data_zyx, idx, int(patch_size)).astype(np.float32)
        x = torch.from_numpy(patch[None, None, ...]).to(device)
        logits = model(x)
        if logits.ndim != 5 or int(logits.shape[1]) != 1:
            raise RuntimeError(f'Expected binary logits shape (B,1,D,H,W), got {tuple((int(v) for v in logits.shape))}')
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f'binary model produced nonfinite logits in patch {idx}')
        probs = torch.sigmoid(logits[:, 0])
        p_np = probs[0].detach().cpu().numpy().astype(np.float32)
        p_np = p_np[:dz, :dy, :dx]
        certainty = np.abs(p_np - 0.5)
        cur_cert = best_certainty[ib:ie, jb:je, kb:ke]
        cur_prob = best_prob[ib:ie, jb:je, kb:ke]
        mask = certainty > cur_cert
        cur_cert[mask] = certainty[mask]
        cur_prob[mask] = p_np[mask]
        best_certainty[ib:ie, jb:je, kb:ke] = cur_cert
        best_prob[ib:ie, jb:je, kb:ke] = cur_prob
    return best_prob

def resolve_feature_size(hparams: Dict[str, object], explicit: Optional[int], default: int=DEFAULT_FEATURE_SIZE) -> int:
    if explicit is not None:
        return int(explicit)
    value = hparams.get('feature_size')
    if value is not None:
        return int(value)
    return int(default)

def resolve_use_checkpoint(hparams: Dict[str, object], explicit: Optional[bool], default: bool=False) -> bool:
    if explicit is not None:
        return bool(explicit)
    value = hparams.get('use_checkpoint')
    if value is not None:
        return bool(value)
    return bool(default)

def resolve_binary_threshold(hparams: Dict[str, object], explicit: Optional[float]) -> float:
    if explicit is not None:
        return float(explicit)
    value = hparams.get('threshold')
    if value is not None:
        return float(value)
    return DEFAULT_BINARY_THRESHOLD

def combine_binary_and_group_labels(binary_prob_zyx: np.ndarray, group_label_zyx: np.ndarray, *, binary_threshold: float) -> np.ndarray:
    if binary_prob_zyx.shape != group_label_zyx.shape:
        raise RuntimeError(f'binary/group shape mismatch: binary={binary_prob_zyx.shape} group={group_label_zyx.shape}')
    foreground = binary_prob_zyx > float(binary_threshold)
    output = np.zeros(binary_prob_zyx.shape, dtype=np.uint8)
    output[foreground] = group_label_zyx[foreground].astype(np.uint8) + 1
    return output

def _extract_state_dict_and_hparams(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    ckpt = torch.load(path, map_location='cpu')
    hparams: Dict[str, object] = {}
    if isinstance(ckpt, dict):
        hp = ckpt.get('hyper_parameters', {})
        if isinstance(hp, dict):
            hparams = dict(hp)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        sd = ckpt['model']
        if not isinstance(sd, dict):
            raise RuntimeError(f"Unexpected 'model' format in checkpoint: {path}")
        return (sd, hparams)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        if not isinstance(sd, dict):
            raise RuntimeError(f"Unexpected 'state_dict' format in checkpoint: {path}")
        if any((k.startswith('model.') for k in sd)):
            sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
        return (sd, hparams)
    raise RuntimeError(f'Unrecognized checkpoint format: {path}')

def infer_group_n_classes(state_dict: Dict[str, torch.Tensor], hparams: Dict[str, object]) -> int:
    if 'out.conv.conv.weight' in state_dict:
        return int(state_dict['out.conv.conv.weight'].shape[0])
    if 'n_classes' in hparams:
        return int(hparams['n_classes'])
    raise RuntimeError('Cannot infer group output channels; expected out.conv.conv.weight or n_classes in checkpoint')

def build_nucleotide_group_model(checkpoint_path: Path, *, device: torch.device, feature_size: Optional[int], use_checkpoint: Optional[bool]) -> Tuple[nn.Module, Dict[str, object], int]:
    if SwinUNETR is None:
        raise RuntimeError('monai is required to use SwinUNETR')
    sd, hparams = _extract_state_dict_and_hparams(checkpoint_path)
    n_classes = infer_group_n_classes(sd, hparams)
    if n_classes != NUCLEOTIDE_GROUP_N_CLASSES:
        raise RuntimeError(f'{checkpoint_path} has {n_classes} group classes; expected {NUCLEOTIDE_GROUP_N_CLASSES} for nucleotide-group inference')
    model = SwinUNETR(in_channels=1, out_channels=n_classes, feature_size=resolve_feature_size(hparams, feature_size), use_checkpoint=resolve_use_checkpoint(hparams, use_checkpoint), spatial_dims=3)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    return (model, hparams, n_classes)

@torch.no_grad()
def predict_group_softmax_full_map(model: nn.Module, data_zyx: np.ndarray, *, patch_size: int, overlap: int, device: torch.device, n_classes: int) -> np.ndarray:
    shape = data_zyx.shape
    best_conf = np.full(shape, -np.inf, dtype=np.float32)
    best_probs = np.zeros((n_classes, *shape), dtype=np.float32)
    model.eval()
    for idx in iter_patch_indices(shape, int(patch_size), int(overlap)):
        ib, ie, jb, je, kb, ke = idx
        dz, dy, dx = (ie - ib, je - jb, ke - kb)
        patch = extract_patch(data_zyx, idx, int(patch_size)).astype(np.float32)
        x = torch.from_numpy(patch[None, None, ...]).to(device)
        logits = model(x)
        if logits.ndim != 5 or int(logits.shape[1]) != int(n_classes):
            raise RuntimeError(f'Expected group logits shape (B,{n_classes},D,H,W), got {tuple((int(value) for value in logits.shape))}')
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f'typed model produced nonfinite logits in patch {idx}')
        probs = F.softmax(logits, dim=1)
        pmax, _ = torch.max(probs, dim=1)
        probs_np = probs[0].detach().cpu().numpy().astype(np.float32)
        pmax_np = pmax[0].detach().cpu().numpy().astype(np.float32)
        probs_np = probs_np[:, :dz, :dy, :dx]
        pmax_np = pmax_np[:dz, :dy, :dx]
        cur_conf = best_conf[ib:ie, jb:je, kb:ke]
        mask = pmax_np > cur_conf
        cur_conf[mask] = pmax_np[mask]
        best_conf[ib:ie, jb:je, kb:ke] = cur_conf
        for class_idx in range(n_classes):
            cur = best_probs[class_idx, ib:ie, jb:je, kb:ke]
            cur[mask] = probs_np[class_idx][mask]
            best_probs[class_idx, ib:ie, jb:je, kb:ke] = cur
    return best_probs

def build_binary_model(checkpoint_path: Path, *, device: torch.device):
    state, hparams = _extract_state_dict_and_hparams(checkpoint_path)
    model = SwinUNETR(in_channels=1, out_channels=1, feature_size=resolve_feature_size(hparams, None), use_checkpoint=resolve_use_checkpoint(hparams, None), spatial_dims=3)
    model.load_state_dict(state)
    return (model.to(device).eval(), hparams)

@torch.no_grad()
def segment(density: np.ndarray, checkpoints: Path, device: str) -> np.ndarray:
    density = np.asarray(density, dtype=np.float32)
    if density.ndim != 3 or not density.size or (not np.isfinite(density).all()):
        raise ValueError('Density must be a nonempty finite three-dimensional array')
    paths = {f'{kind}{size}': Path(checkpoints) / f'{kind}{size}' / 'best.pt' for kind in ('binary', 'typed') for size in (64, 128)}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing segmentation checkpoint: {path}')
    target_device = torch.device(device)
    binary = np.zeros(density.shape, dtype=np.float64)
    threshold = DEFAULT_BINARY_THRESHOLD
    for size in (64, 128):
        model, hparams = build_binary_model(paths[f'binary{size}'], device=target_device)
        if size == 64:
            threshold = resolve_binary_threshold(hparams, None)
        binary += 0.5 * predict_binary_probability_full_map(model, density, patch_size=size, overlap=size // 2, device=target_device).astype(np.float64)
        del model
        if target_device.type == 'cuda':
            torch.cuda.empty_cache()
    typed = np.zeros((4, *density.shape), dtype=np.float64)
    for size in (64, 128):
        model, _, _ = build_nucleotide_group_model(paths[f'typed{size}'], device=target_device, feature_size=None, use_checkpoint=None)
        typed += 0.5 * predict_group_softmax_full_map(model, density, patch_size=size, overlap=size // 2, device=target_device, n_classes=4).astype(np.float64)
        del model
        if target_device.type == 'cuda':
            torch.cuda.empty_cache()
    return combine_binary_and_group_labels(binary.astype(np.float32), np.argmax(typed, axis=0).astype(np.uint8), binary_threshold=threshold)
