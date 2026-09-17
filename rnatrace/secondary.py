from pathlib import Path
import hashlib
import json
import os
import tempfile
import urllib.request


MODEL_FILES = {
    'RNAformer_32M_config_bprna.yml': '2af84e2caed3a784c5e45101be0a04c37c94f4605d3d3e3ed82a18c4d2948054',
    'RNAformer_32M_state_dict_bprna.pth': 'b2775613e9addafe30d17f7152dc985e8381826842d5d5056030006adfad6769',
}


def ensure_models(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in MODEL_FILES.items():
        destination = directory / name
        if not destination.exists():
            fd, temporary = tempfile.mkstemp(dir=directory, suffix='.download')
            os.close(fd)
            try:
                url = 'https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/' + name
                with urllib.request.urlopen(url, timeout=120) as source, open(temporary, 'wb') as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                with open(temporary, 'rb') as target:
                    actual = hashlib.file_digest(target, 'sha256').hexdigest()
                if actual != digest:
                    raise ValueError(f'RNAformer checksum mismatch: {name}')
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        with destination.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError(f'RNAformer checksum mismatch: {destination}')
    return directory


def decode_pairs(probabilities, sequence):
    canonical = {'AU', 'UA', 'GC', 'CG', 'GU', 'UG'}
    candidates = []
    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            probability = float(probabilities[i, j])
            if probability > 0.5:
                candidates.append({'i': i, 'j': j, 'prob': probability, 'type': f'{sequence[i]}-{sequence[j]}', 'canonical': sequence[i] + sequence[j] in canonical})
    occupied = set()
    selected = []
    for pair in sorted(candidates, key=lambda p: (-p['prob'], p['i'], p['j'])):
        if pair['i'] not in occupied and pair['j'] not in occupied:
            selected.append(pair)
            occupied.update((pair['i'], pair['j']))
    return sorted(selected, key=lambda p: (p['i'], p['j']))


def predict(sequence, directory, device, output):
    import numpy as np
    import torch
    from RNAformer.model.RNAformer import RiboFormer
    from RNAformer.utils.configuration import Config

    directory = ensure_models(directory)
    config = Config(config_file=str(directory / 'RNAformer_32M_config_bprna.yml'))
    config.RNAformer.cycling = 6
    config.RNAformer.flash_attn = False
    model = RiboFormer(config.RNAformer)
    state = torch.load(directory / 'RNAformer_32M_state_dict_bprna.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    dtype = torch.float32
    if torch.device(device).type == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model.to(dtype=dtype)
    source = torch.tensor([['ACGUN'.index(base) for base in sequence]], dtype=torch.long, device=device)
    length = torch.tensor([len(sequence)], dtype=torch.long, device=device)
    flag = torch.ones((1, 1), dtype=dtype, device=device)
    with torch.no_grad():
        logits, _ = model(source, length, flag)
        probabilities = torch.sigmoid(logits[0, :, :, -1]).float().cpu().numpy()
    if not np.all(np.isfinite(probabilities)):
        raise ValueError('RNAformer produced non-finite probabilities')
    pairs = decode_pairs(probabilities, sequence)
    result = {'source': 'rnaformer', 'model': 'RNAformer_32M_bprna', 'index_base': 0, 'first_residue': 1, 'sequence': sequence, 'sequence_length': len(sequence), 'base_pairs': pairs}
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    del model, logits
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return [(pair['i'], pair['j']) for pair in pairs if pair['canonical'] and pair['j'] - pair['i'] >= 4]
