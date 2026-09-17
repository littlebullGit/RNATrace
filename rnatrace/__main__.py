import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys


def read_sequence(path):
    text = Path(path).read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or sum(line.startswith('>') for line in lines) > 1:
        raise ValueError('Provide one nonempty RNA sequence in FASTA or plain text')
    if any(line.startswith('>') for line in lines[1:]):
        raise ValueError('The FASTA header must precede the sequence')
    sequence = ''.join(line for line in lines if not line.startswith('>')).upper().replace('T', 'U')
    if len(sequence) < 2 or set(sequence) - set('ACGU'):
        raise ValueError('Sequence must contain at least two nucleotides and only A, C, G, U (or T)')
    return sequence


def run_stage(module, arguments, log):
    with log.open('w') as stream:
        try:
            subprocess.run([sys.executable, '-m', f'rnatrace.{module}', *map(str, arguments)], stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'{module} failed; see {log}') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reconstruct one RNA chain from a cryo-EM map and sequence')
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, default=Path('checkpoints'))
    parser.add_argument('--rnaformer-models', type=Path)
    parser.add_argument('--contour', type=float, default=0.0, help='Map contour level (default: 0 for masked maps)')
    parser.add_argument('--chimerax', default='ChimeraX', help='ChimeraX executable')
    parser.add_argument('--preprocessed', action='store_true', help='Input already dust-filtered and resampled to 0.5 Å')
    parser.add_argument('--device', help='PyTorch device, default CUDA if available, otherwise CPU')
    parser.add_argument('--openmm-platform', choices=['CPU', 'CUDA', 'OpenCL', 'Reference'], default='CPU')
    args = parser.parse_args(argv)
    sequence = read_sequence(args.sequence)
    if not args.map.is_file():
        parser.error(f'Map does not exist: {args.map}')
    for name in ['binary64', 'binary128', 'typed64', 'typed128']:
        checkpoint = args.checkpoints / name / 'best.pt'
        if not checkpoint.is_file():
            parser.error(f'Missing checkpoint: {checkpoint}. See README for the one-time setup.')
    if not args.preprocessed and shutil.which(args.chimerax) is None:
        parser.error('ChimeraX is required for preprocessing; install it or set --chimerax to its executable')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Use a new or empty output directory')

    import numpy as np
    import torch
    from .preprocessing import prepare_map
    from .segmentation import segment
    from .extraction import extract
    from .secondary import predict
    from .registration import register

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.device(device).type == 'cuda':
        torch.cuda.set_device(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info('1/6 Preparing map')
    density, origin, spacing = prepare_map(args.map, output, args.contour, args.chimerax, args.preprocessed)
    logging.info('2/6 Segmenting with the 64³ + 128³ ensemble')
    labels = segment(density, args.checkpoints, device)
    labels_path = output / 'labels.npy'
    np.save(labels_path, labels)
    metadata_path = output / 'labels_meta.json'
    metadata = {'origin_xyz': origin.tolist(), 'voxel_size_xyz': spacing.tolist(), 'class_names': ['bg', 'p', 'sugar', 'purine', 'pyrimidine'], 'contour': args.contour}
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    logging.info('3/6 Extracting phosphorus sites')
    positions = extract(labels, density, origin, float(spacing[0]), len(sequence))
    del density
    logging.info('4/6 Predicting secondary structure and registering the backbone')
    structure = output / 'secondary_structure.json'
    model_dir = args.rnaformer_models or args.checkpoints / 'rnaformer'
    pairs = predict(sequence, model_dir, device, structure)
    backbone = register(positions, sequence, pairs, labels, origin, spacing, output / 'backbone.json')
    assignments = json.loads(backbone.read_text())['p_atoms']
    residues = [atom['residue_num'] for atom in assignments]
    if len(residues) != len(sequence) or set(residues) != set(range(1, len(sequence) + 1)):
        raise RuntimeError('Registration did not cover the complete sequence. Inspect backbone.json and check the map contour and RNA region.')
    logging.info('5/6 Building all-atom coordinates')
    atomic = output / 'unrefined.pdb'
    run_stage('build_atomic_model', ['--backbone', backbone, '--structure-2d', structure, '--output', atomic], output / 'atomic.log')
    logging.info('6/6 Refining with AMBER14 RNA.OL3 and 2,000 MD steps')
    final = output / 'model.pdb'
    run_stage('refine_geometry', ['--input', atomic, '--structure-2d', structure, '--predictions', labels_path, '--predictions-meta', metadata_path, '--output', final, '--output-cif', output / 'model.cif', '--openmm-platform', args.openmm_platform], output / 'refinement.log')
    if not final.is_file() or final.stat().st_size == 0:
        raise RuntimeError('Refinement produced no final model')
    logging.info('Finished: %s', final)


if __name__ == '__main__':
    main()
