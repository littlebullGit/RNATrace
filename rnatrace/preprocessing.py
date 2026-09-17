from pathlib import Path
import json
import shutil
import subprocess

import mrcfile
import numpy as np


def read_map(path):
    with mrcfile.open(path, permissive=False) as mrc:
        if mrc.data.ndim != 3 or any(n < 2 for n in mrc.data.shape):
            raise ValueError('The input must be a three-dimensional density map')
        axes = [int(mrc.header.maps), int(mrc.header.mapr), int(mrc.header.mapc)]
        if sorted(axes) != [1, 2, 3]:
            raise ValueError('Invalid MRC axis order')
        density = np.transpose(mrc.data, [axes.index(i) for i in [3, 2, 1]]).astype(np.float32)
        spacing = np.array([float(mrc.voxel_size[x]) for x in ['x', 'y', 'z']])
        origin = np.array([float(mrc.header.origin[x]) for x in ['x', 'y', 'z']])
        if np.allclose(origin, 0):
            start = [int(mrc.header.nzstart), int(mrc.header.nystart), int(mrc.header.nxstart)]
            origin = np.array([start[axes.index(i)] for i in [1, 2, 3]]) * spacing
        angles = np.array([float(mrc.header.cellb[x]) for x in ['alpha', 'beta', 'gamma']])
        if not np.allclose(angles, 90):
            raise ValueError('Only orthogonal MRC unit cells are supported')
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(origin)):
        raise ValueError('Map density and origin must be finite')
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Map voxel sizes must be finite and positive')
    return density, origin, spacing


def normalize_density(density, contour):
    if not np.isfinite(contour):
        raise ValueError('Contour must be finite')
    foreground = density[density > contour]
    if not foreground.size:
        raise ValueError('No map density lies above the contour')
    upper = float(np.percentile(foreground, 95))
    if upper <= contour:
        raise ValueError('Map has no usable density range')
    return np.clip((density.astype(np.float32) - contour) / (upper - contour), 0, 1)


def prepare_map(path, output, contour=0.0, chimerax='ChimeraX', preprocessed=False):
    path = Path(path).resolve()
    read_map(path)
    if not preprocessed:
        executable = shutil.which(chimerax)
        if executable is None:
            raise FileNotFoundError(f'ChimeraX executable not found: {chimerax}. Install ChimeraX or pass --chimerax /path/to/ChimeraX')
        if not np.isfinite(contour):
            raise ValueError('Contour must be finite')
        processed = output / 'preprocessed.mrc'
        intermediate = output / 'masked.mrc'
        for filename in (path, processed, intermediate):
            if '\n' in str(filename) or '\r' in str(filename):
                raise ValueError('Map paths cannot contain newlines')
        commands = [
            f'open {json.dumps(str(path))}',
            f'volume #1 style surface level {contour} step 1',
            'volume #1 calculateSurface true',
            'surface dust #1 size 25',
            'volume mask #1 surfaces #1 modelId #2',
            f'save {json.dumps(str(intermediate))} #2',
            'close all',
            f'open {json.dumps(str(intermediate))}',
            'volume resample #1 spacing 0.5 modelId #2',
            'close #1',
            f'volume #2 style surface level {contour} step 1',
            'volume #2 calculateSurface true',
            'surface dust #2 size 25',
            'volume mask #2 surfaces #2 modelId #3',
            f'save {json.dumps(str(processed))} #3',
            'exit',
        ]
        script = output / 'preprocess.cxc'
        script.write_text('\n'.join(commands) + '\n')
        with (output / 'preprocess.log').open('w') as log:
            subprocess.run([executable, '--nogui', '--exit', str(script)], stdout=log, stderr=subprocess.STDOUT, check=True)
        path = processed
    density, origin, spacing = read_map(path)
    if not np.allclose(spacing, 0.5, atol=1e-5):
        raise ValueError('The preprocessed map must have 0.5 Å isotropic voxels')
    return normalize_density(density, contour), origin, spacing
