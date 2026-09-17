# RNATrace

Reconstruct an RNA chain from a cryo-EM map and its sequence. Runs entire fully automated RNATrace pipeline.

Use Python 3.12, [UCSF ChimeraX](https://www.rbvi.ucsf.edu/chimerax/download.html), and preferably an NVIDIA GPU with at least 24 GB VRAM. CPU inference is supported but slower.

```bash
pip install .
pip install --no-deps 'git+https://github.com/automl/RNAformer.git@196e7dea01e5446cfbcb9e9b594cd10a8b537372'
```

The four trained models are included in `checkpoints/`. RNAformer BP-RNA weights download automatically on first use.

```bash
python -m rnatrace --map input.mrc --sequence input.fasta --output results
```

Final coordinates are `results/model.pdb` and `results/model.cif`. FASTA must contain one A/C/G/U sequence (T is converted to U).

Supply an input cryo-EM map. The default contour is **0**, suitable for masked maps. Use `--contour VALUE` for another map contour. Set `--chimerax /path/to/ChimeraX` if it is not on PATH. Use `--preprocessed` only for maps already dust-filtered and resampled to 0.5 Å. Optional: `--checkpoints /path/to/checkpoints` to use other trained checkpoints.
