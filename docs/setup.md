# Setup And Portability

## Supported Platforms

- Linux with CPU or CUDA
- macOS with CPU or Apple MPS
- Windows with CPU or CUDA
- Python 3.9 or newer; Python 3.11 is recommended

## Fresh Clone

```bash
git clone <repository-url>
cd AG_Foundational_Model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,ml]'
```

Windows PowerShell:

```powershell
git clone <repository-url>
Set-Location AG_Foundational_Model
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
```

`pip install -r requirements.txt` is an equivalent convenience command.

## Dependency Groups

Core dependencies:

- NumPy
- pandas
- Pillow
- PyYAML

The core install is sufficient for CLI help and deterministic demo-data
generation. Cataloging, GeoTIFF processing, model training, and tests should use
the `ml` and `dev` extras shown above.

ML dependencies:

- PyTorch
- torchvision
- `timm`
- rasterio
- matplotlib

Development dependencies:

- pytest
- Ruff

For a CUDA workstation, install the PyTorch build recommended for that
machine/driver before installing this package if the default PyPI wheel is not
appropriate.

## Verify The Installation

```bash
python -m ag_foundation --help
ag-foundation --help
python -m pytest -q
```

Both package entrypoints should list:

- `train-mim`
- `train-dino`
- `create-catalog`
- `create-demo-data`
- `slice-geotiffs`

## Self-Contained Verification

```bash
python -m ag_foundation create-demo-data
python -m ag_foundation train-mim --config configs/demo_mim.yaml
python -m ag_foundation train-dino --config configs/demo_dino.yaml
```

This creates 24 RGB PNGs and 24 five-band float32 NPY arrays under `data/demo`.
The data are synthetic and deterministic. They verify loading, adapters,
pretrained model construction, training, metrics, figures, and checkpointing.

## Pretrained Weight Download

`pretrained_backbone: true` is the project default. On the first run, `timm` may
download ImageNet weights through Hugging Face Hub. Requirements:

- internet access for the first uncached run
- enough cache storage for the selected ViT
- optional `HF_TOKEN` for higher Hub request limits

An offline machine must receive the model cache in advance. Disabling pretrained
weights is supported for tests with `--no-pretrained-backbone`, but it is not the
intended research setup.

## Device Selection

`device: auto` selects:

1. CUDA when available
2. Apple MPS when available
3. CPU otherwise

Override it with `device: cuda`, `cuda:<index>`, `mps`, or `cpu`. An explicitly
requested unavailable accelerator fails immediately with a clear error.

## Precision

- `fp32`: safest default and required for the small verification configs
- `fp16`: recommended on compatible CUDA hardware
- `bf16`: recommended on compatible modern CUDA hardware

CPU `fp16` requests compute safely in float32 where required.

## Git Portability

- Source, configs, docs, and small catalogs belong in Git.
- `data/`, `runs/`, checkpoints, and command logs are ignored.
- `.gitattributes` preserves LF shell scripts and Windows PowerShell line endings.
- Config paths are relative to the YAML file.
- Catalog member paths are relative to their configured data root.
- Run manifests record Git commit, branch, and dirty status when `.git` exists.

Do not commit private data, licensed archives, pretrained caches, or large
checkpoints unless the repository is explicitly configured for Git LFS.

This workspace currently needs a Git repository and remote before it can be
cloned:

```bash
git init
git add .
git commit -m "Initial agricultural foundation model"
git branch -M main
git remote add origin <repository-url>
git push -u origin main
```

Review licenses and `git status` before the first commit. Once pushed, the
included GitHub Actions workflow tests Linux, macOS, and Windows installations.
