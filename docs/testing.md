# Testing And Verification

## Automated Suite

Run:

```bash
python -m pytest -q
```

Coverage includes:

- RGB, NPY, GeoTIFF, ZIP, and nested archive loading
- macOS archive artifact filtering
- channel layout and precision validation
- group-disjoint splits
- sample-balanced splitting for uneven groups
- portable catalog generation and reload
- GeoTIFF slicing/stitching, including rotated affine transforms
- reused ZIP handles for large archives
- official `timm` model selection and pretrained flags
- RGB identity adapter and multispectral projection
- MIM and DINO tensor behavior
- DINO teacher-adapter EMA and legacy checkpoint migration
- paired student/teacher crop replay
- strict config parsing, validation, and config-relative paths
- command logging
- run-manifest generation
- atomic checkpoints, resume, RNG state, metrics, figures, and attempt manifests

Tests that require optional rasterio support are skipped if it is unavailable.

## Continuous Integration

`.github/workflows/ci.yml` installs the complete package and runs the suite on:

- Ubuntu with Python 3.9
- Ubuntu with Python 3.11
- macOS with Python 3.11
- Windows with Python 3.11

The workflow runs Ruff and does not download pretrained weights because model
tests replace `timm` with a deterministic test double.

## Verified End-To-End Runs

On June 15, 2026, the following completed successfully on Apple MPS:

| Method | Backbone | Input | Result |
| --- | --- | --- | --- |
| MIM | official pretrained ViT-S | five-band float32 NPY | checkpoint, metrics, curve, reconstruction |
| DINO | official pretrained ViT-S | RGB PNG | checkpoint, metrics, curve, views, similarity |
| DINO | official pretrained ViT-S | five-band float32 NPY | EMA spectral adapters, checkpoint, metrics, views, similarity |

The verification also confirmed:

- editable package installation
- pretrained weight retrieval
- direct CLI command logging
- manifest dependency and model metadata
- safe checkpoint reload with current PyTorch defaults
- complete config and RNG state in both checkpoint types
- two-epoch resume with nonzero final-epoch learning rate
- `64 passed` in a rasterio-capable environment after the implementation audit

## Acceptance Checklist For A New Machine

```bash
python -m pip install -e '.[dev,ml]'
python -m ag_foundation --help
python -m ag_foundation create-demo-data
python -m ag_foundation train-mim --config configs/demo_mim.yaml
python -m ag_foundation train-dino --config configs/demo_dino.yaml
python -m pytest -q
```

Confirm:

- both runs exit with code zero
- `best.pt` and `last.pt` exist
- `metrics.csv` has one row
- `training_metrics.png` opens
- MIM reconstruction opens
- DINO views and similarity figures open
- command log has a successful footer

## Scientific Validation Is Separate

Passing engineering tests does not establish representation quality. A paper
still requires downstream tasks, baselines, ablations, multiple seeds, external
domains, statistical analysis, and data governance.
