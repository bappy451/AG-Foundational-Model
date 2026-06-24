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
- **undersized image zero-padding** (images smaller than `crop_size` are padded, not rejected)
- **GeoTIFF NoData clamping** (signed integers with extreme negatives are clipped to 0)
- **RoPE backbone compatibility** (4D `patch_embed` output flattening and `pos_embed=None` bypass)

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

On June 24, 2026, the following additional robustness fixes were validated:

| Fix | Test | Result |
| --- | --- | --- |
| Undersized image zero-padding | `test_dataset_pads_undersized_image_to_crop_size` | pass |
| Single-dimension padding | `test_dataset_pads_undersized_image_only_in_one_dimension` | pass |
| ZIP archive undersized padding | `test_dataset_zip_undersized_image_is_padded` | pass |
| NoData INT32 clamping | `test_normalize_image_array_clamps_negative_integers_to_zero` | pass |
| NoData INT16 clamping | `test_normalize_image_array_clamps_negative_int16_nodata` | pass |
| Positive integer unchanged | `test_normalize_positive_integers_unchanged` | pass |
| RoPE 4D patch embed flatten | `test_vit_handles_rope_4d_patch_embed_output` | pass |
| RoPE missing pos_embed bypass | `test_vit_handles_missing_pos_embed_rope_backbones` | pass |

On June 19, 2026, the following checks completed successfully in this workspace:

| Method | Backbone | Input | Result |
| --- | --- | --- | --- |
| MIM | official ImageNet ViT-S | five-band float32 NPY | checkpoint, metrics, curve, reconstruction |
| DINO | official ImageNet ViT-S | RGB PNG | checkpoint, metrics, curve, views, similarity |
| DINO | official DINOv2 ViT-S | RGB PNG, patch-14 crop | checkpoint, metrics, curve, views, similarity |
| DINO | initialized from matched MIM ViT-S checkpoint | five-band float32 NPY | `initialize_from` manifest, checkpoint, metrics, views, similarity |

The verification also confirmed:

- dependency checks in shell wrappers
- official pretrained weight retrieval
- direct CLI command logging
- manifest dependency and model metadata
- safe checkpoint reload with current PyTorch defaults
- complete config and RNG state in both checkpoint types
- two-epoch resume with nonzero final-epoch learning rate
- concise compatibility errors for mismatched `initialize_from` checkpoints
- `124 passed, 1 skipped` after all robustness patches (June 24, 2026)

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
