# Implementation Map

## Package Entry

- `src/ag_foundation/__main__.py`: `python -m ag_foundation`
- `src/ag_foundation/cli.py`: command routing
- `src/ag_foundation/command_logging.py`: direct CLI tee logging

## Data

- `data/dataset.py`: discovery, archive URIs, loading, normalization, catalogs,
  augmentation, splitting, and DataLoaders; **undersized images are zero-padded;
  GeoTIFF NoData (signed integer min values) are clamped to 0 before normalization**
- `data/multi_source_dataset.py`: multi-source pretraining dataset spanning many
  ZIP archives and directories simultaneously; source-balanced weighted sampling;
  duplicate detection; catalog-based and root-scan-based discovery
- `data/geotiff.py`: tiled read/write and GeoTIFF stitching
- `data/demo.py`: deterministic RGB and multispectral fixtures

## Models

- `models/official_vit.py`: supported official ViTs, band adapter, pretrained
  normalization, positional interpolation, and token encoding; **RoPE backbone
  support: 4D patch embed auto-flattened to (B,N,C); absolute pos embed skipped
  when `backbone.pos_embed is None` (EVA02, DINOv3)**
- `models/mim.py`: mask generation, reconstruction head, and masked loss
- `models/dino.py`: student/teacher backbones, heads, EMA, centering, and loss
- `models/vit.py`: compatibility exports for supported ViT configuration

## Training

- `training/mim_runner.py`: MIM config, parser, model/optimizer construction
- `training/dino_runner.py`: DINO config, parser, crop and trainer construction
- `training/ssl_trainer.py`: MIM steps, validation, metrics, checkpoints, resume
- `training/dino_trainer.py`: multi-crop augmentation and DINO trainer
- `training/visualization.py`: headless PNG and CSV diagnostics
- `training/state.py`: RNG and DataLoader generator capture/restore
- `training/artifacts.py`: atomic text/checkpoint writes and best snapshots
- `training/experiment_metadata.py`: manifests and Git/environment summaries

## Launchers

- `scripts/train_mim.sh`, `scripts/train_dino.sh`: macOS/Linux logging wrappers
- `scripts/train_mim.ps1`, `scripts/train_dino.ps1`: Windows wrappers
- `scripts/common.sh`, `scripts/windows_common.ps1`: interpreter resolution
- `scripts/ag_foundation.py`: source-tree fallback launcher

## Design Boundaries

- Data loading produces normalized `C,H,W` tensors.
- The adapter owns conversion from `C` sensor bands to three model channels.
- The official ViT owns ImageNet normalization and token encoding.
- Objective modules own only their heads and losses.
- Trainers own optimization, metrics, visualization scheduling, and state.
- Runners own configuration and object construction.

This separation keeps downstream fine-tuning, alternative adapters, additional
SSL losses, and distributed launch support implementable without replacing the
data or backbone layers.
