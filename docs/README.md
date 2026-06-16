# Documentation Index

This directory is the technical and research record for `AG_Foundational_Model`.
The root README is the operational quick start; these documents explain the
contracts, assumptions, implementation choices, and publication plan.

## Operations

1. [Setup and portability](setup.md)
2. [Data formats and catalogs](data-formats.md)
3. [Configuration reference](configuration.md)
4. [Training workflows](training.md)
5. [Experiment tracking and artifacts](experiment-tracking.md)
6. [Reproducibility](reproducibility.md)
7. [Testing and verification](testing.md)
8. [Troubleshooting](troubleshooting.md)

## Design And Research

1. [Architecture](architecture.md)
2. [Implementation map](implementation-map.md)
3. [Dataset audit and strategy](dataset-strategy.md)
4. [Publication roadmap](research-roadmap.md)
5. [Root research proposal](../project_description.md)

## Implemented Contract

- Official `timm` ViT-S, ViT-B, and ViT-L
- ImageNet-pretrained initialization by default
- Always-present learnable 1x1 band adapter
- MIM and DINO pretraining
- RGB, GeoTIFF, multispectral NPY, folder, and ZIP input
- Group-disjoint train/validation splitting
- Portable config-relative paths and root-relative catalogs
- Live metrics, per-epoch figures, diagnostics, and checkpoints
- Atomic checkpoint/metric writes and immutable per-attempt manifests
- Append-only command logging
- Run manifests with data, model, environment, and Git metadata
- Resumable RNG and DataLoader state

## Documentation Policy

When behavior changes, update the corresponding document in the same commit.
Scientific claims belong in the publication roadmap or project description;
engineering guarantees belong in the operational documents.
