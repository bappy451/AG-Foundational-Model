# Documentation Index

This directory is the technical and research record for `AG_Foundational_Model`.
The root README is the operational quick start; these documents explain the
contracts, assumptions, implementation choices, and publication plan.

## Operations

1. [Setup and portability](setup.md)
2. [Project runbook and audit report](runbook.md)
3. [Data formats and catalogs](data-formats.md)
4. [Configuration reference](configuration.md)
5. [Training workflows](training.md)
6. [Experiment tracking and artifacts](experiment-tracking.md)
7. [Reproducibility](reproducibility.md)
8. [Testing and verification](testing.md)
9. [Troubleshooting](troubleshooting.md)

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
- gradient accumulation for 24 GB single-GPU training
- RGB, GeoTIFF, multispectral NPY, folder, and ZIP input
- Group-disjoint train/validation splitting
- Portable config-relative paths and root-relative catalogs
- Live metrics, per-epoch figures, diagnostics, and checkpoints
- Resume and initialize-from support for exact continuation and SSL stage handoff
- Atomic checkpoint/metric writes and immutable per-attempt manifests
- Append-only command logging
- Run manifests with data, model, environment, CUDA, distributed, and Git metadata
- Resumable RNG and DataLoader state

## Documentation Policy

When behavior changes, update the corresponding document in the same commit.
Scientific claims belong in the publication roadmap or project description;
engineering guarantees belong in the operational documents.
