# Publication Roadmap

## Target Standard

A CVPR or Computers and Electronics in Agriculture submission needs evidence
that the model transfers across agricultural domains, not only that SSL loss
decreases.

## Strong Contribution Package

1. A provenance-aware, deduplicated agricultural pretraining corpus
2. A simple sensor adapter that preserves official RGB ImageNet ViTs
3. Controlled MIM versus DINO continual-pretraining comparison
4. RGB-to-multispectral transfer and mixed-sensor training
5. Broad downstream evaluation under source and geography shift
6. Open configs, manifests, checkpoints, and reproducible protocols

## Required Baselines

- ImageNet-pretrained ViT without agricultural pretraining
- random initialization where scientifically informative
- supervised agricultural pretraining
- MIM-only continual pretraining
- DINO-only continual pretraining
- RGB-only versus multispectral
- adapter frozen versus trainable
- adapter versus direct patch-embedding expansion
- relevant public agricultural foundation models when licenses permit

## Evaluation Matrix

Tasks:

- classification
- object detection
- semantic or instance segmentation
- few-shot and linear probing
- full fine-tuning
- retrieval or clustering

Domains:

- lab leaf images
- field images
- weeds and seedlings
- disease and stress
- RGB
- multispectral/GeoTIFF
- unseen crop
- unseen source dataset
- unseen geography, farm, season, or sensor

Report:

- mean and standard deviation across seeds
- label-efficiency curves
- calibration where decisions are safety/economically relevant
- per-domain and macro-averaged performance
- compute, parameter count, throughput, and energy where available

## Critical Ablations

- ViT-S/B/L
- ImageNet start versus random start
- MIM mask ratio
- DINO crop count and teacher momentum
- 1x1 adapter initialization and trainability
- channel subsets and missing-band robustness
- dataset scale
- source-balanced sampling
- duplicate removal
- crop resolution
- MIM then DINO sequential pretraining versus either alone

## DINOv2 Gap

Before calling the method a full DINOv2 reproduction, add and validate:

- iBOT-style masked patch prediction
- KoLeo regularization
- teacher temperature warmup
- robust large-scale crop/augmentation recipe
- source-balanced sampling
- distributed multi-GPU training
- stability studies at large output dimension

Until then, use the precise phrase "DINO-style continual pretraining."

## Data Governance

For each source, document:

- license and redistribution permission
- version and retrieval date
- geographic and demographic limitations
- label provenance
- duplicate/near-duplicate handling
- train/test contamination checks
- excluded samples and reasons

## Suggested Paper Narrative

Problem:

Agricultural imagery spans close-range RGB, field imagery, and multispectral
geospatial sensors, while standard ViTs assume RGB and agricultural labels are
fragmented across datasets.

Method:

Use a minimal learnable spectral adapter in front of official ImageNet ViTs and
continually pretrain with complementary reconstruction and self-distillation
objectives.

Evidence:

Show gains over ImageNet initialization across tasks, label budgets, sensors,
crops, and held-out domains, with rigorous data curation and ablation.

## Milestones

1. Freeze and audit corpus v1.
2. Add real multispectral/GeoTIFF sources.
3. Implement distributed training and balanced sampling.
4. Run ViT-S recipe selection.
5. Scale selected recipes to ViT-B/L.
6. Execute downstream benchmark and ablations.
7. Repeat critical experiments across seeds.
8. Release manifests, configs, code, and permitted checkpoints.

