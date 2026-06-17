# Agricultural Vision Foundation Model

## Project Objective

Build a sensor-adaptive agricultural vision foundation model that starts from
official ImageNet-pretrained Vision Transformers and continues self-supervised
pretraining over heterogeneous RGB, GeoTIFF, and multispectral agricultural
imagery.

The project is intended for a top-tier computer-vision or agricultural
informatics publication. Engineering completion is necessary but not sufficient:
the final contribution must demonstrate transferable representations under real
crop, source, geography, and sensor shift.

## Research Questions

1. How much does agricultural continual pretraining improve over ImageNet alone?
2. Do MIM and DINOv3-style objectives learn complementary agricultural features?
3. Can a lightweight 1x1 spectral adapter preserve official RGB ViTs while
   transferring effectively to multispectral sensors?
4. Which data sources and modalities produce the largest cross-domain gains?
5. How robust are the representations in low-label, unseen-crop, unseen-source,
   and unseen-sensor settings?

## Implemented System

- official `timm` ViT-S, ViT-B, and ViT-L
- ImageNet initialization enabled by default
- always-present learnable 1x1 band adapter
- MAE-style masked image modeling
- DINOv3-style student-teacher continual pretraining
- EMA teacher adapter, backbone, and projection head with paired global crops
- RGB, multiband GeoTIFF, NPY, ZIP, and nested-ZIP input
- group-disjoint train/validation splitting
- model-output visualization and live metrics
- full checkpoint state and experiment provenance
- atomic artifacts and immutable per-invocation manifest history
- cross-platform shell and PowerShell launchers
- deterministic fresh-clone demo workflow

## Proposed Contributions

### Sensor-Adaptive Official ViT

A minimal spectral projection adapts arbitrary agricultural bands to the
three-channel interface expected by official pretrained ViTs. The central
scientific claim must be supported by comparisons against RGB-only input,
expanded patch embeddings, frozen adapters, and alternative spectral modules.

### Complementary SSL Study

The common data/backbone/tracking stack enables a controlled MIM-versus-DINO
comparison. A stronger extension is sequential MIM then DINO training or a joint
reconstruction/distillation objective.

### Curated Agricultural Corpus

The corpus contribution should include provenance, licenses, versions, sensor
metadata, duplicate clusters, and domain-aware splits. The current local corpus
contains 317,224 loader-visible RGB images but no real multispectral archive;
this gap must be closed before making multimodal foundation-model claims.

### Cross-Domain Benchmark

Evaluate classification, detection, segmentation, few-shot transfer, and
retrieval across close-range, field, and geospatial data. Reserve entire sources
or geographies for external evaluation.

### Reproducible Research Artifact

Release permitted data manifests, configs, logs, metrics, figures, code,
environment guidance, and checkpoints. Every reported number should map to a
run manifest and immutable commit.

## Experimental Program

Phase 1:

- deduplicate and freeze corpus v1
- establish ImageNet-only baselines
- tune ViT-S MIM and DINO recipes
- validate RGB and multispectral adapter behavior

Phase 2:

- scale selected recipes to ViT-B and ViT-L
- add source-balanced sampling and distributed training
- run broad downstream transfer
- execute adapter, data, objective, and scale ablations

Phase 3:

- repeat key experiments across at least three seeds
- test held-out crops, sources, geographies, and sensors
- report compute and efficiency
- prepare release artifacts and manuscript

## Success Criteria

- consistent transfer gains over ImageNet across multiple task families
- stronger label efficiency, not only full-data accuracy
- measurable benefit on real multispectral or GeoTIFF tasks
- robust gains on held-out domains
- ablations that isolate why the adapter and objective work
- no train/test duplicate contamination
- transparent licenses and data provenance
- complete reproducibility from repository commit to reported result

## Honest Current Status

The software foundation is operational and tested for MIM and DINOv3 with official
pretrained ViT-S models, including five-band input. The current demo and smoke
runs prove implementation correctness only. Dataset curation, real multispectral
coverage, distributed scale, downstream benchmarking, and full DINOv3 paper-scale
parity remain research work required for a top-tier submission.

See [docs/README.md](docs/README.md) for the complete technical documentation.
