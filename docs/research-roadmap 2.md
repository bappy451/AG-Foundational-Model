# Publication Roadmap

## Target Journals & Standards

Depending on where you submit, the focus of the paper needs to shift significantly.

### Targeting CVPR (Computer Vision and Pattern Recognition)
CVPR prioritizes methodological novelty, massive scale, and pushing the state-of-the-art (SOTA) on challenging computer vision benchmarks.
- **Novelty in SSL:** Adapt the architecture for agriculture. Examples: Handling extreme visual similarity between weeds and crops, or introducing a novel masking strategy for MIM that respects plant morphology.
- **Scale and Benchmarking:** Build a rigorous "Ag-Benchmark" suite (evaluating classification, object detection, and semantic segmentation). You must empirically prove that your Ag-Foundation model outperforms generic foundation models (like Meta's DINOv2 or OpenAI's CLIP) on agricultural downstream tasks.
- **Narrative Focus:** A highly technical paper focusing on representation learning, attention maps, ablation studies on pretraining recipes, and zero-shot/few-shot capabilities.

### Targeting CompAG (Computers and Electronics in Agriculture)
CompAG prioritizes practical, robust applications of AI to solve real-world agricultural problems.
- **Robustness & Application:** The novelty is the creation of the foundation model and demonstrating its massive utility. Show that fine-tuning requires significantly less annotated data (few-shot learning) to achieve high accuracy for a specific problem.
- **Cross-Domain Generalization:** Show that a model trained on your massive ~1.87M image dataset generalizes across different geographical regions, lighting conditions, and sensor types.
- **Narrative Focus:** A comprehensive, application-driven paper. Highlight the dataset curation process, computational efficiency of using a foundation model vs. training from scratch, and practical metrics (e.g., inference speed, deployment feasibility).

## Strong Contribution Package

1. A provenance-aware, deduplicated agricultural pretraining corpus of ~1.87M+ images.
2. A simple sensor adapter that preserves official RGB ImageNet ViTs.
3. Controlled MIM versus DINOv3-style continual-pretraining comparison.
4. RGB-to-multispectral transfer and mixed-sensor training.
5. Broad downstream evaluation under source and geography shift.
6. Open configs, manifests, checkpoints, and reproducible protocols.

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
- dataset scale (~1.87M vs sub-samples)
- source-balanced sampling
- duplicate removal
- crop resolution
- MIM then DINO sequential pretraining versus either alone

## DINOv3 Alignment

The current implementation is intentionally DINOv3-style rather than a strict       
paper reproduction. It already includes the pieces that matter for this
project:

- official ViT-S/B/L backbones
- ImageNet initialization
- 1x1 spectral adapter
- student-teacher self-distillation
- Gram anchoring on dense features
- constant teacher-momentum scheduling
- continual pretraining on agricultural RGB and multispectral data

To make it a stronger DINOv3 reproduction and publication baseline, add:

- distributed multi-GPU training
- higher-resolution pretraining runs
- post-hoc distillation into smaller student variants
- larger-scale crop and augmentation sweeps
- source-balanced sampling across mixed agricultural corpora
- stability studies at larger output dimensions

Until then, use the precise phrase "DINOv3-style continual pretraining" rather      
than claiming a full Meta-scale reproduction.

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

1. Standardize the ~1.87 million images (remove duplicates, handle corrupted images, normalize resolutions).
2. Freeze and audit corpus v1.
3. Establish 4-5 downstream tasks (e.g., Leaf Disease Classification, Weed Segmentation, Crop Yield Detection) and hold out specific datasets strictly for testing.
4. Add real multispectral/GeoTIFF and pest-focused sources.
5. Implement distributed training and balanced sampling.
6. Train baseline ResNet/ViT from scratch and fine-tune ImageNet-pretrained DINOv2 to set benchmarks.
7. Run ViT-S recipe selection on the curated 1.87M dataset.
8. Scale selected recipes to ViT-B/L.
9. Execute downstream benchmark and ablations.
10. Repeat critical experiments across seeds.
11. Release manifests, configs, code, and permitted checkpoints.
