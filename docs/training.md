# Training Workflows

## Quick Engineering Runs

```bash
python -m ag_foundation create-demo-data
bash scripts/train_mim.sh --config configs/demo_mim.yaml
bash scripts/train_dino.sh --config configs/demo_dino.yaml
```

These verify code paths with one epoch and tiny images. They are not scientific
experiments.

## Local Pretraining Smoke Runs

The current workspace includes two large archives in the sibling
`Pretraining/` directory. The smoke configs use a portable 64-image catalog:

```bash
bash scripts/train_mim.sh --config configs/pretraining_seedlings_smoke.yaml
bash scripts/train_dino.sh --config configs/pretraining_dino_smoke.yaml
```

After cloning elsewhere, preserve the same sibling layout or edit `data_root`.

## Production MIM

Start from `configs/train_mim.example.yaml`.

Recommended initial experiment:

- ViT-B
- 224 or larger crop divisible by 16
- ImageNet initialization
- mask ratio 0.75
- bf16 or fp16 on supported GPUs
- 50 to 200 epochs depending on corpus size
- group-disjoint validation
- gradient checkpointing when memory-constrained

Run:

```bash
bash scripts/train_mim.sh --config configs/my_mim.yaml
```

## Production DINO

Start from `configs/train_dino.example.yaml`.

The DINO head can consume substantial memory when `dino_out_dim` is 65,536.
Reduce batch size, enable gradient checkpointing, or begin with a smaller output
dimension while validating the pipeline. Publication comparisons must report
the exact head and crop settings.

```bash
bash scripts/train_dino.sh --config configs/my_dino.yaml
```

## Continual Pretraining Meaning

Both objectives begin from ImageNet weights and update the complete trainable
student backbone plus the band adapter. DINO's teacher adapter, backbone, and
head follow their student counterparts by EMA. This is domain-adaptive
continual pretraining, not training from scratch.

## Live Monitoring

The terminal reports:

- epoch and batch position
- current and running-average loss
- learning rate
- current teacher EMA momentum for DINO
- validation loss
- epoch duration
- generated artifact paths

After each epoch, inspect:

- `metrics.csv`
- `figures/training_metrics.png`
- MIM reconstruction or DINO view/similarity figures
- `last.pt`

## Resume

Automatic:

```yaml
runtime:
  output_dir: ../runs/my_run
  epochs: 100
  resume: true
```

Explicit:

```bash
python -m ag_foundation train-dino \
  --config configs/my_dino.yaml \
  --epochs 100 \
  --resume-from runs/my_run/last.pt
```

If the checkpoint completed epoch 40 and `epochs` is 100, training continues at
epoch 41. If `epochs` is 40, no additional epoch runs.

The epoch-level cosine schedule applies a nonzero learning rate to every
training epoch. For example, a two-epoch run uses base LR and half base LR,
rather than spending the second epoch at zero LR.

## Scaling Guidance

- Establish correctness with ViT-S before allocating ViT-L resources.
- Increase crop size before claiming high-resolution behavior.
- Measure archive I/O utilization before adding GPUs.
- Use source-balanced or dataset-balanced sampling for heterogeneous corpora.
- Deduplicate before long pretraining.
- Keep one immutable config and one output directory per experiment.
- Repeat key results across at least three seeds.

## Current Limits

- Launchers are single-process; distributed samplers activate only when an
  external process group exists.
- There is no automatic gradient accumulation.
- There is no built-in W&B/TensorBoard backend; local CSV, JSON, PNG, logs, and
  checkpoints are the source of truth.
- The DINO implementation is DINOv3-style with Gram anchoring and still does
  not attempt the paper's full large-scale training and distillation stack.
