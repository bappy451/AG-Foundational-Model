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

The current workspace includes a `Pretraining/` directory with ~40 source
datasets.  The smoke config runs 2 epochs on the entire multi-source corpus:

```bash
# Full-dataset 2-epoch smoke test (RTX 4090 optimized)
python -m ag_foundation train-dino --config configs/smoke_test.yaml
```

Smoke test settings (`configs/smoke_test.yaml`):

| Setting | Value | Notes |
| --- | --- | --- |
| `epochs` | 2 | fast validation cycle |
| `batch_size` | 8 | safe for 24 GB VRAM |
| `gradient_accumulation_steps` | 4 | effective batch = 32 |
| `precision` | bf16 | RTX 4090 optimized |
| `warmup_epochs` | 1 | 1 of 2 epochs is warmup |
| `visualization_every` | 1 | visualize after each epoch |
| `data_root` | `../Pretraining` | all ~40 source datasets |

Alternative quick runs with smaller config:

```bash
bash scripts/train_mim.sh --config configs/demo_mim.yaml
bash scripts/train_dino.sh --config configs/demo_dino.yaml
```

## Production MIM

Start from `configs/train_mim.example.yaml`.

Recommended initial experiment:

- ViT-B
- crop divisible by the selected patch size
- official checkpoint initialization
- mask ratio 0.75
- `bf16` on an RTX 4090 (this is the new default).
- batch size 4 to 8 with `gradient_accumulation_steps: 2` or higher as needed
- `epoch_batches` configured if using WebDataset Tarballs.
- 50 to 200 epochs depending on corpus size
- group-disjoint validation
- gradient checkpointing when memory-constrained

Run (Local WDS/CPU loader — Windows or Linux without DALI):

```bash
python -m ag_foundation train-mim --config configs/wds_mim_pretrain.yaml
```

Run (Google Colab / Linux with NVIDIA DALI GPU loader):

```python
# Install the package first
!pip install -e .[ml]

# Then launch with DALI
!python -m ag_foundation train-mim --config configs/wds_mim_pretrain.yaml --use-dali
```

Run (Compile for extra speed on Linux/Colab):

```bash
python -m ag_foundation train-mim --config configs/my_mim.yaml --use-dali --compile
```

## Production DINO

Start from `configs/train_dino.example.yaml`.

The DINO head can consume substantial memory when `dino_out_dim` is 65,536.
Reduce batch size, enable gradient checkpointing, use gradient accumulation, or
begin with a smaller output dimension while validating the pipeline.
Publication comparisons must report the exact head, crop, precision, and
accumulation settings.

Run (Local WDS/CPU loader — Windows or Linux without DALI):

```bash
python -m ag_foundation train-dino --config configs/wds_dino_pretrain.yaml
```

Run (Google Colab / Linux with NVIDIA DALI GPU loader):

```python
# Install the package first
!pip install -e .[ml]

# Then launch with DALI
!python -m ag_foundation train-dino --config configs/wds_dino_pretrain.yaml --use-dali
```

Run (Compile for extra speed on Linux/Colab):

```bash
python -m ag_foundation train-dino --config configs/my_dino.yaml --use-dali --compile
```

## Continual Pretraining Meaning

Both objectives begin from a selected official checkpoint family and update the
complete trainable student backbone plus the band adapter. DINO's teacher
adapter, backbone, and head follow their student counterparts by EMA. This is
domain-adaptive continual pretraining, not training from scratch.

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
  --resume-from runs/my_run/last.pt \
  --compile
```

If the checkpoint completed epoch 40 and `epochs` is 100, training continues at
epoch 41. If `epochs` is 40, no additional epoch runs.

The epoch-level cosine schedule applies a nonzero learning rate to every
training epoch. For example, a two-epoch run uses base LR and half base LR,
rather than spending the second epoch at zero LR.

## Continual Pretraining

Use `runtime.initialize_from` or `--initialize-from` when you want to start a
new MIM or DINO run from an earlier SSL checkpoint without restoring optimizer
state, history, or epoch counters. This is the recommended path for MIM →
DINO continual pretraining and for cross-stage ablations.

`resume` is for exact continuation of the same run. `initialize_from` is for
starting a new run from a previous representation.

The source checkpoint and target run should use compatible model geometry:
same ViT family, selected patch source/crop divisibility, and input channel
count. Incompatible handoffs now fail early with a short compatibility message
instead of a long PyTorch tensor-shape dump.

## Scaling Guidance

- Establish correctness with ViT-S before allocating ViT-L resources.
- Use gradient accumulation to raise effective batch size on 24 GB GPUs.
- Increase crop size before claiming high-resolution behavior.
- Measure archive I/O utilization before adding GPUs.
- Use source-balanced or dataset-balanced sampling for heterogeneous corpora.
- Deduplicate before long pretraining.
- Keep one immutable config and one output directory per experiment.
- Repeat key results across at least three seeds.

## Current Limits

- Launchers are single-process; distributed samplers activate only when an
  external process group exists.
- Automatic gradient accumulation is supported through
  `runtime.gradient_accumulation_steps`.
- Command logging is primary-process aware when launched under distributed
  environment variables.
- There is no built-in W&B/TensorBoard backend; local CSV, JSON, PNG, logs, and
  checkpoints are the source of truth.
- The DINO implementation is DINO-style and still does not attempt the
  paper's full large-scale training and distillation stack.

## WebDataset Data Pipeline Internals

This section documents how the CPU and GPU data loading pipelines work internally,
including important design decisions that prevent subtle bugs in multiprocessing environments.

### CPU WDS Loader (`wds_loader.py`)

The CPU pipeline uses `webdataset` with `resampled=True` (infinite random
sampling with replacement) for maximum simplicity and Windows compatibility.

Key design decisions:

- **`resampled=True`** — Enables infinite streaming. Workers randomly resample
  shards rather than cycling through a fixed shard list. This avoids the
  `split_by_worker` pickling issue on Windows (`spawn` multiprocessing).
- **`SizedWebDataset.__iter__` epoch boundary** — PyTorch's `DataLoader` does
  not know when to stop iterating over an infinite `IterableDataset`. The
  `SizedWebDataset` wrapper counts batches in its own `__iter__` and hard-stops
  after exactly `epoch_batches` items. This prevents the
  `Length of IterableDataset ... was reported to be N but M samples have been fetched`
  warning caused by `.with_epoch()` applying its limit per-worker rather than globally.
- **Hard `break` in `ssl_trainer.py`** — An additional safety guard breaks out
  of the training loop after `num_batches` steps regardless of what the
  DataLoader yields. This is the canonical pattern for large-scale WebDataset training.

### GPU DALI Loader (`dali_wds_loader.py`)

The GPU pipeline uses NVIDIA DALI's `fn.readers.webdataset` for hardware-
accelerated JPEG decoding via nvJPEG entirely on the GPU.

Key design decisions:

- **`ext=["jpg;jpeg;png;tif;tiff"]`** — A single semicolon-separated string tells
  DALI these are alternative formats for a **single** image output. Do NOT pass
  a Python list with multiple strings (e.g., `["jpg", "png"]`) — DALI interprets
  each list item as a separate output channel and crashes with
  `ValueError: too many values to unpack`.
- **`missing_component_behavior="skip"`** — Samples without a matching image file
  are silently skipped.
- **DALI startup index scan** — On the first run, DALI scans tar headers to build
  an in-memory index (`Index file not provided, it may take some time...`). This
  is a one-time cost and is safe to ignore.
- **Not supported on Windows** — DALI requires Linux + CUDA. The codebase
  automatically falls back to the CPU loader on Windows.
