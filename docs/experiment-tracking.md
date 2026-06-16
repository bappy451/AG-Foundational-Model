# Experiment Tracking And Artifacts

## Command Log

Every public CLI invocation and training wrapper logs by default.

Default:

```text
command.log
```

Each append-only entry contains:

- start and finish timestamps with timezone
- exact command and arguments
- working directory
- process ID
- live stdout and stderr
- duration
- exit status

Use a run-local log:

```bash
python -m ag_foundation train-mim \
  --config configs/my_mim.yaml \
  --log-file runs/my_mim/command.log
```

Shell and PowerShell wrappers mark logging as wrapper-owned so the Python CLI
does not duplicate the same output.

## Run Manifest

`run_manifest.json` records:

- generated timestamp
- command name, argv, and normalized command text
- absolute output directory
- all resolved arguments
- train/validation sample and batch counts
- data root, group counts, and group previews
- model type and full string representation
- total and trainable parameter counts
- adapter input/output channels
- official backbone name, embedding size, patch size, and image size
- requested initialization, actual `timm` loading, and resume checkpoint
- operating system, Python executable, hostname, and working directory
- Torch, torchvision, and `timm` versions
- accelerator count
- Git commit, branch, and dirty status when available

`resolved_config.yaml` is a convenient flat copy of the same resolved arguments.

## Scalar Metrics

`metrics.csv` is the analysis-friendly table. `metrics.json` additionally stores
system timing, precision, best metric, and final summary.

Per-epoch fields:

- epoch
- training loss
- validation loss
- epoch duration
- learning rate
- actual DINO teacher momentum used on the final batch when applicable

Metric files are rewritten atomically at epoch boundaries, so a completed epoch
remains inspectable during a long run.

## Figures

Shared:

- `figures/training_metrics.png`

MIM:

- `mim_reconstruction_epoch_XXXX.png`
- `mim_reconstruction_latest.png`

Each row shows adapted input, masked input, and reconstruction with visible
patches copied from the target.

DINO:

- `dino_views_epoch_XXXX.png`
- `dino_views_latest.png`
- `dino_similarity_epoch_XXXX.png`
- `dino_similarity_latest.png`
- `diagnostics/dino_similarity_epoch_XXXX.csv`

The similarity matrix compares normalized student and teacher features for a
representative sample. It is a training diagnostic, not a downstream accuracy
metric.

## Checkpoints

`last.pt` is atomically replaced every completed epoch. `best.pt` is atomically
snapshotted from the same serialized checkpoint when validation loss improves;
training loss is used only when no validation loader exists. This avoids
serializing very large ViT checkpoints twice per improving epoch.

Payload:

```text
epoch
model_state_dict
optimizer_state_dict
scheduler_state_dict
grad_scaler_state_dict
history
best_metric
run_config
rng_state
train_loader_generator_state
```

DINO model state includes student and teacher adapters, student and teacher
backbones, projection heads, and center buffer. MIM state includes the adapter,
backbone, mask token, and reconstruction head.

## Attempt History

The top-level manifest/config/model/command files always describe the latest
invocation. Every invocation also receives an immutable timestamped bundle under
`attempts/<timestamp>/`, so resuming a run does not erase the original launch
configuration or environment record.

## What To Archive For A Paper

- Git commit or release tag
- exact YAML
- command log
- run manifest
- metrics CSV/JSON
- best checkpoint
- figures and diagnostics
- dataset manifest, license record, and deduplication report
- seed-level downstream results

Large checkpoints should use institutional storage, an artifact registry, or
Git LFS rather than normal Git history.
