# Reproducibility

## Reproducibility Layers

The project captures four layers:

1. Code: Git metadata in the run manifest
2. Configuration: command, resolved arguments, and YAML
3. Environment: platform, Python executable, package versions, and accelerator
4. State: model, optimizer, scaler, scheduler, RNG, loader generator, and history

## Seeding

At startup, the configured seed initializes:

- Python `random`
- NumPy
- Torch CPU
- all CUDA generators when available
- DataLoader shuffle generator
- deterministic group split
- worker-local Python and NumPy seeds

CUDA cuDNN deterministic mode is enabled and benchmark mode is disabled.

## Resume Semantics

Resume restores:

- model and optimizer
- scheduler and CUDA gradient scaler when present
- history and best metric
- Python random state
- NumPy random state
- Torch CPU state
- CUDA state list when available
- MPS state when available
- training DataLoader generator state

When a complete checkpoint is available, model construction skips the external
ImageNet-weight download and loads the checkpoint directly. The manifest records
whether initialization came from `timm` or a resume checkpoint.

Checkpoint replacement is atomic. The best checkpoint is hard-linked from the
newly written latest checkpoint when the filesystem supports it, with a copy
fallback on other filesystems. Every invocation also preserves an immutable
timestamped manifest bundle under the run's `attempts/` directory.

Visualization temporarily consumes model/data randomness, then restores both RNG
and loader state so plotting does not alter the next training epoch.

## Remaining Sources Of Variation

Bitwise equality across different hardware, PyTorch versions, drivers, or
distributed topologies is not guaranteed. Some accelerator kernels may remain
nondeterministic. Report mean and dispersion across seeds for scientific claims.

## Portable Paths

- YAML paths are relative to the YAML file.
- Catalog paths are relative to `data_root`.
- `::member/path` addresses a member of the configured root archive.
- CLI path overrides follow the current working directory.
- Run manifests intentionally store resolved absolute paths as provenance.

## Clone And Resume On Another Machine

1. Clone the same commit.
2. Recreate the environment.
3. place data at the config-relative location or update only the YAML path.
4. Copy the run directory or at least `last.pt`.
5. Run with the same config and a final `epochs` value.

```bash
python -m ag_foundation train-mim \
  --config configs/my_mim.yaml \
  --resume-from /artifact-store/my_run/last.pt
```

## Recommended Research Practice

- Freeze a dataset manifest before final pretraining.
- Record licenses and redistribution constraints.
- Hash source files or archive members.
- Deduplicate before splitting.
- Split by farm/region/time rather than only class folders.
- Run at least three seeds for key comparisons.
- Do not tune on the held-out external test domains.
- Preserve failed runs and their command logs during method development.
