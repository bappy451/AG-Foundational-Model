# Project Runbook And Audit Report

This report is the practical operating guide for `AG_Foundational_Model`. It
answers three questions:

1. What should I run, step by step?
2. What should I expect after each command?
3. What should I do after the project runs correctly?

It reflects the implementation audited and patched through 2026-06-29.

## Current Project Status

The project is ready for engineering-scale self-supervised pretraining runs on a
single workstation and on Google Colab with an A100/T4 GPU. It supports:

- official `timm` ViT-S, ViT-B, and ViT-L backbones
- ImageNet, DINOv2, DINOv3, and MAE checkpoint initialization
- RGB, GeoTIFF, multispectral NPY, folder, ZIP, TAR, TAR.GZ, and nested-ZIP inputs
- a learnable 1x1 band adapter before the official RGB ViT backbone
- MIM and DINO-style pretraining
- continual pretraining through `initialize_from`
- gradient accumulation for 24 GB GPUs
- command logs, manifests, metrics, figures, checkpoints, and diagnostics
- **zero-padding of undersized images** (small tiles are padded, not rejected)
- **GeoTIFF NoData handling** (extreme negative integers clamped to 0)
- **RoPE backbone compatibility** (DINOv3/EVA02 4D patch embed + no absolute pos embed)
- **WebDataset epoch boundary enforcement** (exact per-epoch batch count regardless of `num_workers`)
- **Windows multiprocessing safety** (fully pickleable `SizedWebDataset` wrapper)
- **DALI multi-format support** (semicolon-separated `ext` for single-image multi-format shards)
- **Google Colab GPU training** via NVIDIA DALI (`--use-dali` flag)

The main operational limitation is intentional: `initialize_from` requires a
compatible checkpoint. The source and target should use the same ViT family,
patch source/crop geometry, and input channel count. For example, a ViT-B
five-band MIM checkpoint can initialize a ViT-B five-band DINO run, but it
should not initialize a ViT-S three-band run.

## Hardware Plan

For your available machine:

- GPU: RTX 4090 24 GB
- RAM: 64 GB
- CPU: Core i9

Start with ViT-S or ViT-B. Use ViT-L only after ViT-B is stable, because DINO
multi-crop training and large projection heads can consume memory quickly.

Recommended first serious settings:

- `precision: bf16`
- `batch_size: 4` to `8`
- `gradient_accumulation_steps: 2` to `8`
- `gradient_checkpointing: true` for ViT-B and ViT-L
- `num_workers: 4` initially
- `pretrained_backbone: true`
- `pretrained_source: imagenet`, `dinov2`, `dinov3`, or `mae`

Effective batch size is:

```text
batch_size * gradient_accumulation_steps
```

Example: `batch_size: 4` and `gradient_accumulation_steps: 8` gives an
effective batch size of 32 while keeping GPU memory lower.

## Step 1: Clone And Enter The Project

```bash
git clone <repository-url>
cd AG_Foundational_Model
```

Expected result:

- You are inside the project root.
- `README.md`, `configs/`, `docs/`, `scripts/`, `src/`, and `tests/` exist.

Check:

```bash
pwd
ls
```

What to do next:

- Continue to environment setup.

## Step 2: Create A Clean Python Environment

We strongly recommend using **Conda** to resolve CUDA and spatial dependencies (`rasterio`) smoothly. An `environment.yml` file is provided in the project root.

```bash
conda env create -f environment.yml
conda activate ag-foundation
```

This will install Python 3.12, PyTorch with CUDA 12.1 support, `timm`, `rasterio`, and the project in editable mode (`-e .[dev,ml]`).

Alternatively, if using `venv` or `virtualenv`:

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,ml]'
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
```

Expected result:

- The package is installed in editable mode.
- `torch`, `torchvision`, `timm`, `rasterio`, `matplotlib`, `pytest`, and
  `ruff` are available.

Check:

```bash
python -m pip show torch timm rasterio pytest ruff
```

What to do if this fails:

- On CUDA Linux, install the PyTorch build recommended by PyTorch for your
  driver first, then run `python -m pip install -e '.[dev,ml]'`.
- If a wrapper uses the wrong interpreter, pass it explicitly with
  `--python /path/to/python`.
- The training wrappers now check for `torch` and `timm` before launching and
  print the install command if the selected interpreter is incomplete.

## Step 3: Verify The CLI

```bash
python -m ag_foundation --help
ag-foundation --help
```

Expected result:

- Both commands print the available CLI commands.
- You should see `train-mim`, `train-dino`, `create-demo-data`,
  `create-catalog`, `slice-geotiffs`, and `audit-pretraining-data`.

What to do next:

- Run the automated tests before training.

## Step 4: Run The Automated Test Suite

```bash
python -m pytest -q
python -m ruff check README.md project_description.md docs src tests scripts configs
```

Expected result:

- Pytest exits with all tests passing.
- Ruff prints `All checks passed!`.

What this proves:

- Data loading works across RGB, NPY, GeoTIFF, ZIP, and nested ZIP.
- Undersized images are zero-padded instead of rejected.
- GeoTIFF NoData integers are clamped to 0 instead of crashing.
- RoPE-based backbones (DINOv3, EVA02) produce valid patch sequences.
- Config validation and config-relative paths work.
- Official ViT selection logic works with a deterministic test double.
- MIM and DINO tensor paths work.
- Metrics, checkpoints, manifests, figures, resume, and gradient accumulation
  are covered.

What this does not prove:

- It does not prove representation quality.
- It does not run a long GPU experiment.
- It does not prove the dataset is publication-ready.

## Step 5: Generate Demo Data

```bash
python -m ag_foundation create-demo-data
```

Expected result:

- `data/demo/rgb/` contains deterministic RGB PNG images.
- `data/demo/multispectral/` contains deterministic five-band NPY arrays.
- `data/demo/dataset_summary.json` is written.

What to do next:

- Use this synthetic data to verify real training entrypoints.

## Step 6: Run A MIM Smoke Training Job

```bash
bash scripts/train_mim.sh --config configs/demo_mim.yaml
```

Windows PowerShell:

```powershell
.\scripts\train_mim.ps1 --config .\configs\demo_mim.yaml
```

Expected terminal output:

- A `Running:` line showing the selected Python interpreter.
- A manifest message similar to `[metadata] Saved run manifest ...`.
- Batch progress lines with `loss`, `avg`, `lr`, and `update`.
- A final `SSLTrainingSummary(...)`.

Expected files:

```text
runs/demo_mim/
|-- best.pt
|-- last.pt
|-- metrics.csv
|-- metrics.json
|-- run_manifest.json
|-- resolved_config.yaml
|-- model_summary.txt
`-- figures/
    |-- training_metrics.png
    `-- mim_reconstruction_epoch_0001.png
```

What to inspect:

```bash
ls runs/demo_mim
python - <<'PY'
import json
from pathlib import Path

metrics = json.loads(Path("runs/demo_mim/metrics.json").read_text())
print(metrics["summary"])
PY
```

What to do next:

- If this passes, MIM training, the multispectral adapter, metrics, figures,
  checkpointing, and logging are working.
- Continue to DINO smoke training.

## Step 7: Run A DINO Smoke Training Job

```bash
bash scripts/train_dino.sh --config configs/demo_dino.yaml
```

Windows PowerShell:

```powershell
.\scripts\train_dino.ps1 --config .\configs\demo_dino.yaml
```

Expected terminal output:

- Batch progress lines with `loss`, `avg`, `lr`, and `ema`.
- A validation loss after the epoch.
- A final `SSLTrainingSummary(...)`.

Expected files:

```text
runs/demo_dino/
|-- best.pt
|-- last.pt
|-- metrics.csv
|-- metrics.json
|-- run_manifest.json
|-- resolved_config.yaml
|-- model_summary.txt
|-- diagnostics/
|   `-- dino_similarity_epoch_0001.csv
`-- figures/
    |-- training_metrics.png
    |-- dino_views_epoch_0001.png
    `-- dino_similarity_epoch_0001.png
```

What to inspect:

```bash
ls runs/demo_dino
head -n 5 runs/demo_dino/metrics.csv
```

What to do next:

- If this passes, DINO training, student-teacher EMA, multi-crop views, metrics,
  figures, checkpointing, and diagnostics are working.

## Step 8: Verify DINOv2 Patch-14 Initialization

DINOv2 ViT models use patch size 14, so the crop must be divisible by 14.

```bash
bash scripts/train_dino.sh \
  --config configs/demo_dino.yaml \
  --crop-size 28 \
  --pretrained-source dinov2 \
  --output-dir runs/demo_dino_dinov2
```

Expected result:

- The run completes with one epoch.
- `run_manifest.json` records `pretrained_source: dinov2`.
- `last.pt`, `best.pt`, metrics, and DINO figures are created.

What to do if this fails:

- Use a crop size divisible by 14, such as `224`, `280`, or `336`.
- Confirm internet access or a populated Hugging Face/timm cache for the first
  DINOv2 download.

## Step 9: Verify MIM To DINO Continual Pretraining

Use a checkpoint with matching backbone and input channels. This example starts
DINO from the MIM smoke checkpoint.

```bash
bash scripts/train_dino.sh \
  --config configs/demo_dino.yaml \
  --data-root data/demo/multispectral \
  --output-dir runs/demo_dino_from_mim \
  --model-name B \
  --channels 5 \
  --crop-size 32 \
  --pretrained-source imagenet \
  --initialize-from runs/demo_mim/last.pt
```

Expected result:

- The run starts from the MIM model weights.
- The optimizer, epoch counter, and prior history are not restored.
- `run_manifest.json` records the `initialize_from` checkpoint path.

What to do if this fails:

- Match `model_name`, `channels`, crop/patch geometry, and pretrained source
  between the source checkpoint and target run.
- Use `resume` only for exact continuation of the same run.
- Use `initialize_from` for a new stage that starts from previous weights.

## Step 10: Audit Your Pretraining Dataset

From the project root:

```bash
python scripts/analyze_pretraining_dataset.py \
  --pretraining-root ../Pretraining \
  --dataset-list ../Pretraining/Dataset.txt \
  --output-dir reports/pretraining_dataset_audit
```

Expected files:

```text
reports/pretraining_dataset_audit/
|-- pretraining_dataset_audit.csv
|-- pretraining_dataset_audit.json
`-- pretraining_dataset_audit.md
```

What to inspect:

- total image count
- file formats
- source coverage
- likely task types
- missing source archives
- whether GeoTIFF and multispectral data are present
- recommended additions

What to do next:

- Use this audit every time you add or remove data.
- Keep dataset composition tables for the paper.
- Add multispectral/GeoTIFF sources if the current corpus is RGB-heavy.

## Step 10.5: Full-Dataset 2-Epoch Smoke Test (RTX 4090)
## Step 10.5: Full-Dataset 2-Epoch Smoke Test (RTX 4090)

Before a long pretraining campaign, run the full-dataset smoke test to confirm
the entire pipeline works end-to-end on real data.

Activate the environment:

```powershell
conda activate venv
cd E:\AG_Dataset\AG-Foundational-Model
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Verify dataset sources are detected:

```powershell
python -c "
from ag_foundation.data.multi_source_dataset import scan_pretraining_directory
sources = scan_pretraining_directory('../Pretraining')
print(f'Detected {len(sources)} source datasets')
"
```

Run the 2-epoch smoke test:

```powershell
python -m ag_foundation train-dino --config configs/smoke_test.yaml
```

Expected output directory: `E:\AG_Dataset\runs\smoke_test_dino\`

Expected files after completion:

```text
runs/smoke_test_dino/
|-- best.pt
|-- last.pt
|-- metrics.csv
|-- manifest.json
`-- figures/
    |-- training_metrics.png
    |-- dino_views_ep1.png
    `-- dino_views_ep2.png
```

Key smoke-test config values (`configs/smoke_test.yaml`):

| Setting | Value |
| --- | --- |
| epochs | 2 |
| batch_size | 8 |
| gradient_accumulation_steps | 4 (effective batch = 32) |
| precision | bf16 |
| data_root | ../Pretraining |

If CUDA OOM occurs, reduce `batch_size` to 4 and increase
`gradient_accumulation_steps` to 8 to keep effective batch = 32.

Resume works automatically: re-running the same command picks up from `last.pt`.

## Step 11: Build Pretraining Data Structures

For smaller datasets, the clean pretraining catalog (`Pretraining/catalog.csv`) can index the images
across your archives.

To regenerate the catalog from scratch (e.g. after adding new datasets):

```powershell
conda activate ag-foundation
cd E:\AG_Dataset\AG-Foundational-Model
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
python -u scripts/build_pretraining_catalog.py
```

### High-Performance WebDataset Shards (Recommended)
For massive datasets (~1 TB), we heavily recommend building WebDataset (`.tar`) shards for maximum I/O performance on Linux and Google Colab.

```powershell
python src\ag_foundation\data\build_wds_shards.py \
  --input-dir "E:\AG_Dataset\AG-Foundational-Model\Pretraining" \
  --output-prefix "E:\AG_Dataset\shards\dataset" \
  --max-size 1000000000
```
This generates ~1GB `.tar` shards which can be sequentially streamed across the network directly into GPU memory using DALI.

## Step 11.5: Google Colab GPU Training (A100 / T4)

The project supports training directly in Google Colab using the NVIDIA DALI GPU
loader for maximum throughput. Your WebDataset shards must be accessible from
Colab (e.g., mounted from Google Drive).

### Setup in Colab

```python
# 1. Mount Google Drive (if your shards are stored there)
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to your project directory
import os
os.chdir('/content/drive/MyDrive/Colab_Projects/AG-Foundational-Model')

# 3. Install the package in editable mode
!pip install -e .[ml]

# 4. Verify NVIDIA DALI is installed (included in the ml extras)
!python -c "import nvidia.dali; print('DALI OK')"
```

### Run MIM Pretraining on Colab

```python
!python -m ag_foundation train-mim \
    --config configs/wds_mim_pretrain.yaml \
    --use-dali
```

### Run DINO Pretraining on Colab

```python
!python -m ag_foundation train-dino \
    --config configs/wds_dino_pretrain.yaml \
    --use-dali
```

### Expected DALI Startup Messages (Safe to Ignore)

When training starts, DALI will print two informational messages that are completely safe:

```
Warning: Please set `reader_name` and don't set last_batch_padded and size manually...
[webdataset_loader.cc] Index file not provided, it may take some time to infer it from the tar file
```

The first warning is because we manually set the epoch size. The second means DALI
is scanning tar headers to build an index — this is a one-time cost at startup.
Training will proceed normally after the scan completes.

### Colab Config Tips

- Set `data.num_workers: 2` on Colab (CPU workers for prefetching are shared).
- Set `runtime.precision: bf16` on A100 for maximum speed.
- Set `runtime.precision: fp16` on T4.
- Set `data.epoch_batches` to limit each epoch to a manageable number of batches
  (e.g., `50000`) so checkpoints are saved frequently.

## Step 12: Configure A Real MIM Run

Create a working config:

```bash
cp configs/train_mim.example.yaml configs/mim_vit_b_ag.yaml
```

Edit the important fields:

```yaml
data:
  data_root: ../Pretraining
  catalog_path: ../Pretraining/catalog.csv
  crop_size: 224
  channels: 3
  batch_size: 4
  num_workers: 8
  val_fraction: 0.05

runtime:
  output_dir: ../runs/mim_vit_b_ag
  epochs: 100
  precision: bf16
  device: cuda
  gradient_accumulation_steps: 8
  resume: true

model:
  model_name: B
  pretrained_backbone: true
  pretrained_source: mae
  mask_ratio: 0.75
  gradient_checkpointing: true
```

Run:

```bash
bash scripts/train_mim.sh --config configs/mim_vit_b_ag.yaml
```

Expected result:

- The first run may download official weights.
- The training output directory fills with checkpoints, metrics, figures, and
  manifests.

What to watch:

- GPU memory with `nvidia-smi`.
- Training loss should be finite.
- Validation loss should be finite.
- `metrics.csv` should gain one row per epoch.

If memory is too high:

- reduce `batch_size`
- increase `gradient_accumulation_steps`
- keep `gradient_checkpointing: true`
- reduce `crop_size`
- start with ViT-S

## Step 13: Configure A Real DINO Run

For direct DINO pretraining, a fully optimized configuration for an RTX 4090 is provided at `configs/pretraining_full.yaml`. It utilizes `bf16`, gradient accumulation, and handles multiple data roots via the multi-source dataloader.

```bash
cp configs/pretraining_full.yaml configs/dino_vit_s_ag.yaml
```

Recommended first DINO config for a 4090:

```yaml
data:
  data_root: ../Pretraining
  catalog_path: ../Pretraining/catalog.csv
  crop_size: 224
  channels: 3
  batch_size: 4
  num_workers: 8
  val_fraction: 0.05

runtime:
  output_dir: ../runs/dino_vit_s_ag
  epochs: 100
  precision: bf16
  device: cuda
  gradient_accumulation_steps: 8
  resume: true

model:
  model_name: S
  pretrained_backbone: true
  pretrained_source: dinov3
  dino_out_dim: 65536
  num_global_crops: 2
  num_local_crops: 4
  gradient_checkpointing: true
```

Run:

```bash
bash scripts/train_dino.sh --config configs/dino_vit_s_ag.yaml
```

For MIM to DINO continual pretraining:

```yaml
runtime:
  output_dir: ../runs/dino_vit_b_from_mim
  initialize_from: ../runs/mim_vit_b_ag/last.pt
  resume: false

model:
  model_name: B
```

Run:

```bash
bash scripts/train_dino.sh --config configs/dino_vit_b_from_mim.yaml
```

Expected result:

- DINO creates `dino_views_epoch_XXXX.png`,
  `dino_similarity_epoch_XXXX.png`, and diagnostics CSVs.
- The manifest records whether weights came from `timm`, `resume_from`, or
  `initialize_from`.

## Step 14: Monitor A Running Experiment

In another terminal:

```bash
tail -f command.log
watch -n 2 nvidia-smi
tail -f runs/mim_vit_b_ag/metrics.csv
```

Expected result:

- `command.log` shows live stdout/stderr and a final exit footer.
- `nvidia-smi` shows GPU memory and utilization.
- `metrics.csv` grows once per epoch.

Useful inspection commands:

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("runs/mim_vit_b_ag")
manifest = json.loads((run / "run_manifest.json").read_text())
metrics = json.loads((run / "metrics.json").read_text())
print("model:", manifest["model"]["type"])
print("init:", manifest["model"]["initialization"])
print("summary:", metrics["summary"])
PY
```

## Step 15: Resume Interrupted Training

If a run stops and `last.pt` exists:

```bash
bash scripts/train_mim.sh --config configs/mim_vit_b_ag.yaml --resume
```

Or explicitly:

```bash
bash scripts/train_mim.sh \
  --config configs/mim_vit_b_ag.yaml \
  --resume-from runs/mim_vit_b_ag/last.pt
```

Expected result:

- The trainer restores model, optimizer, scaler, history, RNG state, and epoch.
- Training continues toward `runtime.epochs`.

Important:

- `epochs` is the final target epoch, not additional epochs.
- If the checkpoint finished epoch 40 and config says `epochs: 100`, training
  resumes at epoch 41 and stops at epoch 100.

## Step 16: Decide What To Do After A Successful Run

After a run completes:

1. Archive the config, manifest, metrics, figures, and command log.
2. Record dataset version, image count, channel count, and source list.
3. Run at least three seeds for any claim you want to publish.
4. Evaluate frozen and fine-tuned representations on downstream agricultural
   tasks.
5. Compare against ImageNet-only ViT, MAE-only, DINO-only, and MIM to DINO.
6. Add RGB-only versus multispectral ablations.
7. Add out-of-domain tests across crop type, geography, season, sensor, and
   resolution.
8. Keep negative results; they are valuable for the paper and for debugging.

## Publication-Grade Experiment Ladder

Recommended sequence:

1. Engineering sanity: demo MIM and demo DINO pass.
2. Dataset audit: total counts, modality balance, and missing modalities known.
3. Small real-data smoke: ViT-S, one to three epochs, one seed.
4. Main MIM: ViT-B, 100+ epochs, three seeds.
5. Main DINO: ViT-S or ViT-B, 100+ epochs, three seeds.
6. Continual stage: MIM to DINO with matched model/channel settings.
7. Downstream tasks: classification, detection, segmentation, retrieval, and
   domain transfer where data are available.
8. Ablations: adapter on/off, pretrained source, objective, channels, crop size,
   data mixture, and model scale.
9. External validation: datasets not used during pretraining.
10. Paper package: dataset card, model card, configs, metrics, and training
    logs.

## Commands Used In The 2026-06-19 Audit

Automated tests:

```bash
python -m pytest -q
python -m ruff check README.md project_description.md docs src tests scripts configs
```

MIM smoke with official ImageNet initialization:

```bash
python scripts/ag_foundation.py create-demo-data --output-dir /tmp/ag-demo
bash scripts/train_mim.sh \
  --python /opt/anaconda3/bin/python \
  --config configs/demo_mim.yaml \
  --data-root /tmp/ag-demo/multispectral \
  --output-dir /tmp/ag-runs/mim \
  --model-name B \
  --pretrained-source imagenet \
  --device cpu
```

DINO smoke with official ImageNet initialization:

```bash
bash scripts/train_dino.sh \
  --python /opt/anaconda3/bin/python \
  --config configs/demo_dino.yaml \
  --data-root /tmp/ag-demo/rgb \
  --output-dir /tmp/ag-runs/dino \
  --model-name S \
  --pretrained-source imagenet \
  --device cpu
```

DINOv2 smoke:

```bash
bash scripts/train_dino.sh \
  --python /opt/anaconda3/bin/python \
  --config configs/demo_dino.yaml \
  --data-root /tmp/ag-demo/rgb \
  --output-dir /tmp/ag-runs/dino_dinov2 \
  --model-name S \
  --crop-size 28 \
  --pretrained-source dinov2 \
  --device cpu
```

Matched MIM to DINO handoff:

```bash
bash scripts/train_dino.sh \
  --python /opt/anaconda3/bin/python \
  --config configs/demo_dino.yaml \
  --data-root /tmp/ag-demo/multispectral \
  --output-dir /tmp/ag-runs/dino_from_mim_b \
  --model-name B \
  --channels 5 \
  --crop-size 32 \
  --pretrained-source imagenet \
  --device cpu \
  --initialize-from /tmp/ag-runs/mim/last.pt
```

Expected result from these audit commands:

- all test and lint gates pass
- MIM writes checkpoint, metrics, manifest, curve, and reconstruction figure
- DINO writes checkpoint, metrics, manifest, curve, views, similarity figure,
  and diagnostics CSV
- DINOv2 resolves the patch-14 official checkpoint path correctly
- matched MIM to DINO handoff records `initialize_from` in the manifest

## Readiness Summary

The project is ready for controlled pretraining experiments. The test suite
passes **124 tests** (1 skipped, Windows bash test). All critical robustness
patches are active:

- undersized image zero-padding (no more ValueError on small tiles)
- GeoTIFF NoData integer clamping (no more crashes on INT32 minimum)
- RoPE backbone compatibility (DINOv3/EVA02 work without code changes)

The next major work is scientific, not plumbing:

- run the 2-epoch full-dataset smoke test (Step 10.5)
- finish dataset balancing and license review
- add missing multispectral and GeoTIFF coverage
- run long MIM and DINO pretraining on the RTX 4090
- build downstream evaluation tasks
- collect ablations and external-domain results
- prepare a reproducible experiment package for CVPR or Computers and
  Electronics in Agriculture style review

## Advanced: Google Colab & NVIDIA DALI

For cloud deployments (like Google Colab instances running Ubuntu), you can leverage NVIDIA DALI for ultra-fast GPU-accelerated JPEG decoding. DALI decodes WebDataset `.tar` shards directly into GPU memory via `nvJPEG`, entirely bypassing CPU bottlenecks.

1. Install DALI on Colab (Ensure a T4, L4, or A100 runtime is selected):
```bash
!pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120
```

2. Run Pretraining with the `--use-dali` flag:
```bash
# MIM
python -m ag_foundation train-mim --config configs/wds_mim_pretrain.yaml --use-dali

# DINO
python -m ag_foundation train-dino --config configs/wds_dino_pretrain.yaml --use-dali
```
