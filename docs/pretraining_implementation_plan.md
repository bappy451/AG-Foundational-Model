# Pretraining Implementation Plan — Robustness & Compatibility Update

> Last updated: 2026-06-24

## Summary

This document captures the robustness and architectural fixes applied to the
AG Foundation Model pretraining pipeline based on real-world operational
observations.  All changes follow TDD: tests were written first, then the
source was updated to make them pass.

---

## Changes Implemented

### 1. Undersized Image Zero-Padding ✅

**Problem:** Images smaller than `crop_size` crashed the `DataLoader` with a
`ValueError`, causing training to abort on any small tile in the corpus.

**Solution:** `AgricultureImageDataset.__getitem__` and
`MultiSourcePretrainingDataset.__getitem__` now zero-pad images on the
right/bottom to reach `crop_size` using `torch.nn.functional.pad`.

**Files changed:**
- `src/ag_foundation/data/dataset.py` — added `F.pad` path and `F` import
- `src/ag_foundation/data/multi_source_dataset.py` — applied same fix

**Tests added (`tests/test_data_loading.py`):**
- `test_dataset_pads_undersized_image_to_crop_size`
- `test_dataset_pads_undersized_image_only_in_one_dimension`
- `test_dataset_zip_undersized_image_is_padded`

---

### 2. GeoTIFF NoData Integer Clamping ✅

**Problem:** GIS/GeoTIFF files encode missing pixels with extreme negative
integers (e.g. `INT32_MIN = -2147483647`).  The `_normalize_image_array`
function raised `ValueError` on any negative integer value.

**Solution:** Before normalization, signed-integer arrays are clipped to
`[0, max]` with `np.clip(array, a_min=0, a_max=None)`.  This neutralizes
NoData pixels without discarding the image.

**Files changed:**
- `src/ag_foundation/data/dataset.py` — replaced `ValueError` with `np.clip`

**Tests added (`tests/test_data_loading.py`):**
- `test_normalize_image_array_clamps_negative_integers_to_zero`
- `test_normalize_image_array_clamps_negative_int16_nodata`
- `test_normalize_positive_integers_unchanged`

---

### 3. RoPE Backbone Compatibility ✅

**Problem:** DINOv3 and EVA02 backbones use Rotary Position Embeddings (RoPE)
instead of absolute positional embeddings.  Their `patch_embed` returns a 4D
`(B, C, H, W)` tensor (not the standard `(B, N, C)`) and `backbone.pos_embed`
is `None`.  Both caused shape errors or `RuntimeError` crashes.

**Solution:** `RemoteSensingViT` was patched:
1. `embed_patches`: if `patch_tokens.ndim == 4`, flatten and transpose to
   `(B, N, C)` before returning.
2. `add_position_embeddings`: if `backbone.pos_embed is None`, skip the
   absolute positional embedding step and return directly.

**Files changed:**
- `src/ag_foundation/models/official_vit.py`

**Tests added (`tests/test_models.py`):**
- `test_vit_handles_rope_4d_patch_embed_output`
- `test_vit_handles_missing_pos_embed_rope_backbones`

---

<<<<<<< HEAD
### 4. TAR Archive and 5.17M Catalog Integration ✅

**Problem:** The 3 large PlantCLEF datasets were distributed as nested TAR/TAR.GZ archives containing over 3 million images, which `dataset.py` natively did not support. The data loading relied purely on crawling ZIPs, omitting 655GB of valid imagery. Furthermore, ground-truth masks were implicitly getting read into the training loader.

**Solution:** 
1. Added a `scripts/build_pretraining_catalog.py` workflow to index all images without extraction, generating a `catalog.csv` of 5,175,016 validated images.
2. Filtered out all `_mask`, `labels`, and `Evaluation/` directories in the catalog generator, and double-filtered in `dataset.py`'s catalog loading routine.
3. Added `_read_tar_member` using `tarfile` streaming to `dataset.py` allowing instant random-access to TAR elements for PlantCLEF support.

**Files changed:**
- `src/ag_foundation/data/dataset.py` — added `tarfile` streaming support and tight `_is_ground_truth_path` filters.
- `scripts/build_pretraining_catalog.py` — new index builder script.
- All configs (`smoke_test.yaml`, `pretraining_full.yaml`, etc.) updated to use the new catalog.

---

### 5. Multi-Source Dataset Bug Fixes ✅
=======
### 4. Multi-Source Dataset Bug Fixes ✅
>>>>>>> 33c63a88879f064cce6e7e60a11fa3ba55e170bd

Several latent bugs in `multi_source_dataset.py` were also corrected:

| Bug | Location | Fix |
| --- | --- | --- |
| Dead `continue` after `except` block | `_build_from_roots` line 183 | Removed unreachable statement |
| Duplicate `_load_image(record)` call | `__getitem__` line 302 | Removed second call |
| Unused `random` import | top-level | Removed |
| Unused `numpy` import | top-level | Replaced with `F` import |

---

<<<<<<< HEAD
### 6. DataLoader Memory & I/O Bottleneck Fix (5.17M Items) ✅

**Problem:** Scaling to 5.17 million images caused the initialization script to allocate millions of heavy Python `ImageRecord` objects, skyrocketing system RAM usage over 30 GB and causing the garbage collector to thrash. In addition, 5.17 million absolute path resolutions using `Path.resolve()` created a severe Windows I/O lock, stalling initialization for an hour.

**Solution:** 
1. Rewrote the `_resolve_records` logic to store lightweight `tuple[str, str]` primitives directly from pandas, drastically dropping RAM to <1GB.
2. The `__getitem__` method now instantiates `ImageRecord` lazily on-the-fly.
3. Path resolution was accelerated by grouping archives and caching via `@functools.lru_cache`.

**Files changed:**
- `src/ag_foundation/data/dataset.py`

---

### 7. RTX 4090 / PyTorch Resource Optimization ✅

**Problem:** Default arguments severely bottlenecked GPU utilization (`num_workers=0`, `precision=fp32`), leaving 24 logical CPU cores idle and wasting GPU VRAM. Standard fp32 computation also leaves Ada Lovelace Tensor Cores heavily underutilized.

**Solution:**
1. Altered runner defaults to `num_workers=8`, `prefetch_factor=4`, and `precision=bf16`.
2. Injected `torch.set_float32_matmul_precision("high")` and `torch.backends.cudnn.benchmark = True` in the base `SSLTrainer` to ensure operations utilize native hardware tensor cores efficiently.
3. Added a `--compile` flag to allow optional PyTorch 2.0 graph compilation.

**Files changed:**
- `src/ag_foundation/training/mim_runner.py`
- `src/ag_foundation/training/dino_runner.py`
- `src/ag_foundation/training/ssl_trainer.py`

---

=======
>>>>>>> 33c63a88879f064cce6e7e60a11fa3ba55e170bd
## Test Results

```
124 passed, 1 skipped in ~13s
```

The 1 skip is the Windows bash simulation test (expected and intentional).

---

## Smoke Test Configuration

`configs/smoke_test.yaml` — 2-epoch full-dataset run on RTX 4090:

```yaml
epochs: 2
batch_size: 8
gradient_accumulation_steps: 4   # effective batch = 32
precision: bf16
warmup_epochs: 1
visualization_every: 1
data_root: ../Pretraining          # all ~40 source datasets
```

Run command:

```bash
python -m ag_foundation train-dino --config configs/smoke_test.yaml
```
