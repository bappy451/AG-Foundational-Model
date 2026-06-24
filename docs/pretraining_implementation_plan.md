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

### 4. Multi-Source Dataset Bug Fixes ✅

Several latent bugs in `multi_source_dataset.py` were also corrected:

| Bug | Location | Fix |
| --- | --- | --- |
| Dead `continue` after `except` block | `_build_from_roots` line 183 | Removed unreachable statement |
| Duplicate `_load_image(record)` call | `__getitem__` line 302 | Removed second call |
| Unused `random` import | top-level | Removed |
| Unused `numpy` import | top-level | Replaced with `F` import |

---

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
