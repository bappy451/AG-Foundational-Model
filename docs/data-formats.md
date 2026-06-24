# Data Formats And Catalogs

## Supported Inputs

| Format | Extensions | Expected layout |
| --- | --- | --- |
| RGB image | `.jpg`, `.jpeg`, `.png` | Pillow-readable RGB |
| GeoTIFF | `.tif`, `.tiff` | rasterio bands-first |
| NumPy | `.npy` | `C,H,W`, `H,W,C`, or 2D single-band |
| Archive | `.zip` | supported images and nested ZIPs |

macOS metadata such as `__MACOSX` and `._*` members is skipped.

## Channel Rules

Always set `data.channels` for production experiments.

- RGB uses `channels: 3`.
- A four-band RGB-NIR product uses `channels: 4`.
- A five-band sensor uses `channels: 5`.
- **Single-channel sources (e.g. grayscale, single-band GeoTIFFs) are automatically broadcasted to 3 channels if `channels: 3` is requested**. This allows the official Hugging Face ViT ImageNet pretrained model (which strictly expects 3 input channels) to ingest 1-channel data natively without failing or requiring code modification.
- Every sample in one run must otherwise expose the exact same number of channels as configured.

For NPY data, specifying channels resolves whether the first or last axis is the
band axis. If both axes match or neither matches, loading fails with an explicit
error rather than guessing.

## Numeric Normalization

Non-negative integer images are converted to float32 and divided by the maximum
value of the integer dtype. Signed integer arrays may carry GIS/GeoTIFF NoData
values (e.g. `INT32_MIN = -2147483647`). These extreme negative integers are
automatically clamped to 0 before normalization so the pipeline remains stable
without discarding valid data. Examples:

- `uint8`: divide by 255
- `uint16`: divide by 65535
- `int32` with NoData: clamp to 0 first, then divide by 2147483647

Floating images must already be finite and within `[0, 1]`. NaN, infinity, or
out-of-range values raise an error. Sensor-specific percentile clipping,
reflectance calibration, and per-band standardization should be performed
before this generic loader when scientifically required.

## Spatial Rules

- Images smaller than `crop_size` in any dimension are **zero-padded** on the
  right/bottom to reach the required size.  This preserves small but valid
  images (e.g. tiny GIS tiles, edge-padded species photos) instead of
  discarding them.
- Training uses random crops and flips.
- MIM RGB training also applies color jitter to three-channel images.
- Validation uses deterministic center crops.
- DINO disables dataset color augmentation because its multi-crop augmenter
  owns the view transformations.
- `crop_size` must be divisible by the selected ViT patch size.

## Group-Disjoint Split

The loader derives a `group` from each sample's parent directory. It assigns
complete groups deterministically while minimizing the difference between the
requested validation sample count and the achievable group-disjoint count. No
group appears in both splits.

For rigorous experiments, make the catalog's `group` column represent the
leakage boundary you care about:

- source dataset
- farm
- field
- acquisition date
- geographic region
- plant instance

Class-folder grouping is not automatically equivalent to geographic or
instance-disjoint evaluation.

## Portable Catalog Format

Create a catalog with:

```bash
python -m ag_foundation create-catalog \
  --data-root /data/archive.zip \
  --output-path catalogs/archive.csv
```

Columns:

```csv
path,group
::archive/class_a/image_001.png,archive/class_a
::archive/class_b/image_002.png,archive/class_b
```

`::member/path` means "read this member from the archive specified by
`data_root`." For a directory root, ordinary files and nested archives are saved
relative to that root. This makes committed catalogs portable across machines.

Do not hand-edit a catalog to include the original machine's absolute path.

## GeoTIFF Training

Small GeoTIFF files may be loaded directly. For large scenes, tile them first:

```bash
python -m ag_foundation slice-geotiffs \
  --input-path /data/raw-scenes \
  --output-dir /data/tiles-224 \
  --tile-size 224 \
  --stride 224 \
  --output-format tif \
  --workers auto
```

TIFF output preserves:

- all bands
- dtype
- CRS
- per-tile affine transform
- source offsets and valid edge dimensions

When process-pool semaphores are unavailable, the slicer automatically falls
back to sequential processing instead of failing. This keeps GeoTIFF preparation
usable in restricted desktop, CI, or shared-cluster environments.

PNG/JPEG output converts to a three-channel display image and should not be used
when the additional spectral bands or original radiometry are required.

## Archives

ZIP members are read without extracting the complete archive. Each DataLoader
worker lazily opens a root archive once and reuses that handle, avoiding repeated
central-directory parsing for every sample. Nested ZIPs are supported. For very
large archives, random access may still be slower than a sharded or extracted
training layout; benchmark I/O before a long campaign.

## Demo Data

`create-demo-data` generates:

- four semantic source groups
- 24 RGB PNGs
- 24 five-band float32 NPY arrays
- deterministic patterns from seed 27
- `dataset_summary.json`

It is an engineering fixture, not a dataset contribution.
