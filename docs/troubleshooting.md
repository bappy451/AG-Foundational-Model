# Troubleshooting

## `No module named ag_foundation`

Install the repository package inside the active environment:

```bash
python -m pip install -e '.[dev,ml]'
```

The scripts can import directly from `src`, but the public
`python -m ag_foundation` entrypoint requires installation.

## `dataclass() got an unexpected keyword argument 'slots'`

That error is associated with Python older than 3.10 when code uses dataclass
slots. The current project supports Python 3.9 and does not require dataclass
slots. Verify that the active interpreter is the one where the package was
installed:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

## Pretrained Download Fails

- verify internet access
- set `HF_TOKEN` if rate-limited
- retry after checking available disk space
- pre-populate the Hugging Face/timm cache on offline machines

Do not silently switch publication experiments to random initialization.

## Channel Mismatch

The configured `data.channels` must exactly match the number of bands present in the imagery. For NPY arrays, ensure one axis unambiguously matches that count. 

**Exception:** If you configure `channels: 3` (e.g. to use standard RGB pretrained models), but you pass a single-channel image (like a grayscale image or a 1-band climatic GeoTIFF), the loader will automatically broadcast the 1 channel to 3 channels. This allows you to use official Hugging Face ImageNet ViT weights on single-band GIS data without encountering shape errors.

## GeoTIFF NoData Values Causing Crashes

GIS files frequently encode missing pixels as extreme negative integers
(e.g., `-2147483647` for INT32 NoData).  The loader now **clamps these values
to 0** automatically before normalization instead of raising a `ValueError`.

If you see unexpected black regions in reconstruction visualizations, this is
normal — those are NoData pixels rendered as 0 in float space.

For floating-point rasters, the pipeline still rejects NaN or out-of-range
values.  Apply calibration or clipping upstream for float32 rasters.

## Floating Raster Outside `[0, 1]`

Calibrate or normalize the raster before training. The loader intentionally
rejects arbitrary floating ranges because silent min-max scaling can destroy
cross-scene radiometric meaning.

## Image Smaller Than `crop_size`

Images smaller than `crop_size` are automatically **zero-padded** on the
right and bottom to match `crop_size`.  Padding is applied before the random
or center crop, so the padded region may appear as a black border in
visualization outputs.

If you want to avoid padding entirely:
- Tile or resize the source data before training.
- Reduce `crop_size` so it fits within all images.

## WebDataset Tarball Corrupted Image Errors

If you encounter a `webdataset.autodecode.DecodingError` or `PIL.UnidentifiedImageError` when running training over WebDataset Tarballs, this means an image inside the tarball is truncated or corrupted. 
Our WebDataset loader is configured to use `wds.warn_and_continue`, which gracefully ignores the corrupted image and continues training without crashing. You will see a warning in the terminal, but it will not halt your run.

## Crop Not Divisible By The Selected Patch Size

Use a crop that matches the selected backbone family. Examples:

- 224, 256, or 384 for 16-patch backbones
- 28, 56, or 224 for DINOv2's 14-patch backbone

If you switch `pretrained_source`, recheck the crop size before launching a
long run.

## Fewer Than Two Groups

Train/validation splitting requires at least two distinct catalog groups. Add a
meaningful `group` column or reorganize the directory hierarchy.

## Out Of Memory

If you encounter `torch.OutOfMemoryError: CUDA out of memory`:
- **Reduce `batch_size` and increase `gradient_accumulation_steps`** (e.g., if batch size drops from 64 to 16, increase accumulation steps by 4x to maintain the same effective batch size without exceeding VRAM).
- use fp16 or bf16 on supported hardware
- enable gradient checkpointing
- use ViT-S before ViT-B/L
- reduce DINO output/head dimensions during engineering tests
- reduce crop count or crop resolution

## Resume Runs No Epochs

`epochs` is the final target epoch. Increase it above the checkpoint's saved
epoch.

## Catalog Breaks After Moving Data

Regenerate it with the current `create-catalog` command. Portable catalogs use
relative paths or `::archive/member` paths. Old catalogs containing absolute
machine paths should not be reused.

## Windows Log File Is Locked

Terminate orphaned Python or PowerShell training processes, then retry with a
different `--log-file` path if necessary.

## Metrics Plot Has One Point

That is expected for a one-epoch demo. Publication runs should produce a curve
over many epochs.

## MIM Reconstruction Looks Noisy

Early one-epoch reconstructions are expected to be poor, especially on tiny
32 x 32 verification crops with only four patches. Judge MIM output after a
proper training schedule and resolution.
