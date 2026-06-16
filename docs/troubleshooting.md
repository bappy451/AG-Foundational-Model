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

The configured `data.channels` must equal every sample's band count. For NPY,
also ensure one axis unambiguously matches that count.

## Floating Raster Outside `[0, 1]`

Calibrate or normalize the raster before training. The loader intentionally
rejects arbitrary floating ranges because silent min-max scaling can destroy
cross-scene radiometric meaning.

## Crop Larger Than Image

Reduce `crop_size` or tile/resize the source data. The loader does not upscale
small images automatically.

## Crop Not Divisible By 16

Use a crop such as 224, 256, 384, or another multiple of 16.

## Fewer Than Two Groups

Train/validation splitting requires at least two distinct catalog groups. Add a
meaningful `group` column or reorganize the directory hierarchy.

## Out Of Memory

- reduce batch size
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

