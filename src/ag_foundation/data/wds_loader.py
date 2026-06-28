"""CPU-based WebDataset loader for streaming TAR shards.

This module is used on Windows / CPU-only environments.
For GPU-accelerated loading on Linux, use ``dali_wds_loader`` instead.
"""
from __future__ import annotations

import sys

import torch
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

if sys.platform == "win32":
    from webdataset.gopen import gopen_schemes

    gopen_schemes["winfile"] = lambda url, mode="rb", bufsize=8192, **kw: open(
        url.replace("winfile://", ""), mode, buffering=bufsize
    )


class SizedWebDataset(IterableDataset):
    """Wraps a WebDataset pipeline so that it reports a finite ``__len__``."""

    def __init__(self, pipeline, length: int) -> None:
        self.pipeline = pipeline
        self.length = length

    def __iter__(self):
        return iter(self.pipeline)

    def __len__(self) -> int:
        return self.length


def build_wds_dataloader(
    tar_urls: list[str],
    batch_size: int = 64,
    num_workers: int = 8,
    epoch_batches: int | None = None,
    crop_size: int = 224,
) -> DataLoader:
    """Build a WebDataset DataLoader that streams TAR shards on the CPU.

    Images are resized to ``crop_size × crop_size`` and converted to
    ``float32`` tensors in ``[0, 1]`` with shape ``(C, H, W)``.  Heavy
    augmentations (Kornia/DINO multi-crop) happen downstream on the GPU.

    Args:
        tar_urls:      List of shard paths / URLs (may include ``winfile://``
                       prefixes on Windows).
        batch_size:    Samples per batch.
        num_workers:   DataLoader worker processes.
        epoch_batches: Maximum batches per epoch.  ``None`` means unlimited.
        crop_size:     Height and width to resize each image to.

    Returns:
        A ``DataLoader`` that yields ``{"image": tensor}`` dicts.
    """
    to_tensor = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ]
    )

    pipeline = (
        wds.WebDataset(tar_urls, resampled=True)
        .shuffle(1000)
        .decode("pil", handler=wds.warn_and_continue)
        .rename(image="jpg;png;jpeg;tif;tiff", handler=wds.warn_and_continue)
        .map_dict(image=to_tensor)
        .batched(batch_size, partial=False)
    )

    if epoch_batches is not None:
        pipeline = pipeline.with_epoch(epoch_batches)
        dataset = SizedWebDataset(pipeline, epoch_batches)
    else:
        dataset = pipeline

    loader = DataLoader(
        dataset,
        batch_size=None,  # batching is handled inside the WebDataset pipeline
        num_workers=num_workers,
        pin_memory=True,  # crucial for fast CPU → GPU transfer
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return loader
