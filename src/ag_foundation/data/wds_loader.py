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
    """Wraps a WebDataset pipeline so that it reports a finite ``__len__``.

    The ``__iter__`` method enforces the epoch boundary by stopping after
    exactly ``length`` batches, regardless of how many workers are running.
    This avoids the PyTorch DataLoader warning about IterableDataset length
    mismatch that occurs when ``with_epoch`` cuts off per-worker rather than
    globally.
    """

    def __init__(self, pipeline, length: int) -> None:
        self.pipeline = pipeline
        self.length = length

    def __iter__(self):
        count = 0
        for batch in self.pipeline:
            if count >= self.length:
                return
            yield batch
            count += 1

    def __len__(self) -> int:
        return self.length

import random
import torchvision.transforms.functional as TF
from PIL import Image

class BoundedMultiscaleCrop:
    """
    Solves extreme scale variance across disparate datasets (from 64px to 6000px).
    1. Upscales images smaller than crop_size.
    2. Takes a random 40-100% scale crop of the image (which is already bounded to 1024px in the shards).
    3. Resizes the resulting crop exactly to crop_size x crop_size.
    """
    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        min_side = min(h, w)
        
        # Stage 1: Gently upscale very small images
        if min_side < self.crop_size:
            # We add a buffer so we still have room to do a random crop
            image = TF.resize(image, [self.crop_size + 32, self.crop_size + 32], antialias=True)
            w, h = image.size
            
        # Stage 2: Scale-restricted random crop (40-100% of image area)
        scale = random.uniform(0.4, 1.0)
        crop_h = max(int(h * scale), self.crop_size)
        crop_w = max(int(w * scale), self.crop_size)
        
        top = random.randint(0, h - crop_h) if h > crop_h else 0
        left = random.randint(0, w - crop_w) if w > crop_w else 0
        image = TF.crop(image, top, left, crop_h, crop_w)
        
        # Stage 3: Final resize to exact dimensions for tensor batching
        image = TF.resize(image, [self.crop_size, self.crop_size], antialias=True)
        
        # Basic augmentations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
            
        return image

def build_wds_dataloader(
    tar_urls: list[str],
    batch_size: int = 64,
    num_workers: int = 8,
    epoch_batches: int | None = None,
    crop_size: int = 224,
) -> DataLoader:
    """Build a WebDataset DataLoader that streams TAR shards on the CPU.

    Images are dynamically multi-scale cropped to ``crop_size × crop_size`` and converted to
    ``float32`` tensors in ``[0, 1]`` with shape ``(C, H, W)``. 

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
            BoundedMultiscaleCrop(crop_size=crop_size),
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
        dataset: IterableDataset = SizedWebDataset(pipeline, epoch_batches)
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
