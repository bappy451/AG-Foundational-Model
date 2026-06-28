"""
NVIDIA DALI WebDataset loader for GPU-accelerated image loading.

This module is intentionally importable on any platform. DALI-specific
objects are only instantiated inside `build_dali_wds_dataloader`, which
raises ImportError with a human-readable message when DALI is absent
(e.g. native Windows) rather than crashing at import time.
"""
from __future__ import annotations


def _check_dali() -> None:
    """Raise ImportError with helpful guidance if DALI is not available."""
    try:
        import nvidia.dali  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "NVIDIA DALI is not installed. "
            "Install it with:  pip install nvidia-dali-cuda120  "
            "(requires Linux + CUDA; not supported on native Windows). "
            "Use the CPU WDS loader (omit --use-dali) on Windows."
        ) from exc


class DaliDictWrapper:
    """Wraps DALI's DALIGenericIterator to yield dicts matching the CPU WDS loader."""

    def __init__(self, dali_iterator, epoch_batches: int | None) -> None:
        self.dali_iterator = dali_iterator
        self.epoch_batches = epoch_batches

    def __iter__(self):
        for batch in self.dali_iterator:
            yield {"image": batch[0]["image"]}

    def __len__(self) -> int:
        return self.epoch_batches if self.epoch_batches else 0


def build_dali_wds_dataloader(
    tar_urls: list[str],
    batch_size: int = 64,
    num_workers: int = 4,
    epoch_batches: int | None = None,
    crop_size: int = 512,
) -> DaliDictWrapper:
    """
    Build a DALI-backed dataloader that streams WebDataset tarballs
    and decodes/resizes images entirely on the GPU via nvJPEG.

    Args:
        tar_urls:       List of absolute paths or URLs to .tar shards.
        batch_size:     Number of samples per batch.
        num_workers:    DALI internal pipeline threads.
        epoch_batches:  Maximum batches per epoch (None = unlimited).
        crop_size:      Height and width to resize images to.

    Returns:
        A DaliDictWrapper that yields ``{"image": gpu_tensor}`` dicts.

    Raises:
        ImportError: When NVIDIA DALI is not installed.
    """
    _check_dali()

    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy

    @pipeline_def
    def _wds_pipeline(tar_paths, crop):
        """
        DALI pipeline: read → GPU decode → resize → CHW float32 [0, 1].
        """
        jpegs, _ = fn.readers.webdataset(
            paths=tar_paths,
            ext=["jpg", "png", "jpeg", "tif", "tiff"],
            missing_component_behavior="skip",
            random_shuffle=True,
            initial_fill=1000,
        )
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.resize(
            images,
            resize_x=crop,
            resize_y=crop,
            interp_type=types.INTERP_LINEAR,
        )
        # HWC uint8 → CHW float32 [0.0, 1.0]
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )
        return images

    pipe = _wds_pipeline(
        tar_paths=tar_urls,
        crop=crop_size,
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0,
    )

    iterator_size = epoch_batches * batch_size if epoch_batches else -1

    iterator = DALIGenericIterator(
        [pipe],
        output_map=["image"],
        size=iterator_size,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
    )

    return DaliDictWrapper(iterator, epoch_batches)
