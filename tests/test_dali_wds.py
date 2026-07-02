"""Tests for the NVIDIA DALI WebDataset loader.

These tests are automatically skipped on Windows because DALI is only
supported on Linux + CUDA.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# Skip the whole module on platforms where DALI is not supported
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="NVIDIA DALI is not supported on native Windows.",
)


def test_dali_wds_loader_import_works():
    """The module must be importable without crashing, even without DALI installed."""
    from ag_foundation.data import dali_wds_loader  # noqa: F401


def test_build_dali_wds_dataloader_raises_without_dali(monkeypatch):
    """build_dali_wds_dataloader should raise ImportError when DALI is absent."""
    import ag_foundation.data.dali_wds_loader as dali_mod

    # Simulate a missing DALI install
    original = dali_mod._check_dali

    def _raise():
        raise ImportError("mocked missing dali")

    monkeypatch.setattr(dali_mod, "_check_dali", _raise)

    with pytest.raises(ImportError):
        dali_mod.build_dali_wds_dataloader(tar_urls=["fake.tar"], batch_size=4)

    monkeypatch.setattr(dali_mod, "_check_dali", original)


def test_dali_wds_loader_with_real_shard():
    """End-to-end test using a real shard file (skipped if not available)."""
    shard_path = "E:/AG_Dataset/shards/dataset-000000.tar"
    if not os.path.exists(shard_path):
        pytest.skip(f"Could not find sample shard at {shard_path} for testing.")

    try:
        from ag_foundation.data.dali_wds_loader import build_dali_wds_dataloader
    except ImportError as exc:
        pytest.skip(f"DALI not installed: {exc}")

    batch_size = 4
    crop_size = 512

    loader = build_dali_wds_dataloader(
        tar_urls=[shard_path],
        batch_size=batch_size,
        num_workers=2,
        epoch_batches=10,
        crop_size=crop_size,
    )

    assert loader is not None

    batch = next(iter(loader))

    assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"
    assert "image" in batch

    images = batch["image"]

    # DALI should yield tensors on the GPU
    assert images.is_cuda
    assert images.shape == (batch_size, 3, crop_size, crop_size)
    assert images.dtype == torch.float32
