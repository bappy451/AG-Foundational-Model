"""Tests for the CPU-based WebDataset loader."""
from __future__ import annotations

import sys
import tarfile
from io import BytesIO

import pytest
import torch
from PIL import Image


def _to_wds_url(path: str) -> str:
    """Convert an absolute filesystem path to a WebDataset-compatible URL.

    On Windows, WebDataset requires a ``winfile://`` prefix for local paths.
    On Linux/macOS, the path can be used as-is.
    """
    if sys.platform == "win32":
        # Normalise backslashes and apply the winfile:// scheme
        return "winfile://" + path.replace("\\", "/")
    return path


@pytest.fixture
def mock_wds_shard(tmp_path):
    """Create a minimal TAR shard with 4 JPEG images for testing."""
    shard_path = tmp_path / "dataset-000000.tar"

    img = Image.new("RGB", (256, 256), color="red")
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format="JPEG")
    img_bytes = img_bytes_io.getvalue()

    with tarfile.open(shard_path, "w") as tar:
        for i in range(4):
            info = tarfile.TarInfo(name=f"{i:06d}.jpg")
            info.size = len(img_bytes)
            tar.addfile(info, BytesIO(img_bytes))

    # Return the URL-form that WebDataset can open on this platform
    return _to_wds_url(str(shard_path))


def test_wds_loader_outputs_dict_with_image_tensor(mock_wds_shard):
    """The loader must yield dicts with key 'image' containing a CPU float32 tensor."""
    from ag_foundation.data.wds_loader import build_wds_dataloader  # noqa: F401 – also registers winfile handler

    crop_size = 64  # Use small size for fast tests

    loader = build_wds_dataloader(
        tar_urls=[mock_wds_shard],
        batch_size=2,
        num_workers=0,  # 0 workers avoids multiprocessing issues in pytest
        crop_size=crop_size,
    )

    batch = next(iter(loader))

    # Loader returns a dict, not a tuple
    assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"
    assert "image" in batch, f"Expected 'image' key in batch dict, got {list(batch.keys())}"

    images = batch["image"]

    assert isinstance(images, torch.Tensor)
    assert images.shape == (2, 3, crop_size, crop_size), f"Unexpected shape: {images.shape}"
    assert images.device.type == "cpu"
    assert images.dtype == torch.float32
    # Values should be in [0, 1] from transforms.ToTensor()
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0 + 1e-5


def test_wds_loader_default_crop_size(mock_wds_shard):
    """Default crop_size must be 224 (matches standard ViT patch grid)."""
    from ag_foundation.data.wds_loader import build_wds_dataloader

    loader = build_wds_dataloader(
        tar_urls=[mock_wds_shard],
        batch_size=2,
        num_workers=0,
    )

    batch = next(iter(loader))
    images = batch["image"]
    assert images.shape == (2, 3, 224, 224), f"Expected 224x224, got {images.shape}"
