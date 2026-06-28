import os
import sys
import pytest
import torch

# Skip DALI tests on Windows since DALI only supports Linux (Colab)
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="NVIDIA DALI is not supported on native Windows."
)

def test_dali_wds_loader():
    try:
        from ag_foundation.data.dali_wds_loader import build_dali_wds_dataloader
    except ImportError:
        pytest.fail("Could not import build_dali_wds_dataloader. Ensure the module exists.")

    # We assume there is at least one tarball available in the shards directory
    # If not, this test will just skip or fail gracefully depending on the environment.
    shard_pattern = "E:/AG_Dataset/shards/dataset-000000.tar"
    if not os.path.exists(shard_pattern):
        pytest.skip(f"Could not find a sample shard at {shard_pattern} for testing.")

    batch_size = 4
    crop_size = 512
    
    loader = build_dali_wds_dataloader(
        tar_urls=[shard_pattern],
        batch_size=batch_size,
        num_workers=2,
        epoch_batches=10,
        crop_size=crop_size
    )
    
    assert loader is not None
    
    # Fetch one batch
    iterator = iter(loader)
    batch = next(iterator)
    
    # It must be a dictionary with an "image" key to match the PyTorch pipeline
    assert isinstance(batch, dict)
    assert "image" in batch
    
    images = batch["image"]
    
    # DALI should yield tensors on the GPU
    assert images.is_cuda
    
    # Shape should be (Batch, Channels, Height, Width)
    assert images.shape == (batch_size, 3, crop_size, crop_size)
    
    # The pixel values should be normalized or in the correct range (usually 0-1 if float)
    assert images.dtype == torch.float32
