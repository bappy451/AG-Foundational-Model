import pytest
import torch
from ag_foundation.data.kornia_augment import KorniaMultiCropAugmentation

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_kornia_multi_crop_augmentation():
    B = 2
    H, W = 256, 256
    
    # Mock CPU batch from WebDataset
    cpu_batch = torch.rand((B, 3, H, W))
    
    # Send to GPU as done in the training loop
    gpu_batch = cpu_batch.to('cuda')
    
    augmenter = KorniaMultiCropAugmentation(
        global_crops_scale=(0.14, 1.0),
        local_crops_scale=(0.05, 0.14),
        global_crops_number=2,
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96
    ).cuda()
    
    output_crops = augmenter(gpu_batch)
    
    assert isinstance(output_crops, tuple)
    assert len(output_crops) == 10 # 2 global + 8 local
    
    for i, crop in enumerate(output_crops):
        assert crop.device.type == 'cuda'
        if i < 2:
            assert crop.shape == (B, 3, 224, 224), f"Global crop {i} shape mismatch"
        else:
            assert crop.shape == (B, 3, 96, 96), f"Local crop {i} shape mismatch"
