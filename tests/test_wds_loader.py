import pytest
import torch
import os
import tarfile
from io import BytesIO
from PIL import Image

from ag_foundation.data.wds_loader import build_wds_dataloader

@pytest.fixture
def mock_wds_shard():
    local_tmp = "tmp/test_wds_loader"
    os.makedirs(local_tmp, exist_ok=True)
    shard_path = os.path.join(local_tmp, "dataset-000000.tar")
    
    # Create a dummy image
    img = Image.new('RGB', (256, 256), color = 'red')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    with tarfile.open(shard_path, 'w') as tar:
        for i in range(4):
            # Write jpg
            info = tarfile.TarInfo(name=f"{i:06d}.jpg")
            info.size = len(img_bytes)
            tar.addfile(info, BytesIO(img_bytes))
            
    return shard_path

def test_wds_loader_outputs_cpu_tensors(mock_wds_shard):
    loader = build_wds_dataloader(
        tar_urls=[mock_wds_shard],
        batch_size=2,
        num_workers=0 # Use 0 for testing to avoid multiprocessing issues in pytest
    )
    
    batch = next(iter(loader))
    
    # WebDataset returns a tuple for each component requested.
    # We will request only the image, so we expect a tuple of (tensor,) or just dict depending on decode
    # In our implementation, we'll return tuples.
    images = batch[0]
    
    assert isinstance(images, torch.Tensor)
    assert images.shape == (2, 3, 256, 256)
    assert images.device.type == 'cpu'
    assert images.dtype == torch.float32
