import zipfile
import pytest
import os
import tarfile
from ag_foundation.data.build_wds_shards import shard_dir_to_wds

@pytest.fixture
def mock_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "pretraining"
    dataset_dir.mkdir()
    
    # 1. Create a zip file with mixed content
    zip_path = dataset_dir / "archive1.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("valid_image1.jpg", b"fake_jpg_bytes")
        zf.writestr("invalid_labels.json", b"{}")
        zf.writestr("invalid_image3_mask.png", b"fake_mask_bytes")
        
    # 2. Create another zip file
    zip_path2 = dataset_dir / "archive2.zip"
    with zipfile.ZipFile(zip_path2, 'w') as zf:
        zf.writestr("nested/path/valid_image2.png", b"fake_png_bytes")
        
    # 3. Create normal files in the directory
    (dataset_dir / "valid_image3.tiff").write_bytes(b"fake_tiff_bytes")
    (dataset_dir / "invalid_gt.jpg").write_bytes(b"fake_gt_bytes")
    
    # 4. Create an Evaluation folder that should be ignored
    eval_dir = dataset_dir / "Evaluation"
    eval_dir.mkdir()
    (eval_dir / "image4.jpg").write_bytes(b"fake_eval_bytes")
        
    return str(dataset_dir)

def test_shard_dir_to_wds(mock_dataset_dir):
    local_tmp = "tmp/test_dir_wds"
    output_prefix = os.path.join(local_tmp, "shards", "dataset")
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    shard_dir_to_wds(
        input_dir=mock_dataset_dir,
        output_prefix=output_prefix,
        max_count=2, # Small count to force multiple shards
        max_size=1024 * 1024 # 1 MB
    )
    
    shards_dir = os.path.join(local_tmp, "shards")
    shard_files = sorted([f for f in os.listdir(shards_dir) if f.endswith(".tar")])
    
    # We have exactly 3 valid images:
    # - archive1.zip/valid_image1.jpg
    # - archive2.zip/nested/path/valid_image2.png
    # - valid_image3.tiff
    # With max_count=2, this should create exactly 2 shards.
    assert len(shard_files) == 2, f"Expected 2 shards, found {len(shard_files)}"
    
    all_members = []
    for sf in shard_files:
        with tarfile.open(os.path.join(shards_dir, sf), 'r') as tar:
            all_members.extend(tar.getnames())
            
    # Check that all valid extensions are present
    assert any(m.endswith(".jpg") for m in all_members)
    assert any(m.endswith(".png") for m in all_members)
    assert any(m.endswith(".tiff") for m in all_members)
    
    # Check that invalid ones are omitted
    assert not any("mask" in m for m in all_members)
    assert not any(m.endswith(".json") for m in all_members)
    assert not any("gt" in m.lower() for m in all_members)
    assert not any("evaluation" in m.lower() for m in all_members)
