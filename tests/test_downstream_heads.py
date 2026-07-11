import pytest
import torch
import torch.nn as nn
from ag_foundation.models.official_vit import RemoteSensingViT
from ag_foundation.models.cls import (
    RemoteSensingClassifier,
    RemoteSensingCounting,
    RemoteSensingOBBDetection,
    RemoteSensingTemporal,
)

@pytest.fixture
def dummy_backbone():
    # Use a small backbone for testing
    return RemoteSensingViT(
        image_size=224,
        model_name="S",
        pretrained_source="imagenet",
        pretrained_backbone=False,
    )

def test_counting_head(dummy_backbone):
    model = RemoteSensingCounting(
        backbone=dummy_backbone,
        density_upsample_factor=4,
        density_channels=128,
    )
    # 224 / 16 = 14 patches. Upsample factor 4 -> 56x56 density map.
    inputs = torch.randn(2, 3, 224, 224)
    density_map, count = model(inputs)
    
    assert density_map.shape == (2, 1, 56, 56)
    assert count.shape == (2, 1)

def test_obb_detection_head(dummy_backbone):
    model = RemoteSensingOBBDetection(
        backbone=dummy_backbone,
        num_classes=10,
        obb_num_convs=2,
        fpn_out_channels=128,
    )
    # Output should include classification, bounding box center, w, h, and angle.
    # Since it's FCOS style, it might output a dense prediction map.
    # We expect cls_scores, bbox_preds, angle_preds
    inputs = torch.randn(2, 3, 224, 224)
    cls_scores, bbox_preds, angle_preds = model(inputs)
    
    # 224 / 16 = 14 patches -> 14x14 grid
    assert cls_scores.shape == (2, 10, 14, 14)
    assert bbox_preds.shape == (2, 4, 14, 14)  # l, t, r, b
    assert angle_preds.shape == (2, 1, 14, 14)

def test_temporal_head(dummy_backbone):
    model = RemoteSensingTemporal(
        backbone=dummy_backbone,
        num_classes=5,
        temporal_hidden_dim=256,
        temporal_dropout=0.1,
    )
    # Temporal takes two inputs (T1, T2)
    t1_inputs = torch.randn(2, 3, 224, 224)
    t2_inputs = torch.randn(2, 3, 224, 224)
    
    logits = model(t1_inputs, t2_inputs)
    assert logits.shape == (2, 5)
