import torch
import pytest

from ag_foundation.models.cls import RemoteSensingClassifier
from ag_foundation.models.official_vit import RemoteSensingViT
from ag_foundation.training.cls_trainer import ClassificationTrainer


@pytest.fixture
def dummy_cls_model():
    backbone = RemoteSensingViT(
        image_size=64,
        model_name="vit_tiny_patch16_224",
        pretrained_backbone=False,
    )
    return RemoteSensingClassifier(backbone, num_classes=5)


@pytest.fixture
def dummy_train_loader():
    class DummyLoader:
        def __init__(self):
            self.batch_size = 2
            
        def __iter__(self):
            for _ in range(3):
                # Simulated batch of images and labels
                yield {
                    "image": torch.randn(2, 3, 64, 64),
                    "label": torch.randint(0, 5, (2,)),
                }
                
        def __len__(self):
            return 3
            
    return DummyLoader()


def test_classification_trainer_step(dummy_cls_model, dummy_train_loader):
    optimizer = torch.optim.Adam(dummy_cls_model.parameters(), lr=1e-3)
    
    trainer = ClassificationTrainer(
        model=dummy_cls_model,
        train_loader=dummy_train_loader,
        optimizer=optimizer,
        device="cpu",
    )
    
    batch = next(iter(dummy_train_loader))
    
    # Run a single step
    metrics = trainer.train_step(batch)
    
    assert "loss" in metrics
    assert "acc1" in metrics
    assert "acc5" in metrics
    assert isinstance(metrics["loss"], float)
    assert 0.0 <= metrics["acc1"] <= 100.0


def test_classification_trainer_epoch(dummy_cls_model, dummy_train_loader):
    optimizer = torch.optim.Adam(dummy_cls_model.parameters(), lr=1e-3)
    
    trainer = ClassificationTrainer(
        model=dummy_cls_model,
        train_loader=dummy_train_loader,
        optimizer=optimizer,
        device="cpu",
    )
    
    epoch_metrics = trainer.train_epoch(0, 1)
    
    assert "loss" in epoch_metrics
    assert "acc1" in epoch_metrics
    assert epoch_metrics["batches"] == 3
