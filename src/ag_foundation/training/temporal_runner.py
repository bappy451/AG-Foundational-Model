from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

from ag_foundation.models.cls import RemoteSensingTemporal
from ag_foundation.models.official_vit import RemoteSensingViT
from ag_foundation.training.artifacts import save_training_checkpoint

logger = logging.getLogger("ag_foundation.train_temporal")


def parse_train_temporal_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune AG-Foundation on Temporal Tasks")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    return parser.parse_args(argv)

def _load_config(config_path: str) -> dict[str, Any]:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class LongitudinalNutrientDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", crop_size: int = 224, split_ratio: float = 0.8):
        self.data_root = Path(data_root) / "Longitudinal_Nutrient_Deficiency"
        self.crop_size = crop_size
        
        self.samples = []
        if self.data_root.exists():
            fields = sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("field_")])
            
            # Simple train/val split based on field names
            split_idx = int(len(fields) * split_ratio)
            if split == "train":
                fields = fields[:split_idx]
            else:
                fields = fields[split_idx:]
                
            for field_dir in fields:
                t1_path = field_dir / "image_i0.png"
                t2_path = field_dir / "image_i1.png"
                mask_path = field_dir / "nutrient_mask_g0.png"
                
                if t1_path.exists() and t2_path.exists() and mask_path.exists():
                    self.samples.append({
                        't1': t1_path,
                        't2': t2_path,
                        'mask': mask_path
                    })
                    
        self.transform = T.Compose([
            T.Resize((crop_size, crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return max(1, len(self.samples)) if self.samples else 100

    def __getitem__(self, idx):
        if not self.samples:
            return torch.randn(3, self.crop_size, self.crop_size), torch.randn(3, self.crop_size, self.crop_size), torch.tensor(0, dtype=torch.long)
            
        sample = self.samples[idx]
        try:
            img1 = Image.open(sample['t1']).convert('RGB')
            img2 = Image.open(sample['t2']).convert('RGB')
            mask = Image.open(sample['mask']).convert('L')
            
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
            # Compute severity class (0-4) based on deficiency pixels
            mask_arr = np.array(mask)
            deficiency_ratio = np.sum(mask_arr > 0) / mask_arr.size
            
            # Bin into 5 classes
            if deficiency_ratio < 0.05:
                label = 0
            elif deficiency_ratio < 0.15:
                label = 1
            elif deficiency_ratio < 0.30:
                label = 2
            elif deficiency_ratio < 0.50:
                label = 3
            else:
                label = 4
                
        except Exception:
            return torch.randn(3, self.crop_size, self.crop_size), torch.randn(3, self.crop_size, self.crop_size), torch.tensor(0, dtype=torch.long)
            
        return img1, img2, torch.tensor(label, dtype=torch.long)

def _collate_fn(batch):
    images_t1 = torch.stack([b[0] for b in batch])
    images_t2 = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return {"image_t1": images_t1, "image_t2": images_t2, "label": labels}

def run_train_temporal(args: argparse.Namespace) -> str:
    config = _load_config(args.config)
    runtime_cfg = config.get("runtime", {})
    model_cfg = config.get("model", {})
    optimizer_cfg = config.get("optimizer", {})
    data_cfg = config.get("data", {})

    data_root = data_cfg["data_root"]
    crop_size = data_cfg.get("crop_size", 224)
    batch_size = data_cfg.get("batch_size", 64)
    epochs = runtime_cfg.get("epochs", 100)
    device = runtime_cfg.get("device", "cuda")
    output_dir = Path(runtime_cfg.get("output_dir", "../runs/temporal_finetuning"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = LongitudinalNutrientDataset(data_root, split="train", crop_size=crop_size)
    val_dataset = LongitudinalNutrientDataset(data_root, split="val", crop_size=crop_size)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    
    pretrained_weights = model_cfg.get("pretrained_weights")
    use_timm_pretrained = not pretrained_weights
    
    backbone = RemoteSensingViT(
        image_size=crop_size,
        model_name=model_cfg.get("model_name", "B"),
        pretrained_backbone=use_timm_pretrained,
        pretrained_source=model_cfg.get("pretrained_source", "imagenet"),
    )
    
    if pretrained_weights:
        if not os.path.isabs(pretrained_weights):
            config_dir = os.path.dirname(os.path.abspath(args.config))
            pretrained_weights = os.path.normpath(os.path.join(config_dir, pretrained_weights))
        if os.path.exists(pretrained_weights):
            checkpoint = torch.load(pretrained_weights, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            cleaned_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
            backbone.load_state_dict(cleaned_state_dict, strict=False)
            print(f"Loaded pretrained backbone from {pretrained_weights}")
        
    model = RemoteSensingTemporal(
        backbone=backbone,
        num_classes=model_cfg.get("num_classes", 5) or 5,
        temporal_hidden_dim=model_cfg.get("temporal_hidden_dim", 512),
        temporal_dropout=model_cfg.get("temporal_dropout", 0.3),
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_cfg.get("learning_rate", 5e-4), weight_decay=optimizer_cfg.get("weight_decay", 0.05))
    
    best_acc = 0.0
    
    # Calculate class weights to handle severe imbalance
    print("Calculating class weights from training set...")
    class_counts = torch.zeros(model_cfg.get("num_classes", 5) or 5)
    for sample in train_dataset.samples:
        try:
            mask = Image.open(sample['mask']).convert('L')
            mask_arr = np.array(mask)
            deficiency_ratio = np.sum(mask_arr > 0) / mask_arr.size
            if deficiency_ratio < 0.05: lbl = 0
            elif deficiency_ratio < 0.15: lbl = 1
            elif deficiency_ratio < 0.30: lbl = 2
            elif deficiency_ratio < 0.50: lbl = 3
            else: lbl = 4
            class_counts[lbl] += 1
        except Exception:
            class_counts[0] += 1
            
    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            img1 = batch["image_t1"].to(device)
            img2 = batch["image_t2"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(img1, img2)
            
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                img1 = batch["image_t1"].to(device)
                img2 = batch["image_t2"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(img1, img2)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                val_loss += loss.item()
                
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_acc = correct / max(1, total)
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "best_acc": best_acc}
        save_training_checkpoint(checkpoint, output_dir, improved=(val_acc == best_acc))
            
    print(f"Finished Temporal fine-tuning. Best Val Acc: {best_acc:.4f}")
    return f"Finished fine-tuning. Best Val Acc: {best_acc:.4f}"

def main(argv: list[str] | None = None) -> None:
    args = parse_train_temporal_args(argv)
    run_train_temporal(args)
