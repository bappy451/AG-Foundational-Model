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

from ag_foundation.models.cls import RemoteSensingOBBDetection
from ag_foundation.models.official_vit import RemoteSensingViT
from ag_foundation.training.artifacts import save_training_checkpoint

logger = logging.getLogger("ag_foundation.train_det")


def parse_train_det_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune AG-Foundation on Detection Tasks")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    return parser.parse_args(argv)

def _load_config(config_path: str) -> dict[str, Any]:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class PlantSegOBBDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", crop_size: int = 224, num_classes: int = 10):
        self.data_root = Path(data_root)
        self.split_dir = self.data_root / split
        self.crop_size = crop_size
        self.grid_size = crop_size // 16
        self.num_classes = num_classes
        
        self.samples = []
        
        img_dir = self.split_dir / "images"
        lbl_dir = self.split_dir / "labels"
        
        if img_dir.exists() and lbl_dir.exists():
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    lbl_file = lbl_dir / (img_file.stem + ".txt")
                    if lbl_file.exists():
                        self.samples.append({
                            "image": img_file,
                            "label": lbl_file
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
            return torch.randn(3, self.crop_size, self.crop_size), (
                torch.zeros(self.num_classes, self.grid_size, self.grid_size),
                torch.zeros(4, self.grid_size, self.grid_size),
                torch.zeros(1, self.grid_size, self.grid_size)
            )
            
        sample = self.samples[idx]
        try:
            image = Image.open(sample["image"]).convert("RGB")
        except Exception:
             return torch.randn(3, self.crop_size, self.crop_size), (
                torch.zeros(self.num_classes, self.grid_size, self.grid_size),
                torch.zeros(4, self.grid_size, self.grid_size),
                torch.zeros(1, self.grid_size, self.grid_size)
            )
             
        image = self.transform(image)
        
        # Dense targets
        cls_target = torch.zeros(self.num_classes, self.grid_size, self.grid_size)
        bbox_target = torch.zeros(4, self.grid_size, self.grid_size)
        angle_target = torch.zeros(1, self.grid_size, self.grid_size)
        
        # Parse YOLO OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)
        with open(sample["label"], "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    c_id = int(parts[0])
                    coords = [float(p) for p in parts[1:9]]
                    xs = coords[0::2]
                    ys = coords[1::2]
                    
                    # Center
                    cx = sum(xs) / 4.0
                    cy = sum(ys) / 4.0
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    
                    # Map to grid
                    grid_x = int(cx * self.grid_size)
                    grid_y = int(cy * self.grid_size)
                    
                    if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size and c_id < self.num_classes:
                        cls_target[c_id, grid_y, grid_x] = 1.0
                        
                        # Simplified FCOS regression targets: distances to l, t, r, b
                        # Here just store cx, cy, w, h offsets scaled to grid size
                        bbox_target[0, grid_y, grid_x] = cx * self.grid_size - grid_x
                        bbox_target[1, grid_y, grid_x] = cy * self.grid_size - grid_y
                        bbox_target[2, grid_y, grid_x] = w * self.grid_size
                        bbox_target[3, grid_y, grid_x] = h * self.grid_size
                        
                        # Angle (simplified to 0 for this basic loader)
                        angle_target[0, grid_y, grid_x] = 0.0
                        
        return image, (cls_target, bbox_target, angle_target)


def _collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    cls_targets = torch.stack([b[1][0] for b in batch])
    bbox_targets = torch.stack([b[1][1] for b in batch])
    angle_targets = torch.stack([b[1][2] for b in batch])
    return {"image": images, "cls_target": cls_targets, "bbox_target": bbox_targets, "angle_target": angle_targets}

def run_train_det(args: argparse.Namespace) -> str:
    config = _load_config(args.config)
    runtime_cfg = config.get("runtime", {})
    model_cfg = config.get("model", {})
    optimizer_cfg = config.get("optimizer", {})
    data_cfg = config.get("data", {})

    data_root = data_cfg["data_root"]
    crop_size = data_cfg.get("crop_size", 224)
    batch_size = data_cfg.get("batch_size", 8)
    epochs = runtime_cfg.get("epochs", 50)
    device = runtime_cfg.get("device", "cuda")
    output_dir = Path(runtime_cfg.get("output_dir", "../runs/det_finetuning"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = model_cfg.get("num_classes", 10)
    train_dataset = PlantSegOBBDataset(data_root, split="train", crop_size=crop_size, num_classes=num_classes)
    val_dataset = PlantSegOBBDataset(data_root, split="valid", crop_size=crop_size, num_classes=num_classes)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=4)
    
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
        
    model = RemoteSensingOBBDetection(
        backbone=backbone,
        num_classes=num_classes,
        obb_num_convs=model_cfg.get("obb_num_convs", 4),
        fpn_out_channels=model_cfg.get("fpn_out_channels", 256),
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_cfg.get("learning_rate", 2e-4), weight_decay=optimizer_cfg.get("weight_decay", 0.05))
    
    # Loss functions
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    l1_loss_fn = torch.nn.SmoothL1Loss()
    
    best_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            cls_target = batch["cls_target"].to(device)
            bbox_target = batch["bbox_target"].to(device)
            angle_target = batch["angle_target"].to(device)
            
            optimizer.zero_grad()
            cls_scores, bbox_preds, angle_preds = model(images)
            
            # Mask out regression loss where there is no object
            obj_mask = (cls_target.sum(dim=1, keepdim=True) > 0).float()
            
            loss_cls = bce_loss_fn(cls_scores, cls_target)
            loss_box = l1_loss_fn(bbox_preds * obj_mask, bbox_target * obj_mask)
            loss_angle = l1_loss_fn(angle_preds * obj_mask, angle_target * obj_mask)
            
            loss = loss_cls + 2.0 * loss_box + 0.5 * loss_angle
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                cls_target = batch["cls_target"].to(device)
                bbox_target = batch["bbox_target"].to(device)
                angle_target = batch["angle_target"].to(device)
                
                cls_scores, bbox_preds, angle_preds = model(images)
                
                obj_mask = (cls_target.sum(dim=1, keepdim=True) > 0).float()
                loss_cls = bce_loss_fn(cls_scores, cls_target)
                loss_box = l1_loss_fn(bbox_preds * obj_mask, bbox_target * obj_mask)
                loss_angle = l1_loss_fn(angle_preds * obj_mask, angle_target * obj_mask)
                
                loss = loss_cls + 2.0 * loss_box + 0.5 * loss_angle
                val_loss += loss.item()
                
        val_loss /= max(1, len(val_loader))
        
        if val_loss < best_loss:
            best_loss = val_loss
            
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "best_loss": best_loss, "config": config}
        save_training_checkpoint(checkpoint, output_dir, improved=(val_loss == best_loss))
            
    print(f"Finished Detection fine-tuning. Best Val Loss: {best_loss:.4f}")
    return f"Finished fine-tuning. Best Val Loss: {best_loss:.4f}"

def main(argv: list[str] | None = None) -> None:
    args = parse_train_det_args(argv)
    run_train_det(args)
