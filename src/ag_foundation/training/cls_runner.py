from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from ag_foundation.models.cls import RemoteSensingClassifier
from ag_foundation.models.official_vit import RemoteSensingViT
from ag_foundation.training.cls_trainer import ClassificationTrainer
from ag_foundation.training.artifacts import save_training_checkpoint
from ag_foundation.training.visualization import save_training_curves

import shutil

logger = logging.getLogger("ag_foundation.train_cls")


def _save_config_snapshot(config: dict, output_dir: Path, config_path: str) -> None:
    """Save a snapshot of the config and source config path for reproducibility."""
    import yaml
    snapshot = {"source_config": os.path.abspath(config_path), "config": config}
    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False)
    # Also copy the raw config file
    shutil.copy2(config_path, output_dir / "config.yaml")


def parse_train_cls_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune AG-Foundation on Classification Tasks")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    return parser.parse_args(argv)


def _load_config(config_path: str) -> dict[str, Any]:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {"image": images, "label": labels}


def run_train_cls(
    args: argparse.Namespace,
    *,
    command_argv: list[str] | None = None,
    progress_callback: Any | None = None,
) -> str:
    config = _load_config(args.config)
    
    data_cfg = config.get("data", {})
    runtime_cfg = config.get("runtime", {})
    model_cfg = config.get("model", {})
    optimizer_cfg = config.get("optimizer", {})

    data_root = data_cfg["data_root"]
    crop_size = data_cfg.get("crop_size", 224)
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)
    
    epochs = runtime_cfg.get("epochs", 20)
    device = runtime_cfg.get("device", "cuda")
    precision = runtime_cfg.get("precision", "bf16")
    output_dir = Path(runtime_cfg.get("output_dir", "../runs/cls_finetuning"))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Transforms
    train_transform = T.Compose([
        T.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=5),
        T.ToTensor(),
    ])
    
    val_transform = T.Compose([
        T.Resize(crop_size + 32),
        T.CenterCrop(crop_size),
        T.ToTensor(),
    ])
    
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "valid")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_root, "test")
        
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)
    
    # Save class map
    class_to_idx = train_dataset.class_to_idx
    with open(output_dir / "class_map.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
        
    # Calculate class weights for CrossEntropyLoss
    from collections import Counter
    train_labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    # weights[i] = total / count of class i
    weights = [total_samples / class_counts.get(i, 1) for i in range(len(class_to_idx))]
    class_weight_tensor = torch.tensor(weights, dtype=torch.float32)
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    
    pretrained_weights = model_cfg.get("pretrained_weights")
    use_timm_pretrained = not pretrained_weights
    
    backbone = RemoteSensingViT(
        image_size=crop_size,
        model_name=model_cfg.get("model_name", "B"),
        pretrained_backbone=use_timm_pretrained,
        pretrained_source=model_cfg.get("pretrained_source", "imagenet"),
        drop_rate=model_cfg.get("drop_rate", 0.1),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.1),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
    )
    
    # Load pretrained weights from AG-Foundation
    pretrained_weights = model_cfg.get("pretrained_weights")
    if pretrained_weights:
        # Resolve path relative to the config file location if it's a relative path
        if not os.path.isabs(pretrained_weights):
            config_dir = os.path.dirname(os.path.abspath(args.config))
            pretrained_weights = os.path.normpath(os.path.join(config_dir, pretrained_weights))
        
        if not os.path.exists(pretrained_weights):
            raise FileNotFoundError(
                f"Pretrained weights not found at: {pretrained_weights}\n"
                f"Check that 'pretrained_weights' in the config is correct.\n"
                f"If you want to use ImageNet weights, set 'pretrained_weights' to an empty string."
            )
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Handle prefixes if loaded from MIM
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("student_backbone."):
                cleaned_state_dict[k.replace("student_backbone.", "")] = v
            elif k.startswith("backbone."):
                cleaned_state_dict[k.replace("backbone.", "")] = v
        loaded_keys = backbone.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Loaded pretrained backbone from {pretrained_weights}")
        print(f"  Missing keys: {len(loaded_keys.missing_keys)} | Unexpected keys: {len(loaded_keys.unexpected_keys)}")
        
    model = RemoteSensingClassifier(
        backbone=backbone,
        num_classes=len(class_to_idx),
        precision=precision,
        head_dropout=model_cfg.get("head_dropout", 0.2),
    )
    
    def get_layerwise_lr_groups(model, base_lr, weight_decay, layer_decay):
        total_blocks = len(model.backbone.backbone.blocks)
        param_groups = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Determine layer ID (0 to total_blocks + 1)
            if name.startswith("head") or name.startswith("fc"):
                layer_id = total_blocks + 1
            elif name.startswith("backbone.backbone.blocks."):
                block_idx = int(name.split(".")[3])
                layer_id = block_idx + 1
            else:
                layer_id = 0
                
            lr_scale = layer_decay ** (total_blocks + 1 - layer_id)
            lr = base_lr * lr_scale
            
            # Decouple weight decay
            if param.ndim <= 1 or name.endswith(".bias") or "cls_token" in name or "pos_embed" in name:
                wd = 0.0
            else:
                wd = weight_decay
                
            group_name = f"layer_{layer_id}_wd_{wd}"
            if group_name not in param_groups:
                param_groups[group_name] = {"params": [], "weight_decay": wd, "lr": lr}
            param_groups[group_name]["params"].append(param)
            
        return list(param_groups.values())

    base_lr = optimizer_cfg.get("learning_rate", 5e-5)
    weight_decay = optimizer_cfg.get("weight_decay", 0.05)
    layer_decay = model_cfg.get("layer_decay", 1.0)
    
    if layer_decay < 1.0:
        print(f"Applying Layer-wise Learning Rate Decay (factor={layer_decay}, base_lr={base_lr})")
        param_groups = get_layerwise_lr_groups(model, base_lr, weight_decay, layer_decay)
        optimizer = torch.optim.AdamW(param_groups)
        # Print LR distribution for verification
        unique_lrs = sorted(set(g["lr"] for g in param_groups))
        print(f"  LLRD effective LRs: min={unique_lrs[0]:.2e}  max={unique_lrs[-1]:.2e}  ({len(unique_lrs)} groups)")

    else:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=base_lr,
        )
    
    # Cosine Scheduler with linear warmup
    def lr_schedule(epoch: int, total: int) -> float:
        warmup = runtime_cfg.get("warmup_epochs", 5)
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = float(epoch - warmup) / float(max(1, total - warmup))
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epoch_lr_schedule=lr_schedule,
        device=device,
        precision=precision,
        progress_callback=progress_callback,
        label_smoothing=optimizer_cfg.get("label_smoothing", 0.1),
        class_weights=class_weight_tensor,
        grad_clip_norm=runtime_cfg.get("grad_clip_norm", None),
    )
    
    best_acc = 0.0
    history = []
    early_stop_patience = runtime_cfg.get("early_stopping_patience", None)
    epochs_without_improvement = 0
    
    # Save config snapshot for reproducibility
    _save_config_snapshot(config, output_dir, args.config)
    freeze_backbone_epochs = int(model_cfg.get("freeze_backbone_epochs", 5 if model_cfg.get("pretrained_source") == "dinov3" else 0))
    
    for epoch in range(epochs):
        # LP-FT Two-Stage Fine-Tuning
        if freeze_backbone_epochs > 0:
            if epoch < freeze_backbone_epochs:
                model.backbone.requires_grad_(False)
                if epoch == 0:
                    print(f"[LP-FT] Stage 1: Backbone frozen for epochs 1-{freeze_backbone_epochs} (Linear Probing head only)")
            elif epoch == freeze_backbone_epochs:
                model.backbone.requires_grad_(True)
                print(f"[LP-FT] Stage 2: Backbone unfrozen at epoch {epoch + 1} (End-to-end LLRD Fine-Tuning)")

        trainer._apply_epoch_learning_rate(epoch, epochs)
        train_metrics = trainer.train_epoch(epoch, epochs)
        val_metrics = trainer.evaluate(epoch)
        
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc1": train_metrics.get("acc1", 0.0),
            "train_acc5": train_metrics.get("acc5", 0.0),
            "val_loss": val_metrics["loss"],
            "val_acc1": val_metrics.get("acc1", 0.0),
            "val_acc5": val_metrics.get("acc5", 0.0),
            "learning_rate": trainer._current_learning_rate(),
        }
        history.append(record)
        
        # Save checkpoints
        improved = False
        if val_metrics["acc1"] > best_acc:
            best_acc = val_metrics["acc1"]
            improved = True
            
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "history": history,
            "config": config,
        }
        
        save_training_checkpoint(
            checkpoint,
            output_dir,
            improved=improved,
        )
        
        # Early stopping
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stop_patience and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {early_stop_patience} epochs).")
            break
            
    # Visualize training curves
    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(output_dir / "metrics.csv", index=False)
    save_training_curves(history, output_dir, method_name="Classification")
    
    return f"Finished fine-tuning. Best Val Acc: {best_acc:.1f}%"


def main(argv: list[str] | None = None) -> None:
    args = parse_train_cls_args(argv)
    from ag_foundation.progress import command_progress_context
    with command_progress_context(argv) as progress:
        progress_cb = progress.update if progress else None
        summary = run_train_cls(args, command_argv=list(argv or []), progress_callback=progress_cb)
        if summary:
            print(summary)
