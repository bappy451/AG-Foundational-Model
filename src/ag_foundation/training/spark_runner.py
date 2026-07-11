"""
spark_runner.py
===============
Pretraining runner for the SparK YOLO foundation model.
Streams from the 1024px bounded WebDataset shards and dynamically crops to 512x512.
Calculates the masked L1/L2 reconstruction loss.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms.functional as F_tv

from ag_foundation.data.wds_loader import build_wds_dataloader
from ag_foundation.models.spark_yolo import SparKYoloModel

class SparkYoloRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['runtime']['device'])
        
        # YOLO11-L config: width_multiple=1.0, depth_multiple=1.0 
        self.model = SparKYoloModel(
            width_multiple=1.0, 
            depth_multiple=1.0,
            mask_ratio=cfg['model']['mask_ratio'],
            patch_size=cfg['model']['patch_size']
        ).to(self.device)

        if cfg['model'].get('pretrained', False):
            # Attempt to load official weights if provided
            pt_path = cfg['model'].get('pretrained_weights_path', 'yolo11l.pt')
            self.model.load_ultralytics_weights(pt_path)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=cfg['optimizer']['learning_rate'], 
            weight_decay=cfg['optimizer']['weight_decay']
        )
        
        self.warmup_epochs = 5
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=cfg['runtime']['epochs'] - self.warmup_epochs,
            eta_min=1e-6
        )
        
        self.loss_fn = nn.L1Loss(reduction='none')
        
    def get_dataloaders(self):
        import glob
        # Find all shards
        tar_urls = sorted(glob.glob(self.cfg['data']['data_root']))
        if not tar_urls:
            raise FileNotFoundError(f"No shards found at {self.cfg['data']['data_root']}")
            
        # Standard 80/20 or custom val_fraction split
        val_fraction = self.cfg['data'].get('val_fraction', 0.02)
        val_count = max(1, int(len(tar_urls) * val_fraction))
        train_urls = tar_urls[:-val_count]
        val_urls = tar_urls[-val_count:]
        
        # Add winfile:// prefix for Windows WebDataset
        train_urls = [f"winfile://{p}" for p in train_urls]
        val_urls = [f"winfile://{p}" for p in val_urls]

        train_loader = build_wds_dataloader(
            train_urls,
            batch_size=self.cfg['data']['batch_size'],
            num_workers=self.cfg['data']['num_workers'],
            epoch_batches=self.cfg['data']['epoch_batches'],
            crop_size=self.cfg['data']['crop_size']
        )
        
        val_loader = build_wds_dataloader(
            val_urls,
            batch_size=self.cfg['data']['batch_size'],
            num_workers=self.cfg['data']['num_workers'],
            epoch_batches=self.cfg['data'].get('val_epoch_batches', 100),
            crop_size=self.cfg['data']['crop_size']
        )
        return train_loader, val_loader

    def train_epoch(self, dataloader, epoch, progress_cb=None):
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        import itertools
        
        # 5-epoch linear warmup for AdamW stability
        is_warmup = (epoch <= self.warmup_epochs)
        
        for step, batch in enumerate(itertools.islice(dataloader, num_batches), start=1):
            if is_warmup:
                global_step = (epoch - 1) * num_batches + step
                total_warmup_steps = self.warmup_epochs * num_batches
                warmup_factor = max(0.01, global_step / total_warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.cfg['optimizer']['learning_rate'] * warmup_factor
                    
            images = batch['image'].to(self.device)
            
            # Compute target normalization outside autocast (in float32) for numerical stability
            # p = self.cfg['model']['patch_size']
            # N, C, H, W = images.shape
            # h, w = H // p, W // p
            # patches = images.reshape(N, C, h, p, w, p)
            # mean = patches.mean(dim=(1, 3, 5), keepdim=True)
            # var = patches.var(dim=(1, 3, 5), unbiased=False, keepdim=True)
            # norm_images = (patches - mean) / torch.sqrt(var + 1e-6)
            # norm_images = norm_images.reshape(N, C, H, W)
            
            # Forward pass (tf32 is enabled by default on RTX 4090, avoiding spconv bfloat16 bugs)
            reconstructed, mask = self.model(images)
                
            # 1. Patch-wise Normalization (Per Channel)
            B, C, H, W = images.shape
            patch_size = 32
            
            # Reshape target to patches: (B, C, H/32, 32, W/32, 32)
            target_patches = images.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
            # Per-patch, per-channel mean and variance
            mean = target_patches.mean(dim=(3, 5), keepdim=True)
            var = target_patches.var(dim=(3, 5), unbiased=False, keepdim=True)
            normalized_targets = (target_patches - mean) / (var + 1e-6).sqrt()
            
            # Reshape reconstructed to patches
            reconstructed_patches = reconstructed.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
            
            # 2. L1 Loss (Mean Absolute Error) & 3. Mask-Exclusive
            # Calculate L1 loss per patch
            loss_patches = (reconstructed_patches.float() - normalized_targets.float()).abs().mean(dim=(1, 3, 5)) # Shape: (B, H/32, W/32)
            
            # Get mask grid
            # mask is (B, 1, H, W). We want (B, H/32, W/32). Since it's nearest-neighbor upsampled, we just slice.
            mask_patches = mask.squeeze(1)[:, ::patch_size, ::patch_size].float() # Shape: (B, H/32, W/32)
            
            # Mask-exclusive loss
            loss = (loss_patches * mask_patches).sum() / (mask_patches.sum() + 1e-6)
            
            # Scale loss for backward pass to prevent gradient explosion (Inf) in 100+ layer residual network.
            # Adam optimizer updates are invariant to constant gradient scaling, so this does not affect the learning rate.
            backward_loss = loss * 1e-4
            
            if not torch.isfinite(loss):
                print(f"WARNING: NaN loss detected at step {step}. Skipping update.")
                self.optimizer.zero_grad()
                continue
            
            self.optimizer.zero_grad()
            # Backward pass
            backward_loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if progress_cb:
                progress_cb(
                    completed=step,
                    total=num_batches,
                    description=f"Train Epoch [{epoch}/{self.cfg['runtime']['epochs']}]",
                    detail=f"loss: {loss.item():.4f} | avg: {(total_loss / step):.4f}"
                )
            elif step % self.cfg['runtime']['log_every'] == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate_epoch(self, dataloader, progress_cb=None):
        self.model.eval()
        total_loss = 0
        batches = 0
        num_batches = len(dataloader)
        
        import itertools
        for step, batch in enumerate(itertools.islice(dataloader, num_batches), start=1):
                
            images = batch['image'].to(self.device)
            
            reconstructed, mask = self.model(images)
                
            B, C, H, W = images.shape
            patch_size = 32
            
            target_patches = images.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
            mean = target_patches.mean(dim=(3, 5), keepdim=True)
            var = target_patches.var(dim=(3, 5), unbiased=False, keepdim=True)
            normalized_targets = (target_patches - mean) / (var + 1e-6).sqrt()
            
            reconstructed_patches = reconstructed.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
            loss_patches = (reconstructed_patches.float() - normalized_targets.float()).abs().mean(dim=(1, 3, 5))
            
            mask_patches = mask.squeeze(1)[:, ::patch_size, ::patch_size].float()
            
            # Compute true loss for logging
            loss = (loss_patches * mask_patches).sum() / (mask_patches.sum() + 1e-6)
            
            total_loss += loss.item()
            batches += 1
            
            if progress_cb:
                progress_cb(
                    completed=step,
                    total=num_batches,
                    description=f"Validate",
                    detail=f"loss: {loss.item():.4f} | avg: {(total_loss / batches):.4f}"
                )
            
        return total_loss / max(batches, 1)

    def run(self):
        train_loader, val_loader = self.get_dataloaders()
        epochs = self.cfg['runtime']['epochs']
        out_dir = Path(self.cfg['runtime']['output_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print("Starting SparK YOLO Pretraining...")
        from ag_foundation.progress import command_progress_context
        import sys
        
        with command_progress_context(sys.argv) as progress:
            progress_cb = progress.update if progress else None
            
            for epoch in range(1, epochs + 1):
                avg_train_loss = self.train_epoch(train_loader, epoch, progress_cb)
                avg_val_loss = self.validate_epoch(val_loader, progress_cb)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Step the scheduler (only after warmup)
                if epoch > self.warmup_epochs:
                    self.scheduler.step()
                
                print(f"Epoch {epoch} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
            
                # Save latest
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, out_dir / "spark_yolo_latest.pt")
                
                # Save best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                    }, out_dir / "spark_yolo_best.pt")
                    print(f"  -> New best model saved! (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    # Smoke test entrypoint
    import yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    runner = SparkYoloRunner(cfg)
    runner.run()
