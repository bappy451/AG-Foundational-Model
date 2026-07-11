import torch
import sys
import yaml
from pathlib import Path
from ag_foundation.data.wds_loader import build_wds_dataloader
import glob

def test_norm():
    with open('configs/wds_spark_yolo_pretrain.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    tar_urls = sorted(glob.glob(cfg['data']['data_root']))
    val_fraction = cfg['data'].get('val_fraction', 0.02)
    val_count = max(1, int(len(tar_urls) * val_fraction))
    train_urls = tar_urls[:-val_count]
    train_urls = [f"winfile://{p}" for p in train_urls]
    
    loader = build_wds_dataloader(
        train_urls,
        batch_size=cfg['data']['batch_size'],
        num_workers=0, # fast test
        epoch_batches=100,
        crop_size=cfg['data']['crop_size']
    )
    
    for step, batch in enumerate(loader):
        images = batch['image'].cuda()
        p = cfg['model']['patch_size']
        N, C, H, W = images.shape
        h, w = H // p, W // p
        patches = images.reshape(N, C, h, p, w, p)
        mean = patches.mean(dim=(1, 3, 5), keepdim=True)
        var = patches.var(dim=(1, 3, 5), unbiased=False, keepdim=True)
        norm_images = (patches - mean) / torch.sqrt(var + 1e-6)
        
        if torch.isnan(norm_images).any():
            print(f"NaN found in norm_images at step {step}!")
            print("var min:", var.min().item())
            print("var max:", var.max().item())
            sys.exit(1)
            
    print("Test passed! No NaNs in norm_images over 100 batches.")

if __name__ == '__main__':
    test_norm()
