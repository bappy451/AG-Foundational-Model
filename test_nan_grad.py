import torch
import yaml
import itertools
from ag_foundation.models.spark_yolo import SparKYoloModel
from ag_foundation.data.wds_loader import build_wds_dataloader
import glob
import torch.nn as nn

def test_step():
    with open('configs/wds_spark_yolo_pretrain.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    model = SparKYoloModel(
        width_multiple=1.0,
        depth_multiple=1.0,
        mask_ratio=cfg['model']['mask_ratio'],
        patch_size=cfg['model']['patch_size']
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.05)
    loss_fn = nn.MSELoss(reduction='none')
    
    tar_urls = sorted(glob.glob(cfg['data']['data_root']))
    train_urls = [f"winfile://{p}" for p in tar_urls]
    
    loader = build_wds_dataloader(train_urls, batch_size=24, num_workers=0, crop_size=640)
    
    for step, batch in enumerate(loader):
        images = batch['image'].cuda()
        
        # Target Normalization
        p = cfg['model']['patch_size']
        N, C, H, W = images.shape
        h, w = H // p, W // p
        patches = images.reshape(N, C, h, p, w, p)
        mean = patches.mean(dim=(1, 3, 5), keepdim=True)
        var = patches.var(dim=(1, 3, 5), unbiased=False, keepdim=True)
        norm_images = (patches - mean) / torch.sqrt(var + 1e-6)
        norm_images = norm_images.reshape(N, C, H, W)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reconstructed, mask = model(images)
            mask_expanded = mask.expand_as(images)
            loss_full = loss_fn(reconstructed, norm_images)
            loss = (loss_full * mask_expanded).sum() / (mask_expanded.sum() + 1e-6)
            
        print("Loss at step 1:", loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        
        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
        
        if nan_grads:
            print("NaN gradients found in:", nan_grads)
        else:
            print("No NaN gradients!")
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Forward pass step 2
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec2, mask2 = model(images)
            loss_full2 = loss_fn(rec2, norm_images)
            loss2 = (loss_full2 * mask2.expand_as(images)).sum() / (mask2.expand_as(images).sum() + 1e-6)
            
        print("Loss at step 2:", loss2.item())
        break

if __name__ == '__main__':
    test_step()
