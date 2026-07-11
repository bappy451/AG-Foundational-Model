import torch
torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import yaml
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
    
    # Load pretrained weights
    pt_path = cfg['model'].get('pretrained_weights_path', 'yolo11l.pt')
    model.load_ultralytics_weights(pt_path)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=0.05)
    loss_fn = nn.L1Loss(reduction='none')
    
    tar_urls = sorted(glob.glob(cfg['data']['data_root']))
    train_urls = [f"winfile://{p}" for p in tar_urls]
    
    loader = build_wds_dataloader(
        train_urls,
        batch_size=cfg['data']['batch_size'],
        num_workers=0,
        crop_size=cfg['data']['crop_size']
    )
    
    for step, batch in enumerate(loader, start=1):
        images = batch['image'].cuda()
        
        reconstructed, mask = model(images)
            
        # 1. Patch-wise Normalization (Per Channel)
        B, C, H, W = images.shape
        patch_size = 32
        
        target_patches = images.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        mean = target_patches.mean(dim=(3, 5), keepdim=True)
        var = target_patches.var(dim=(3, 5), unbiased=False, keepdim=True)
        normalized_targets = (target_patches - mean) / (var + 1e-6).sqrt()
            
        reconstructed_patches = reconstructed.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        
        # 2. L1 Loss (Mean Absolute Error) & 3. Mask-Exclusive
        loss_patches = (reconstructed_patches.float() - normalized_targets.float()).abs().mean(dim=(1, 3, 5))
        
        # mask is (B, 1, H, W). We want (B, H/32, W/32). Since it's nearest-neighbor upsampled, we just slice.
        mask_patches = mask.squeeze(1)[:, ::patch_size, ::patch_size].float()
        
        loss = (loss_patches * mask_patches).sum() / (mask_patches.sum() + 1e-6)
        loss = loss * 1e-4
            
        print(f"Step {step} - Loss:", loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        
        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
        
        if nan_grads:
            print(f"NaN gradients found at step {step} in {len(nan_grads)} layers!")
            break
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        if step >= 10:
            print("Successfully completed 10 steps without NaNs!")
            break

if __name__ == '__main__':
    test_step()
