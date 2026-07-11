import torch
from ag_foundation.models.spark_yolo import SparKYoloModel

def run():
    model = SparKYoloModel(width_multiple=1.0, depth_multiple=1.0, mask_ratio=0.6, patch_size=32).cuda()
    model.load_ultralytics_weights('yolo11l.pt')
    images = torch.rand(24, 3, 640, 640).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for step in range(1, 10):
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            reconstructed, mask = model(images)
        
        B, C, H, W = images.shape
        patch_size = 32
        target_patches = images.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        mean = target_patches.mean(dim=(3, 5), keepdim=True)
        var = target_patches.var(dim=(3, 5), unbiased=False, keepdim=True)
        normalized_targets = (target_patches - mean) / (var + 1e-6).sqrt()
        
        reconstructed_patches = reconstructed.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        loss_patches = (reconstructed_patches.float() - normalized_targets.float()).abs().mean(dim=(1, 3, 5))
        mask_patches = mask.squeeze(1)[:, ::patch_size, ::patch_size].float()
        
        loss = (loss_patches * mask_patches).sum() / (mask_patches.sum() + 1e-6)
        loss.backward()
        
        nan_layers = []
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                nan_layers.append(name)
        
        print(f'Step {step} - Loss: {loss.item()} - NaNs: {nan_layers}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

if __name__ == '__main__':
    run()
