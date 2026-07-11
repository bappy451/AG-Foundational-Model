import torch
import sys
sys.path.append('src')
from ag_foundation.training.spark_runner import SparkYoloRunner

cfg = {
    'runtime': {'device': 'cuda'},
    'model': {'mask_ratio': 0.6, 'patch_size': 32, 'pretrained': False},
    'optimizer': {'learning_rate': 0.001, 'weight_decay': 0.05},
    'data': {
        'data_root': 'E:/AG_Dataset/shards/*.tar',
        'val_fraction': 0.02,
        'batch_size': 2,
        'num_workers': 0,
        'epoch_batches': 10,
        'crop_size': 640
    }
}
runner = SparkYoloRunner(cfg)
train_loader, val_loader = runner.get_dataloaders()
print("Checking dataloader for NaNs...")
for step, batch in enumerate(train_loader):
    images = batch['image']
    if torch.isnan(images).any():
        print("Images contain NaNs at step", step)
        break
    else:
        print("Step", step, "OK")
    if step > 5:
        break
