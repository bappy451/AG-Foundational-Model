import torch
from ag_foundation.training.spark_runner import SparkYoloRunner
import yaml
with open('configs/wds_spark_yolo_pretrain.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
runner = SparkYoloRunner(cfg)
train_loader, _ = runner.get_dataloaders()
import itertools
for batch in itertools.islice(train_loader, 1):
    images = batch['image']
    print(f'Images shape: {images.shape}')
    print(f'Images min: {images.min()}')
    print(f'Images max: {images.max()}')
    print(f'Images mean: {images.mean()}')
