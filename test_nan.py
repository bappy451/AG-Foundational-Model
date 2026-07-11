import torch
import sys
sys.path.append('src')
from ag_foundation.models.spark_yolo import SparKYoloModel

model = SparKYoloModel(width_multiple=1.0, depth_multiple=1.0, mask_ratio=0.6, patch_size=32).cuda()
model.load_ultralytics_weights('yolo11l.pt')
model.eval()
with torch.no_grad():
    x = torch.rand(2, 3, 640, 640).cuda()
    out, mask = model(x)
    print('Output NaNs:', torch.isnan(out).any().item())
