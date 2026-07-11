import torch
import glob
results = []
for f in glob.glob('E:/AG_Dataset/runs/grid_search/trial_*/spark_yolo_best.pt'):
    try:
        ckpt = torch.load(f, map_location='cpu')
        val_loss = ckpt.get('val_loss', float('inf'))
        results.append((f, val_loss))
    except:
        pass
if results:
    best = min(results, key=lambda x: x[1])
    print('Best:', best)
    for r in sorted(results, key=lambda x: x[1]):
        print(f'{r[1]:.4f}  {r[0]}')
else:
    print('No checkpoints found.')
