#!/usr/bin/env python3
import yaml
import subprocess
import itertools
import csv
from pathlib import Path
import os

CONFIG_PATH = r"E:\AG_Dataset\AG-Foundational-Model\configs\wds_spark_yolo_pretrain.yaml"
OUTPUT_CSV = r"E:\AG_Dataset\AG-Foundational-Model\runs\grid_search_results.csv"

# The hyperparameter grid to search over (State of the Art for SparK)
# Mask Ratio: 0.6 is standard for CNNs, but testing 0.4 and 0.75 reveals dataset-specific robustness.
# Learning Rate: SparK can require lower LRs (1e-4) or handle standard (1e-3) depending on scaling.
GRID = {
    "learning_rate": [0.001, 0.0005, 0.0001],
    "weight_decay": [0.05, 0.01],
    "mask_ratio": [0.40, 0.60, 0.75]
}

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def run_experiment(lr, wd, mr, trial):
    # Load config and inject new params
    cfg = load_config()
    cfg['optimizer']['learning_rate'] = lr
    cfg['optimizer']['weight_decay'] = wd
    cfg['model']['mask_ratio'] = mr
    
    # Grid search is fast - only 2 epochs
    cfg['runtime']['epochs'] = 2
    # Adjust epoch batches down so 2 epochs goes faster (e.g. 500 steps instead of 16000)
    cfg['data']['epoch_batches'] = 100 
    cfg['data']['val_epoch_batches'] = 20
    
    # Set isolated output dir
    out_dir = f"../runs/grid_search/trial_{trial}"
    cfg['runtime']['output_dir'] = out_dir
    save_config(cfg)
    
    print(f"\n{'='*60}")
    print(f" TRIAL {trial} | LR: {lr} | WD: {wd} | Mask Ratio: {mr}")
    print(f"{'='*60}")
    
    import sys
    # Run spark_runner using the same python executable
    cmd = [
        sys.executable, r"E:\AG_Dataset\AG-Foundational-Model\src\ag_foundation\training\spark_runner.py",
        "--config", CONFIG_PATH
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse the output to find the final validation loss
    best_val_loss = None
    for line in result.stdout.splitlines():
        print("  " + line)
        if "New best model saved!" in line and "Val Loss:" in line:
            # Extract loss
            try:
                best_val_loss = float(line.split("Val Loss:")[1].strip().strip(')'))
            except Exception:
                pass
                
    if result.returncode != 0:
        print(" [ERROR] Run failed!")
        print(result.stderr)
        
    return best_val_loss

def main():
    keys = list(GRID.keys())
    combinations = list(itertools.product(*[GRID[k] for k in keys]))
    
    results = []
    
    # Create results file
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "learning_rate", "weight_decay", "mask_ratio", "val_loss"])
    
    for i, combo in enumerate(combinations, 1):
        lr, wd, mr = combo
        val_loss = run_experiment(lr, wd, mr, i)
        
        # Save result incrementally
        with open(OUTPUT_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, lr, wd, mr, val_loss])
            
        results.append((val_loss or float('inf'), combo))
        
    print(f"\n{'='*60}")
    print(" GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    
    best = min(results, key=lambda x: x[0])
    print(f"BEST CONFIGURATION:")
    print(f"  Val Loss:      {best[0]}")
    print(f"  Learning Rate: {best[1][0]}")
    print(f"  Weight Decay:  {best[1][1]}")
    print(f"  Mask Ratio:    {best[1][2]}")

if __name__ == "__main__":
    main()
