#!/usr/bin/env python3
"""
Pretraining Catalog Preparation Utility

This script reads a raw catalog (e.g. catalog_1.csv), rigorously filters out 
any evaluation datasets and ground truth/mask images, and creates a clean 
`catalog.csv` for high-performance dataloading during the pretraining loop.

It can also be imported directly into the dataloader to read the clean catalog.
"""

import csv
import sys
from pathlib import Path
from collections import Counter

def is_ground_truth(path_str: str) -> bool:
    """Check if the path likely belongs to a segmentation mask or ground truth label."""
    path_lower = path_str.lower()
    gt_patterns = [
        '/masks/', '\\masks\\', '/mask/', '\\mask\\',
        '_mask.', '_masks.', 'mask.',
        '/labels/', '\\labels\\', '/label/', '\\label\\',
        '_label.', 'label.',
        '_gt.', '_groundtruth.', '/gt/', '\\gt\\'
    ]
    for pattern in gt_patterns:
        if pattern in path_lower:
            return True
    return False

def is_evaluation(path_str: str) -> bool:
    """Check if the path belongs to the Evaluation directory."""
    # Datasets moved to Pretraining/Evaluation should be excluded from pretraining
    return "Evaluation" in path_str

def process_catalog(input_csv: Path, output_csv: Path) -> None:
    """Reads input catalog, filters it, writes to output, and prints analysis."""
    format_counts = Counter()
    total_images = 0
    excluded_gt = 0
    excluded_eval = 0
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    
    print(f"Reading raw catalog from: {input_csv}")
    
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', encoding='utf-8', newline='') as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        # Keep header if present
        header = next(reader, None)
        if header:
            writer.writerow(header)
        
        for row in reader:
            if not row: continue
            path = row[0]
            
            if is_evaluation(path):
                excluded_eval += 1
                continue
                
            if is_ground_truth(path):
                excluded_gt += 1
                continue
                
            ext = Path(path).suffix.lower()
            if ext in valid_extensions:
                writer.writerow(row)
                format_counts[ext] += 1
                total_images += 1

    print(f"\n--- Clean Pretraining Catalog Built at {output_csv} ---")
    print(f"Total Valid Pretraining Images: {total_images:,}")
    print(f"Ground Truth/Mask Images Excluded: {excluded_gt:,}")
    print(f"Evaluation Images Excluded: {excluded_eval:,}")
    print("\nFormats Distribution in New Catalog:")
    for ext, count in format_counts.most_common():
        print(f"  {ext.ljust(6)} : {count:,}")

def load_clean_catalog(catalog_csv: str) -> list:
    """
    Utility function for the PyTorch DataLoader to efficiently load the pre-filtered catalog.
    Returns a list of dicts: [{'path': str, 'group': str}]
    """
    records = []
    with open(catalog_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

if __name__ == '__main__':
    # Default paths
    project_root = Path(r"e:\AG_Dataset\AG-Foundational-Model\Pretraining")
    input_catalog = project_root / "catalog_1.csv"
    output_catalog = project_root / "catalog.csv"
    
    if not input_catalog.exists():
        print(f"Error: Could not find input catalog {input_catalog}")
        sys.exit(1)
        
    process_catalog(input_catalog, output_catalog)
