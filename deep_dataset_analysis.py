"""
Comprehensive pretraining dataset analysis.
Inspects each ZIP file in Pretraining/ by sampling images,
and does a deep dive into the shards directory.
Outputs a full table of: dataset name, file size, estimated/actual image count,
and sampled image shapes.
"""
import os
import sys
import io
import glob
import zipfile
import tarfile
import random
import json
from pathlib import Path
from io import BytesIO
from collections import defaultdict, Counter
from PIL import Image

# Route ALL output to a UTF-8 file to avoid Windows cp1252 crash from emoji filenames
LOG_PATH = r"E:\AG_Dataset\AG-Foundational-Model\pretraining_analysis_log.txt"
_log_file = open(LOG_PATH, 'w', encoding='utf-8')

def safe(s):
    """Encode any string to ASCII-safe for printing, replacing unknown chars."""
    return str(s).encode('ascii', errors='replace').decode('ascii')

def p(*args, **kwargs):
    """Print to both the log file and stdout (safely)."""
    msg = ' '.join(str(a) for a in args)
    _log_file.write(msg + '\n')
    _log_file.flush()
    try:
        print(safe(msg), **kwargs)
    except Exception:
        pass

PRETRAINING_DIR = r"E:\AG_Dataset\AG-Foundational-Model\Pretraining"
SHARDS_DIR = r"E:\AG_Dataset\shards"
SAMPLES_PER_ZIP = 15   # images to sample per dataset for shape analysis
SHARD_CHECK_COUNT = 10  # number of shards to open
SAMPLES_PER_SHARD = 30

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def human_size(nb):
    for unit in ['B','KB','MB','GB','TB']:
        if nb < 1024:
            return f"{nb:.1f} {unit}"
        nb /= 1024
    return f"{nb:.1f} PB"

def is_image(name):
    return Path(name).suffix.lower() in SUPPORTED_EXTS

def sample_images_from_zip(zip_path, n_samples=SAMPLES_PER_ZIP):
    """Return list of (width, height, mode) for sampled images in zip."""
    results = []
    img_count = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = [m for m in zf.namelist() if is_image(m) and '__MACOSX' not in m]
            img_count = len(members)
            sampled = random.sample(members, min(n_samples, len(members)))
            for name in sampled:
                try:
                    with zf.open(name) as f:
                        data = f.read()
                    img = Image.open(BytesIO(data))
                    results.append((img.width, img.height, img.mode))
                except Exception:
                    pass
    except Exception as e:
        return img_count, results, str(e)
    return img_count, results, None

def analyze_pretraining_dir(directory):
    rows = []
    items = sorted(os.listdir(directory))
    for item in items:
        full_path = os.path.join(directory, item)
        size_bytes = 0
        img_count = 0
        shapes = []
        error = None
        
        if item.endswith('.zip') and os.path.isfile(full_path):
            size_bytes = os.path.getsize(full_path)
            img_count, shapes, error = sample_images_from_zip(full_path, SAMPLES_PER_ZIP)
            
        elif os.path.isdir(full_path):
            # Walk and count
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    fp = os.path.join(root, f)
                    size_bytes += os.path.getsize(fp)
                    if is_image(f):
                        img_count += 1
                        if len(shapes) < SAMPLES_PER_ZIP:
                            try:
                                img = Image.open(fp)
                                shapes.append((img.width, img.height, img.mode))
                                img.close()
                            except Exception:
                                pass
        else:
            size_bytes = os.path.getsize(full_path) if os.path.isfile(full_path) else 0

        # Compute shape statistics
        if shapes:
            widths = [s[0] for s in shapes]
            heights = [s[1] for s in shapes]
            modes = Counter(s[2] for s in shapes)
            w_min, w_max = min(widths), max(widths)
            h_min, h_max = min(heights), max(heights)
            avg_w = sum(widths) / len(widths)
            avg_h = sum(heights) / len(heights)
            dominant_mode = modes.most_common(1)[0][0]
            shape_summary = f"W:[{w_min}-{w_max}] H:[{h_min}-{h_max}] Avg:{avg_w:.0f}x{avg_h:.0f} Mode:{dominant_mode}"
            # Variance flag
            variance = "HIGH" if (w_max / max(w_min, 1)) > 5 or (h_max / max(h_min, 1)) > 5 else "LOW"
        else:
            shape_summary = "N/A"
            variance = "N/A"
        
        grayscale_pct = 0
        if shapes:
            grayscale_pct = round(sum(1 for s in shapes if s[2] in ('L', 'LA')) / len(shapes) * 100)
        
        rows.append({
            'dataset': item,
            'size': human_size(size_bytes),
            'size_bytes': size_bytes,
            'img_count': img_count,
            'shape_summary': shape_summary,
            'scale_variance': variance,
            'grayscale_pct': grayscale_pct,
            'error': error,
        })
        p(f"  OK {safe(item)[:55]:<55} | {human_size(size_bytes):>9} | {img_count:>7} imgs | {shape_summary}")
    return rows

def analyze_shards(shards_dir):
    p(f"\n--- Shard Analysis: {shards_dir} ---")
    shards = sorted(glob.glob(os.path.join(shards_dir, '*.tar')))
    total_size = sum(os.path.getsize(s) for s in shards)
    p(f"Total shards: {len(shards)}")
    p(f"Total size: {human_size(total_size)}")
    
    # Deep sample
    random.seed(42)
    to_check = random.sample(shards, min(SHARD_CHECK_COUNT, len(shards)))
    shapes = []
    modes = Counter()
    formats = Counter()
    corrupt = 0
    tiny = 0
    huge = 0
    
    for shard in to_check:
        try:
            with tarfile.open(shard, 'r') as tar:
                members = tar.getmembers()
                img_members = [m for m in members if is_image(m.name)]
                sampled = random.sample(img_members, min(SAMPLES_PER_SHARD, len(img_members)))
                for m in sampled:
                    try:
                        f = tar.extractfile(m)
                        img = Image.open(f)
                        img.load()
                        shapes.append((img.width, img.height))
                        modes[img.mode] += 1
                        formats[img.format or 'UNKNOWN'] += 1
                        if img.width < 64 or img.height < 64:
                            tiny += 1
                        if img.width > 4000 or img.height > 4000:
                            huge += 1
                        img.close()
                    except Exception:
                        corrupt += 1
        except Exception as e:
            p(f"  Error opening shard: {e}")
    
    if shapes:
        widths = [s[0] for s in shapes]
        heights = [s[1] for s in shapes]
        p(f"\nSampled {len(shapes)} images from {len(to_check)} shards:")
        p(f"  Width:  min={min(widths):5d}  max={max(widths):6d}  avg={sum(widths)/len(widths):7.1f}")
        p(f"  Height: min={min(heights):5d}  max={max(heights):6d}  avg={sum(heights)/len(heights):7.1f}")
        
        # Distribution buckets
        buckets = Counter()
        for w, h in shapes:
            if w < 64 or h < 64: buckets['<64px'] += 1
            elif w < 224 or h < 224: buckets['64-223px'] += 1
            elif w == 224 and h == 224: buckets['224x224 exact'] += 1
            elif w < 512: buckets['224-511px'] += 1
            elif w < 1024: buckets['512-1023px'] += 1
            elif w < 2048: buckets['1024-2047px'] += 1
            elif w < 4096: buckets['2048-4095px'] += 1
            else: buckets['>4096px'] += 1
        p(f"\n  Size Distribution:")
        for k, v in sorted(buckets.items()):
            p(f"    {k:>15}: {v:>3} ({v/len(shapes)*100:.1f}%)")
        
        p(f"\n  Image Modes: {dict(modes)}")
        p(f"  Formats: {dict(formats)}")
        p(f"  Corrupt/unreadable: {corrupt}")
        p(f"  Tiny (<64px): {tiny} ({tiny/len(shapes)*100:.1f}%)")
        p(f"  Huge (>4000px): {huge} ({huge/len(shapes)*100:.1f}%)")
    
    return shapes, formats, modes

if __name__ == '__main__':
    random.seed(42)
    p("=" * 90)
    p(" PRETRAINING DATASET COMPREHENSIVE ANALYSIS ")
    p("=" * 90)
    p(f"\n--- Analyzing ZIP files and directories in {PRETRAINING_DIR} ---")
    rows = analyze_pretraining_dir(PRETRAINING_DIR)
    
    # Save table as JSON for artifact use
    out_path = r"E:\AG_Dataset\AG-Foundational-Model\pretraining_analysis.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    p(f"\n[Saved detailed results to {out_path}]")
    
    # Summary table
    total_imgs = sum(r['img_count'] for r in rows)
    total_size_gb = sum(r['size_bytes'] for r in rows) / (1024**3)
    p(f"\n{'='*90}")
    p(f" SUMMARY: {len(rows)} items | ~{total_imgs:,} total images | {total_size_gb:.1f} GB on disk")
    p(f"{'='*90}")
    
    high_variance = [r for r in rows if r['scale_variance'] == 'HIGH']
    p(f" HIGH scale variance datasets: {len(high_variance)}")
    for r in high_variance:
        p(f"   -> {safe(r['dataset'])[:60]}: {r['shape_summary']}")
    
    grayscale = [r for r in rows if r['grayscale_pct'] > 20]
    p(f" Datasets with >20% grayscale images: {len(grayscale)}")
    for r in grayscale:
        p(f"   -> {safe(r['dataset'])[:60]}: {r['grayscale_pct']}% grayscale")
    
    # Shard analysis
    analyze_shards(SHARDS_DIR)
    _log_file.close()
    print(f"\nFull log saved to: {LOG_PATH}")
