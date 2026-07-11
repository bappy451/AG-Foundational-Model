import os
import glob
import tarfile
from io import BytesIO
from PIL import Image
import collections
import zipfile

def analyze_shards(shards_dir, num_shards_to_check=5, samples_per_shard=20):
    shards = glob.glob(os.path.join(shards_dir, '*.tar'))
    print(f"--- Analyzing Shards in {shards_dir} ---")
    print(f"Total shards found: {len(shards)}")
    if not shards:
        return
    
    total_size = sum(os.path.getsize(s) for s in shards)
    print(f"Total size of shards: {total_size / (1024**3):.2f} GB")
    
    sizes = []
    modes = collections.Counter()
    formats = collections.Counter()
    corrupt = 0
    
    import random
    random.seed(42)
    shards_to_check = random.sample(shards, min(num_shards_to_check, len(shards)))
    
    for shard in shards_to_check:
        try:
            with tarfile.open(shard, 'r') as tar:
                members = tar.getmembers()
                img_members = [m for m in members if m.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                
                sampled = random.sample(img_members, min(samples_per_shard, len(img_members)))
                for m in sampled:
                    f = tar.extractfile(m)
                    try:
                        img = Image.open(f)
                        formats[img.format] += 1
                        modes[img.mode] += 1
                        sizes.append(img.size) # (width, height)
                    except Exception:
                        corrupt += 1
        except Exception as e:
            print(f"Error reading shard {shard}: {e}")
            
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"\nShard Sample Analysis ({len(sizes)} images from {len(shards_to_check)} shards):")
        print(f"Width - Min: {min(widths)}, Max: {max(widths)}, Avg: {sum(widths)/len(widths):.1f}")
        print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Avg: {sum(heights)/len(heights):.1f}")
        aspect_ratios = [w/h for w, h in sizes]
        print(f"Aspect Ratio - Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Avg: {sum(aspect_ratios)/len(aspect_ratios):.2f}")
        print(f"Image Modes: {dict(modes)}")
        print(f"Image Formats: {dict(formats)}")
        print(f"Corrupted/Unreadable images: {corrupt}")
        
        # Check for extreme anomalies
        small = sum(1 for w, h in sizes if w < 64 or h < 64)
        huge = sum(1 for w, h in sizes if w > 4000 or h > 4000)
        print(f"Extremely small images (<64px): {small}")
        print(f"Extremely large images (>4000px): {huge}")

def analyze_raw(raw_dir):
    print(f"\n--- Analyzing Raw Datasets in {raw_dir} ---")
    
    zip_files = glob.glob(os.path.join(raw_dir, '*.zip'))
    dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    print(f"Found {len(zip_files)} ZIP files and {len(dirs)} directories.")
    
    # Check dataset YAML
    yml_path = os.path.join(raw_dir, '..', 'Dataset.yml')
    if os.path.exists(yml_path):
        with open(yml_path, 'r') as f:
            yml_content = f.read()
            print(f"Dataset.yml declares {yml_content.count('- name:')} datasets.")

if __name__ == '__main__':
    analyze_shards(r'E:\AG_Dataset\shards')
    analyze_raw(r'E:\AG_Dataset\AG-Foundational-Model\Pretraining')
