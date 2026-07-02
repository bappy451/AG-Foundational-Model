import os
import glob
import tarfile
from collections import Counter
import io
from PIL import Image
import sys

def analyze_shards(shards_dir):
    tar_files = glob.glob(os.path.join(shards_dir, "*.tar"))
    print(f"Found {len(tar_files)} tar files.", flush=True)
    
    total_files = 0
    modality_counts = Counter()
    size_counts = Counter()
    
    images_to_sample_per_tar = 5
    
    for idx, tf_path in enumerate(tar_files):
        try:
            sampled_in_this_tar = 0
            with tarfile.open(tf_path, 'r') as tar:
                # Do a linear scan which is extremely fast
                for m in tar:
                    total_files += 1
                    if m.isfile():
                        ext = os.path.splitext(m.name)[1].lower()
                        modality_counts[ext] += 1
                        
                        # Extract the first N images we see without seeking backwards
                        if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'] and sampled_in_this_tar < images_to_sample_per_tar:
                            try:
                                f = tar.extractfile(m)
                                if f is not None:
                                    img_bytes = f.read()
                                    with Image.open(io.BytesIO(img_bytes)) as img:
                                        size_counts[img.size] += 1
                                    sampled_in_this_tar += 1
                            except Exception:
                                pass
        except Exception as e:
            print(f"Error reading {tf_path}: {e}", flush=True)
            
        if (idx + 1) % 10 == 0 or (idx + 1) == len(tar_files):
            print(f"Processed {idx + 1}/{len(tar_files)} tar files...", flush=True)

    print("\n" + "="*50, flush=True)
    print("EDA DATA ANALYSIS OF SHARDS", flush=True)
    print("="*50, flush=True)
    print(f"Total Tar Files: {len(tar_files)}", flush=True)
    print(f"Total Files inside shards: {total_files}", flush=True)
    print("\n--- Modalities (File Extensions) ---", flush=True)
    for ext, count in modality_counts.most_common():
        print(f"  {ext if ext else '<no extension>'}: {count}", flush=True)
        
    print("\n--- Sampled Image Dimensions ---", flush=True)
    total_sampled = sum(size_counts.values())
    print(f"Total images sampled for dimensions: {total_sampled}", flush=True)
    for size, count in size_counts.most_common(10):
        percentage = (count / total_sampled) * 100 if total_sampled > 0 else 0
        print(f"  {size[0]}x{size[1]}: {count} ({percentage:.1f}%)", flush=True)
        
if __name__ == "__main__":
    analyze_shards(r"E:\AG_Dataset\shards")
