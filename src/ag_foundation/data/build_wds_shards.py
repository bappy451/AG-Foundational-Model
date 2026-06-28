import zipfile
import webdataset as wds
import os
from tqdm import tqdm
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
EXCLUDED_TOKENS = {"mask", "label", "_gt", "evaluation"}

def _is_valid_image(filename: str) -> bool:
    # Use normalized path with forward slashes for consistent token checking
    normalized_name = filename.replace("\\", "/").lower()
    ext = os.path.splitext(normalized_name)[1]
    
    if ext not in SUPPORTED_EXTENSIONS:
        return False
        
    if any(token in normalized_name for token in EXCLUDED_TOKENS):
        return False
        
    return True

def shard_dir_to_wds(input_dir: str, output_prefix: str, max_count: int = 10000, max_size: float = 1e9):
    """
    Recursively scans input_dir for valid images (both loose files and inside ZIPs)
    and streams them directly into WebDataset tar shards.
    """
    output_dir = os.path.dirname(os.path.abspath(output_prefix))
    base_prefix = os.path.basename(output_prefix)
    
    os.makedirs(output_dir, exist_ok=True)
    pattern = f"{base_prefix}-%06d.tar"
    
    # 1. Gather all targets
    print(f"Scanning {input_dir} for files and ZIP archives...")
    targets = [] # List of tuples: (type, container_path, internal_path_or_none)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            full_path = os.path.join(root, file)
            # Skip output shards if they are accidentally placed in the input dir
            if full_path.endswith(".tar"):
                continue
                
            if file.lower().endswith('.zip'):
                # Defer inside-zip scanning to avoid memory/time overhead here if there are huge zip tables
                targets.append(("zip", full_path, None))
            elif _is_valid_image(full_path):
                targets.append(("file", full_path, None))
                
    print(f"Found {len(targets)} root items (files + ZIPs) to process.")
    
    # 2. Process and write to shards
    total_written = 0
    original_cwd = os.getcwd()
    try:
        # WebDataset interprets absolute paths on Windows (e.g. C:\) as URI schemes and fails.
        # We circumvent this by changing into the output directory and writing locally.
        os.chdir(output_dir)
        
        with wds.ShardWriter(pattern, maxsize=max_size, maxcount=max_count) as sink:
            # Wrap the root targets in a progress bar
            for t_type, file_path, _ in tqdm(targets, desc="Processing root items"):
                
                if t_type == "file":
                    ext = os.path.splitext(file_path)[1].lstrip('.')
                    try:
                        with open(file_path, 'rb') as f:
                            file_bytes = f.read()
                        sample = {
                            "__key__": f"{total_written:09d}",
                            ext: file_bytes
                        }
                        sink.write(sample)
                        total_written += 1
                    except Exception as e:
                        print(f"\nFailed to read loose file {file_path}: {e}")
                        
                elif t_type == "zip":
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zf:
                            infolist = zf.infolist()
                            # Filter valid images inside the ZIP
                            valid_zip_files = [info for info in infolist if not info.is_dir() and _is_valid_image(info.filename)]
                            
                            # We don't show a nested tqdm bar to avoid messy outputs, 
                            # but we can process them sequentially.
                            for info in valid_zip_files:
                                ext = os.path.splitext(info.filename)[1].lstrip('.')
                                try:
                                    with zf.open(info) as f:
                                        file_bytes = f.read()
                                    sample = {
                                        "__key__": f"{total_written:09d}",
                                        ext: file_bytes
                                    }
                                    sink.write(sample)
                                    total_written += 1
                                except Exception as e:
                                    print(f"\nFailed to read {info.filename} from {file_path}: {e}")
                    except Exception as e:
                        print(f"\nFailed to open ZIP archive {file_path}: {e}")
    finally:
        os.chdir(original_cwd)

    print(f"\nDone! Successfully written {total_written} valid images to WebDataset shards.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a Directory of ZIPs and Images to WebDataset Shards")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to input directory containing ZIPs and/or images")
    parser.add_argument("--output-prefix", type=str, required=True, help="Prefix for output shards, e.g., /data/shards/dataset")
    parser.add_argument("--max-count", type=int, default=10000, help="Max images per shard")
    parser.add_argument("--max-size", type=float, default=1e9, help="Max bytes per shard (default 1GB)")
    args = parser.parse_args()
    
    shard_dir_to_wds(args.input_dir, args.output_prefix, args.max_count, args.max_size)
