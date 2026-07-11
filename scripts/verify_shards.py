#!/usr/bin/env python3
"""
verify_shards.py
================
After rebuilding shards, this script verifies every shard to confirm:

  1. No shard is more than 1.5x the target size (catches 13 GB anomaly bug)
  2. All sampled images are RGB, min 64px, min-side <= 1100px
  3. No corrupt images in sample
  4. Prints per-shard size histogram and total image count estimate

Usage
-----
  python scripts/verify_shards.py --shards-dir "E:/AG_Dataset/shards"
  python scripts/verify_shards.py --shards-dir "E:/AG_Dataset/shards" --deep
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import tarfile
from collections import Counter
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    print("[ERROR] Pillow required: pip install Pillow", file=sys.stderr)
    sys.exit(1)

TARGET_SHARD_SIZE = 1_000_000_000   # 1 GB
ANOMALY_THRESHOLD = 1.5             # flag shards > 1.5x target
SAMPLES_PER_SHARD = 20              # images to sample per shard for quality check
MAX_EXPECTED_MIN_SIDE = 1100        # builder bounds min-side to 1024px, with tolerance


def check_shard(shard_path: str, deep: bool = False, target_size: int = TARGET_SHARD_SIZE) -> dict:
    result = {
        "path": shard_path,
        "size_bytes": os.path.getsize(shard_path),
        "image_count": 0,
        "corrupt": 0,
        "non_rgb": 0,
        "too_large": 0,
        "too_small": 0,
        "widths": [],
        "heights": [],
        "ok": True,
        "issues": [],
    }

    size = result["size_bytes"]
    if size > target_size * ANOMALY_THRESHOLD:
        result["ok"] = False
        result["issues"].append(f"OVERSIZED: {size/1e9:.2f} GB > {target_size*ANOMALY_THRESHOLD/1e9:.2f} GB threshold")

    try:
        with tarfile.open(shard_path, "r:") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".jpg")]
            result["image_count"] = len(members)

            if deep or len(members) <= SAMPLES_PER_SHARD:
                to_sample = members
            else:
                to_sample = random.sample(members, SAMPLES_PER_SHARD)

            for m in to_sample:
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        result["corrupt"] += 1
                        continue
                    img = Image.open(BytesIO(f.read()))
                    img.load()

                    if img.mode != "RGB":
                        result["non_rgb"] += 1
                        result["issues"].append(f"Non-RGB: {m.name} is {img.mode}")

                    w, h = img.size
                    result["widths"].append(w)
                    result["heights"].append(h)

                    if min(w, h) > MAX_EXPECTED_MIN_SIDE:
                        result["too_large"] += 1
                        result["issues"].append(f"Min-side too large: {m.name} is {w}x{h}")
                    if w < 64 or h < 64:
                        result["too_small"] += 1
                        result["issues"].append(f"Too small: {m.name} is {w}x{h}")

                except Exception as exc:
                    result["corrupt"] += 1
                    result["issues"].append(f"Corrupt: {m.name}: {exc}")

    except Exception as exc:
        result["ok"] = False
        result["issues"].append(f"Cannot open shard: {exc}")

    if result["corrupt"] > 0:
        result["ok"] = False
    if result["non_rgb"] > 0:
        result["ok"] = False
    if result["too_large"] > 0:
        result["ok"] = False
    if result["too_small"] > 0:
        result["ok"] = False

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify WebDataset shards after rebuild.")
    parser.add_argument(
        "--shards-dir",
        type=Path,
        default=Path(r"E:\AG_Dataset\shards"),
        help="Directory containing .tar shards",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Sample ALL images in every shard (slow but thorough)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_SHARD_SIZE,
        help="Expected max shard size in bytes (default: 1 GB)",
    )
    args = parser.parse_args()

    shards = sorted(glob.glob(str(args.shards_dir / "*.tar")))
    if not shards:
        print(f"[ERROR] No .tar shards found in {args.shards_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 72}")
    print(f" AG-Foundation Shard Verifier")
    print(f"{'=' * 72}")
    print(f"  Shards dir  : {args.shards_dir}")
    print(f"  Shard count : {len(shards)}")
    print(f"  Mode        : {'DEEP (all images)' if args.deep else f'SAMPLE ({SAMPLES_PER_SHARD} images/shard)'}")
    print(f"  Target size : {args.target_size/1e9:.2f} GB")
    print(f"  Resize check: min-side <= {MAX_EXPECTED_MIN_SIDE}px")
    print(f"{'=' * 72}\n")

    total_size_gb = sum(os.path.getsize(s) for s in shards) / 1e9
    print(f"Total shard size: {total_size_gb:.1f} GB\n")

    # Size distribution
    size_buckets = Counter()
    target_gb = args.target_size / 1e9
    for s in shards:
        gb = os.path.getsize(s) / 1e9
        if gb > target_gb * ANOMALY_THRESHOLD:
            size_buckets[f">{target_gb * ANOMALY_THRESHOLD:.1f} GB (ANOMALY)"] += 1
        elif gb > target_gb:
            size_buckets[f"{target_gb:.1f}-{target_gb * ANOMALY_THRESHOLD:.1f} GB"] += 1
        elif gb > target_gb * 0.5:
            size_buckets[f"{target_gb * 0.5:.1f}-{target_gb:.1f} GB"] += 1
        elif gb > target_gb * 0.1:
            size_buckets[f"{target_gb * 0.1:.1f}-{target_gb * 0.5:.1f} GB"] += 1
        else:
            size_buckets[f"<{target_gb * 0.1:.1f} GB (tail)"] += 1
    print("Shard size distribution:")
    for bucket, count in sorted(size_buckets.items()):
        bar = "#" * count
        print(f"  {bucket:20s}: {count:4d}  {bar}")
    print()

    # Check for the 13 GB anomaly specifically
    oversized = [s for s in shards if os.path.getsize(s) > args.target_size * ANOMALY_THRESHOLD]
    if oversized:
        print(f"[WARN] {len(oversized)} OVERSIZED shards found:")
        for s in oversized:
            print(f"  {Path(s).name}: {os.path.getsize(s)/1e9:.2f} GB")
        print()

    # Sample-based quality check
    print(f"Running quality checks on all {len(shards)} shards...")
    random.seed(42)

    issues_total = 0
    ok_count = 0
    total_images = 0
    all_widths = []
    all_heights = []

    for i, shard in enumerate(shards):
        result = check_shard(shard, deep=args.deep, target_size=args.target_size)
        total_images += result["image_count"]
        all_widths.extend(result["widths"])
        all_heights.extend(result["heights"])

        if result["ok"]:
            ok_count += 1
        else:
            issues_total += 1
            print(f"  [FAIL] {Path(shard).name}:")
            for issue in result["issues"][:5]:
                print(f"         {issue}")

        if (i + 1) % 50 == 0 or (i + 1) == len(shards):
            print(f"  Progress: {i+1}/{len(shards)} shards checked...", flush=True)

    # Summary
    print(f"\n{'=' * 72}")
    print(f" VERIFICATION SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total shards  : {len(shards)}")
    print(f"  Passed        : {ok_count}  ({100*ok_count/len(shards):.1f}%)")
    print(f"  Failed        : {issues_total}")
    print(f"  Total images  : ~{total_images:,} (exact counts from member scan)")
    print(f"  Total size    : {total_size_gb:.1f} GB")

    if all_widths:
        print(f"\n  Image size stats (from {len(all_widths):,} sampled images):")
        print(f"    Width  min={min(all_widths):5d}  max={max(all_widths):5d}  avg={sum(all_widths)/len(all_widths):.0f}")
        print(f"    Height min={min(all_heights):5d}  max={max(all_heights):5d}  avg={sum(all_heights)/len(all_heights):.0f}")

        over1024 = sum(1 for w, h in zip(all_widths, all_heights) if min(w, h) > 1024)
        if over1024:
            print(f"\n  [WARN] {over1024} sampled images have min-side > 1024px -- bounded resize may not have applied")
        else:
            print(f"\n  [OK] All sampled images follow the 1024px min-side bound")

    if issues_total == 0:
        print(f"\n  RESULT: ALL SHARDS PASSED")
    else:
        print(f"\n  RESULT: {issues_total} SHARDS HAVE ISSUES — review output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
