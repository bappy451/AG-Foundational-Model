"""
prepare_plantseg_cls.py

Converts the PlantSeg dataset (YOLO OBB format with flat images/) directories
into an ImageFolder-compatible structure for classification benchmarking.

The disease class is extracted from the image filename prefix (e.g.
"apple_black_rot_1.rf.xxx.jpg" -> class = "apple_black_rot").

Usage:
    python scripts/prepare_plantseg_cls.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

# ----------------------------- CONFIG ----------------------------------------
SRC_ROOT = Path(r"E:\AG_Dataset\01_Evaluation\PlantSeg")
DST_ROOT = Path(r"E:\AG_Dataset\01_Evaluation\PlantSeg_cls")
SPLITS = ["train", "valid", "test"]

# Regex to extract the disease class from the roboflow filename.
# Filenames look like:  apple_black_rot_1.rf.c10c7....jpg
#                       banana_anthracnose_Baidu_0004.rf.xxx.jpg
# Strategy: strip trailing "_<digits>" and "_<rf-hash>" parts, then normalise.
_KNOWN_CLASSES: list[str] = []  # filled dynamically


def extract_class(stem: str) -> str:
    """
    Extract a clean disease-class label from a roboflow image filename stem.

    Roboflow stems look like:
        apple_black_rot_1                    -> apple_black_rot
        apple_black_rot_google_0001          -> apple_black_rot
        banana_anthracnose_Baidu_0004        -> banana_anthracnose
        apple_mosaic_virus_google_0002       -> apple_mosaic_virus
        corn_northern_leaf_blight_2          -> corn_northern_leaf_blight

    Strategy:
      1. Strip the roboflow hash suffix (`.rf.<hash>`) from the stem.
      2. Remove trailing segments that are purely numeric or look like a
         source tag (Baidu / Bing / Google / Google_images / jpg / png etc.)
         followed by an optional numeric counter.
      3. Collapse any remaining double-underscores.
    """
    # Remove roboflow hash: stem is already without extension, but may still
    # contain ".rf." if the caller didn't strip it - handle both.
    stem = re.sub(r"\.rf\.[a-f0-9]+$", "", stem, flags=re.IGNORECASE)

    # Tokens separated by underscores
    parts = stem.split("_")

    # Drop trailing parts that are:
    #   - purely numeric               e.g. "1", "0004"
    #   - source tags                  e.g. "Baidu", "Bing", "Google", "jpg", "png"
    #   - "google" + optional digit    e.g. "google_0002"
    _DROP = {"baidu", "bing", "google", "google_images", "jpg", "png", "images"}
    while parts:
        last = parts[-1].lower()
        if last.isdigit() or last in _DROP:
            parts.pop()
        else:
            break

    return "_".join(parts).lower() if parts else "unknown"


def prepare_split(split: str) -> dict[str, int]:
    src_images = SRC_ROOT / split / "images"
    if not src_images.exists():
        print(f"  [skip] {src_images} not found")
        return {}

    counts: dict[str, int] = {}
    images = list(src_images.glob("*.jpg")) + list(src_images.glob("*.png"))

    for img_path in images:
        cls = extract_class(img_path.stem)
        dst_dir = DST_ROOT / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / img_path.name
        if not dst_file.exists():
            shutil.copy2(img_path, dst_file)
        counts[cls] = counts.get(cls, 0) + 1

    return counts


def main() -> None:
    print(f"Source : {SRC_ROOT}")
    print(f"Dest   : {DST_ROOT}")
    print()

    all_classes: set[str] = set()
    for split in SPLITS:
        print(f"Processing split: {split}")
        counts = prepare_split(split)
        for cls, n in sorted(counts.items()):
            print(f"  {cls:50s}  {n:5d} images")
            all_classes.add(cls)
        print(f"  -> {len(counts)} classes, {sum(counts.values())} images total")
        print()

    print(f"Done. Total unique classes across all splits: {len(all_classes)}")
    for cls in sorted(all_classes):
        print(f"  {cls}")


if __name__ == "__main__":
    main()
