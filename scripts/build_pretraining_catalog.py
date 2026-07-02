#!/usr/bin/env python3
"""
build_pretraining_catalog.py
============================
Scans the Pretraining directory from scratch — reading inside every ZIP and
TAR archive plus every extracted sub-directory — and produces a clean
catalog.csv that is ready for use in the PyTorch pretraining DataLoader.

Rules applied
-------------
  * Only image files (.jpg / .jpeg / .png / .tif / .tiff / .bmp) are kept.
  * Paths containing mask / label / _gt / ground_truth tokens are excluded.
  * The Evaluation/ sub-directory is entirely skipped.
  * Known duplicate ZIPs (e.g. "Plant Disease Expert.zip" vs "-016.zip") are
    de-duplicated by skipping the non-versioned copy.
  * Archives that cannot be opened (corrupted, etc.) are logged and skipped.
  * .tar.gz archives are streamed with live progress every 10k members.
  * Plain .tar archives are also streamed (no random-seek required).

Catalog schema (CSV columns)
-----------------------------
  path         : portable DataLoader path
                   - archive embedded:  "archive_name.zip::inner/path.jpg"
                   - plain file:        "OPPD/images/foo.jpg"
  group        : inferred label / category from directory structure
  source_name  : stem of the originating archive or directory

DataLoader helpers (importable)
--------------------------------
  from scripts.build_pretraining_catalog import load_catalog, open_image_from_record

  records = load_catalog("Pretraining/catalog.csv")
  img = open_image_from_record(Path("Pretraining"), records[0]["path"])

Usage
-----
  python scripts/build_pretraining_catalog.py

  # Custom paths:
  python scripts/build_pretraining_catalog.py ^
      --pretraining-root "e:/AG_Dataset/AG-Foundational-Model/Pretraining" ^
      --output "e:/AG_Dataset/AG-Foundational-Model/Pretraining/catalog.csv"
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import zipfile
import tarfile
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

<<<<<<< HEAD
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Lower-cased substrings that identify masks / labels / ground-truth
EXCLUDE_TOKENS = (
    "/masks/",         "\\masks\\",
    "/mask/",          "\\mask\\",
    "_mask.",          "_masks.",          "mask.",
    "/labels/",        "\\labels\\",
    "/label/",         "\\label\\",
    "_label.",         "label.",
    "_gt.",            "_groundtruth.",
    "/gt/",            "\\gt\\",
    "/annotations/",   "\\annotations\\",
    # Agriculture-Vision segmentation channel images
    "_boundary.",      "_plant.",          "_weed.",
=======
from ag_foundation.data.dataset import AgricultureImageDataset  # noqa: E402
from ag_foundation.data.multi_source_dataset import (  # noqa: E402
    scan_pretraining_directory,
>>>>>>> 33c63a88879f064cce6e7e60a11fa3ba55e170bd
)

# Known exact-duplicate archive stems. Key = skip; Value = keep instead.
KNOWN_DUPLICATES: dict[str, str] = {
    "Plant Disease Expert.zip":                   "Plant Disease Expert-016.zip",
    "Plant Leaves for Image Classification.zip":  "Plant Leaves for Image Classification-004.zip",
    "rice+leaf+diseases.zip":                     "Rice Leaf Diseases Dataset.zip",
}

# Print a live progress line after this many image members (for big TARs)
PROGRESS_INTERVAL = 25_000


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def is_ground_truth(inner_path: str) -> bool:
    p = inner_path.lower().replace("\\", "/")
    return any(tok.replace("\\", "/") in p for tok in EXCLUDE_TOKENS)


def is_valid_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# Group inference
# ---------------------------------------------------------------------------

def infer_group(inner_path: str, source_name: str) -> str:
    """Return the most meaningful parent-folder label for the image."""
    parts = Path(inner_path.replace("\\", "/")).parts
    # skip common uninformative folder names
    SKIP = {".", "images", "train", "val", "test", "valid", "data",
            "image", "img", "imgs", "train2", "val2"}
    parents = [p for p in parts[:-1] if p.lower() not in SKIP]
    if parents:
        return "/".join(parents[-3:])
    return source_name


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------

def scan_zip(archive_path: Path, source_name: str) -> list[dict]:
    rows: list[dict] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.infolist()
            n_total = len(members)
            n_img = 0
            t0 = time.time()
            for idx, info in enumerate(members):
                if info.is_dir():
                    continue
                inner = info.filename
                if not is_valid_image(inner):
                    continue
                if is_ground_truth(inner):
                    continue
                rows.append({
                    "path":        f"{archive_path.name}::{inner}",
                    "group":       infer_group(inner, source_name),
                    "source_name": source_name,
                })
                n_img += 1
                if n_img > 0 and n_img % PROGRESS_INTERVAL == 0:
                    elapsed = time.time() - t0
                    print(f"    ... {n_img:,} images found so far  ({elapsed:.0f}s)", flush=True)
    except zipfile.BadZipFile as exc:
        print(f"\n  [WARN] Cannot open zip {archive_path.name}: {exc}", flush=True)
    return rows


def scan_tar_streaming(archive_path: Path, source_name: str) -> list[dict]:
    """
    Stream a TAR or TAR.GZ without seeking. Works for both compressed and
    uncompressed TARs. Prints progress every PROGRESS_INTERVAL images.
    """
    rows: list[dict] = []
    # Use r: (no compression) for .tar, r:gz for .tar.gz
    if archive_path.name.endswith(".tar.gz"):
        mode = "r:gz"
    elif archive_path.name.endswith(".tar"):
        mode = "r:"
    else:
        mode = "r:*"

    n_img = 0
    t0 = time.time()
    try:
        with tarfile.open(archive_path, mode) as tf:
            while True:
                try:
                    member = tf.next()
                except StopIteration:
                    break
                if member is None:
                    break
                if member.isdir():
                    continue
                inner = member.name
                if not is_valid_image(inner):
                    continue
                if is_ground_truth(inner):
                    continue
                rows.append({
                    "path":        f"{archive_path.name}::{inner}",
                    "group":       infer_group(inner, source_name),
                    "source_name": source_name,
                })
                n_img += 1
                if n_img % PROGRESS_INTERVAL == 0:
                    elapsed = time.time() - t0
                    rate = n_img / max(elapsed, 0.001)
                    print(f"    ... {n_img:,} images found  ({elapsed:.0f}s, {rate:.0f} img/s)",
                          flush=True)
    except (tarfile.TarError, EOFError) as exc:
        print(f"\n  [WARN] Error reading {archive_path.name}: {exc}", flush=True)
    return rows


def scan_directory(dir_path: Path, pretraining_root: Path) -> list[dict]:
    """Recursively walk an extracted directory."""
    rows: list[dict] = []
    source_name = dir_path.name
    n_img = 0
    t0 = time.time()
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d != "Evaluation"]
        for f in files:
            if not is_valid_image(f):
                continue
            full = Path(root) / f
            rel = full.relative_to(pretraining_root).as_posix()
            if is_ground_truth(rel):
                continue
            group = infer_group(str(full.relative_to(dir_path)), source_name)
            rows.append({
                "path":        rel,
                "group":       group,
                "source_name": source_name,
            })
            n_img += 1
            if n_img % PROGRESS_INTERVAL == 0:
                print(f"    ... {n_img:,} images found  ({time.time()-t0:.0f}s)",
                      flush=True)
    return rows


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_catalog(pretraining_root: Path, output_path: Path) -> None:
    pretraining_root = pretraining_root.resolve()
    print(f"\n{'='*70}")
    print(f" Pretraining Catalog Builder  (from scratch)")
    print(f"   Root  : {pretraining_root}")
    print(f"   Output: {output_path}")
    print(f"{'='*70}\n", flush=True)

    all_rows: list[dict] = []
    skipped_dups: list[str] = []
    format_counts: Counter = Counter()
    source_counts: dict[str, int] = {}

    zip_files  = sorted(pretraining_root.glob("*.zip"))
    tar_plain  = sorted(pretraining_root.glob("*.tar"))
    tar_gz     = sorted(pretraining_root.glob("*.tar.gz"))
    dirs       = [d for d in sorted(pretraining_root.iterdir())
                  if d.is_dir() and d.name != "Evaluation"]

    print(f"Discovered:")
    print(f"  {len(zip_files)} ZIP archives")
    print(f"  {len(tar_plain)} plain .tar archives")
    print(f"  {len(tar_gz)} .tar.gz archives")
    print(f"  {len(dirs)} directories")
    print(flush=True)

    # ---- ZIPs ----
    print(f"{'─'*70}")
    print(f"Phase 1/3: ZIP files")
    print(f"{'─'*70}", flush=True)
    for i, zp in enumerate(zip_files, 1):
        if zp.name in KNOWN_DUPLICATES:
            preferred = KNOWN_DUPLICATES[zp.name]
            print(f"  [{i:02}/{len(zip_files)}] SKIP duplicate: {zp.name}")
            print(f"         (use {preferred} instead)", flush=True)
            skipped_dups.append(zp.name)
            continue
        source_name = zp.stem
        size_gb = zp.stat().st_size / 1e9
        t0 = time.time()
        print(f"  [{i:02}/{len(zip_files)}] {zp.name}  ({size_gb:.1f} GB)", flush=True)
        rows = scan_zip(zp, source_name)
        elapsed = time.time() - t0
        n = len(rows)
        print(f"         → {n:,} images  ({elapsed:.1f}s)", flush=True)
        all_rows.extend(rows)
        source_counts[source_name] = n
        for r in rows:
            format_counts[Path(r["path"]).suffix.lower()] += 1

    # ---- Plain TARs ----
    print(f"\n{'─'*70}")
    print(f"Phase 2/3: TAR files (streaming, no extraction)")
    print(f"{'─'*70}", flush=True)
    all_tars = tar_plain + tar_gz
    for i, tp in enumerate(all_tars, 1):
        source_name = tp.name.replace(".tar.gz", "").replace(".tar", "")
        size_gb = tp.stat().st_size / 1e9
        t0 = time.time()
        print(f"  [{i:02}/{len(all_tars)}] {tp.name}  ({size_gb:.1f} GB)", flush=True)
        rows = scan_tar_streaming(tp, source_name)
        elapsed = time.time() - t0
        n = len(rows)
        print(f"         → {n:,} images  ({elapsed:.1f}s total)", flush=True)
        all_rows.extend(rows)
        source_counts[source_name] = n
        for r in rows:
            format_counts[Path(r["path"]).suffix.lower()] += 1

    # ---- Directories ----
    print(f"\n{'─'*70}")
    print(f"Phase 3/3: Extracted directories")
    print(f"{'─'*70}", flush=True)
    for i, dp in enumerate(dirs, 1):
        t0 = time.time()
        print(f"  [{i:02}/{len(dirs)}] {dp.name}/", flush=True)
        rows = scan_directory(dp, pretraining_root)
        elapsed = time.time() - t0
        n = len(rows)
        print(f"         → {n:,} images  ({elapsed:.1f}s)", flush=True)
        all_rows.extend(rows)
        source_counts[dp.name] = n
        for r in rows:
            format_counts[Path(r["path"]).suffix.lower()] += 1

    # ---- Write catalog ----
    print(f"\nWriting catalog to {output_path} ...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "group", "source_name"])
        writer.writeheader()
        writer.writerows(all_rows)

    # ---- Summary ----
    total = len(all_rows)
    print(f"\n{'='*70}")
    print(f" CATALOG COMPLETE")
    print(f"{'='*70}")
    print(f"  Total pretraining images : {total:>12,}")
    print(f"  Duplicate archives skipped: {len(skipped_dups):>10} ({', '.join(skipped_dups)})")
    print(f"  Output file              : {output_path}")
    print(f"\n  Format breakdown:")
    for ext, cnt in format_counts.most_common():
        pct = 100 * cnt / max(total, 1)
        bar = "█" * int(pct / 2)
        print(f"    {ext.ljust(6)}  {cnt:>12,}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Per-source image counts (largest first):")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(total, 1)
        print(f"    {cnt:>12,}  ({pct:4.1f}%)  {src}")
    print(flush=True)


# ---------------------------------------------------------------------------
# DataLoader helpers (importable)
# ---------------------------------------------------------------------------

def load_catalog(catalog_csv: str | Path) -> list[dict]:
    """
    Load the pre-built catalog into memory for use in a PyTorch Dataset.

    Returns list of dicts with keys: 'path', 'group', 'source_name'.

    Example::

        class PretrainingDataset(torch.utils.data.Dataset):
            def __init__(self, catalog_csv, pretraining_root, transform=None):
                self.records = load_catalog(catalog_csv)
                self.root = Path(pretraining_root)
                self.transform = transform

            def __len__(self):
                return len(self.records)

            def __getitem__(self, idx):
                img = open_image_from_record(self.root, self.records[idx]["path"])
                return self.transform(img) if self.transform else img
    """
    records: list[dict] = []
    with open(catalog_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(row)
    return records


def open_image_from_record(pretraining_root: Path, path: str):
    """
    Open a PIL.Image.Image from a catalog record 'path' field.

    Handles:
      - ZIP-embedded:  "archive.zip::inner/path.jpg"
      - TAR-embedded:  "archive.tar::inner/path.jpg"
      - Plain file:    "OPPD/images/foo.jpg"

    Requires Pillow.
    """
    from PIL import Image

    if "::" not in path:
        return Image.open(pretraining_root / path).convert("RGB")

    archive_name, inner = path.split("::", 1)
    archive_path = pretraining_root / archive_name

    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            data = zf.read(inner)
    elif ".tar" in archive_name:
        mode = "r:gz" if archive_name.endswith(".tar.gz") else "r:"
        with tarfile.open(archive_path, mode) as tf:
            fobj = tf.extractfile(tf.getmember(inner))
            data = fobj.read()
    else:
        raise ValueError(f"Unknown archive type: {archive_name}")

    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean pretraining catalog from all ZIP/TAR/directory sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pretraining-root",
        type=Path,
        default=Path(r"e:\AG_Dataset\AG-Foundational-Model\Pretraining"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <pretraining-root>/catalog.csv)",
    )
    args = parser.parse_args()
    output = args.output or (args.pretraining_root / "catalog.csv")
    build_catalog(args.pretraining_root, output)


if __name__ == "__main__":
    main()
