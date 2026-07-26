#!/usr/bin/env python3
"""
build_pretraining_catalog.py
============================
Scans the Pretraining directory from scratch — reading inside every ZIP and
TAR archive plus every extracted sub-directory — and produces a clean
catalog_v2.csv that is ready for use in the shard builder.

Rules applied
-------------
  * Only image files (.jpg / .jpeg / .png / .tif / .tiff / .bmp) are kept.
  * Paths containing mask / label / _gt / ground_truth tokens are excluded.
  * The Evaluation/ sub-directory is entirely skipped.
  * Known DUPLICATE ZIPs are skipped (non-versioned copy skipped, versioned kept).
  * Known LOW-QUALITY datasets are excluded entirely (too small, wrong domain,
    grayscale satellite patches, or eval-set contamination).
  * Datasets with KNOWN OVERLAP (PlantCLEF full-res vs 800px) are de-duplicated.
  * Archives that cannot be opened (corrupted) are logged and skipped.
  * .tar.gz archives are streamed with live progress every 25k members.

Excluded datasets (KNOWN_EXCLUDES) — rationale
-----------------------------------------------
  GeoPlant*.zip                  : 128×128 px AND 53% grayscale satellite tiles — wrong domain/scale
  DeepWeeds*.zip                 : 256×256 px drone POV — out-of-distribution for close-field model
  Chili Plant Disease*.zip       : <500 images each — statistically negligible
  Rice Leaf Diseases Dataset.zip : 120 images — noise (not Rice Plant diseases dataset.zip which is KEPT)
  rice+leaf+diseases.zip         : duplicate of Rice Leaf Diseases, also 120 images
  Agriculture-Vision-2021.tar.gz : Aerial/satellite imagery — out-of-distribution for close-field model
  FAIR1M/                        : Remote sensing — out-of-distribution
  PlantCLEF2024single*.tar       : Full-res version (288 GB) — overlaps with 800px version, keep 800px
  Toxic Plant*.zip               : Evaluation set — must not contaminate pretraining
  Indian Medicinal*.zip          : Evaluation set
  Pea Plant*.zip                 : Evaluation set
  Agriculture crop images*.zip   : Evaluation set (<1100 images)

Included datasets (by user decision — kept despite borderline resolution)
-------------------------------------------------------------------------
  Plants leafs Dataset-022.zip   : 256×256 px, 190k images — included for scale
  Rice Plant diseases dataset.zip : 300×300 px, 4,684 images — included for crop diversity

Note on Cotton Plant Disease:
  60% of images are grayscale (mode=L). The shard builder's _to_rgb() converts them
  to RGB automatically, but they contribute minimal spectral information.
  The MIN_SIDE filter in build_wds_shards.py handles resolution gating at write time.

Catalog schema (CSV columns)
-----------------------------
  path         : DataLoader path
                   - archive embedded:  "archive.zip::inner/path.jpg"
                   - plain file:        "OPPD/images/foo.jpg"
  group        : inferred category from directory structure
  source_name  : stem of the originating archive or directory

Usage
-----
  python scripts/build_pretraining_catalog.py

  # Custom paths:
  python scripts/build_pretraining_catalog.py ^
      --pretraining-root "E:/AG_Dataset/AG-Foundational-Model/Pretraining" ^
      --output "E:/AG_Dataset/AG-Foundational-Model/Pretraining/catalog_v2.csv"
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
import tqdm
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    # OPPD ground-truth annotation files
    "/annotations",
)

# Known exact-duplicate archive names. Key = SKIP; Value = keep instead.
KNOWN_DUPLICATES: dict[str, str] = {
    "Plant Disease Expert.zip":
        "Plant Disease Expert-016.zip",
    "Plant Leaves for Image Classification.zip":
        "Plant Leaves for Image Classification-004.zip",
}

# Known bad/excluded datasets — skip entire archive.
# These are matched by checking if the archive filename EXACTLY MATCHES any entry here.
KNOWN_EXCLUDES: set[str] = {
    # ── Resolution / Domain filters ────────────────────────────────────────────
    # 128×128 px AND 53% grayscale satellite tiles — completely wrong for close-field YOLO
    "GeoPlant_ Spatial Plant Species Prediction Dataset-008.zip",
    # 256×256 px drone overhead POV — out-of-distribution for close-field model
    "DeepWeeds- A Multiclass Weed Species Image Dataset for Deep Learning.zip",
    # Aerial / satellite remote sensing — out-of-distribution for close-field foundation model
    "Agriculture-Vision-2021.tar.gz",

    # ── Too small — statistical noise ─────────────────────────────────────────
    # <500 images each
    "Chili Plant Disease Detection.zip",
    "Chili Plant Disease.zip",
    # 120 images each — noise + duplicates of each other
    "Rice Leaf Diseases Dataset.zip",
    "rice+leaf+diseases.zip",

    # ── Duplicate archives ─────────────────────────────────────────────────────
    # PlantCLEF full-resolution (281 GB) overlaps with the 800px version
    # Keep: PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar
    "PlantCLEF2024singleplanttrainingdata.tar",

    # ── Evaluation-only — must NOT be in pretraining (data leakage) ───────────
    "Toxic Plant Classification.zip",
    "Indian Medicinal Plant Image Dataset.zip",
    "Pea Plant dataset.zip",
    "Agriculture crop images.zip",
    # These are downstream segmentation / disease benchmarks
    "Paddy Doctor- Paddy Disease Classification.zip",
    "PlantSeg_ A Large-Scale In-the-wild Dataset for Plant Disease Segmentation.zip",
    "Edible wild plants.zip",

    # ── FAIR1M directory excluded — remote sensing ─────────────────────────────
    # Handled separately in scan_directory() via the dirs filter below
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


def should_exclude_archive(archive_name: str) -> bool:
    """Return True if this archive should be skipped entirely."""
    return archive_name in KNOWN_EXCLUDES


# ---------------------------------------------------------------------------
# Group inference
# ---------------------------------------------------------------------------

def infer_group(inner_path: str, source_name: str) -> str:
    """Return the most meaningful parent-folder label for the image."""
    parts = Path(inner_path.replace("\\", "/")).parts
    SKIP = {".", "images", "train", "val", "test", "valid", "data",
            "image", "img", "imgs", "train2", "val2", "rgb", "color",
            "raw", "input", "output", "src", "dataset", "datasets"}
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
            n_img = 0
            t0 = time.time()
            for info in tqdm.tqdm(members, desc=f"  Scan {archive_path.name[:20]}", unit="file", leave=False, dynamic_ncols=True):
                if info.is_dir():
                    continue
                inner = info.filename
                if not is_valid_image(inner):
                    continue
                if is_ground_truth(inner):
                    continue
                if "__MACOSX" in inner or inner.startswith("._"):
                    continue
                rows.append({
                    "path":        f"{archive_path.name}::{inner}",
                    "group":       infer_group(inner, source_name),
                    "source_name": source_name,
                })
    except zipfile.BadZipFile as exc:
        print(f"\n  [WARN] Cannot open zip {archive_path.name}: {exc}", flush=True)
    return rows


import subprocess

def scan_tar_streaming(archive_path: Path, source_name: str) -> list[dict]:
    """
    Stream a TAR or TAR.GZ extremely fast using the native OS tar utility (bsdtar).
    Bypasses Python's slow pure-python tarfile parser.
    """
    rows: list[dict] = []
    n_img = 0
    t0 = time.time()
    
    # Use native tar to just list the file contents rapidly
    try:
        proc = subprocess.Popen(
            ["tar", "-tf", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        for line in tqdm.tqdm(proc.stdout, desc=f"  Scan {archive_path.name[:20]}", unit="file", leave=False, dynamic_ncols=True):
            inner = line.strip()
            
            # Skip directories or empty lines
            if not inner or inner.endswith('/'):
                continue
                
            if not is_valid_image(inner):
                continue
            if is_ground_truth(inner):
                continue
                
            rows.append({
                "path":        f"{archive_path.name}::{inner}",
                "group":       infer_group(inner, source_name),
                "source_name": source_name,
            })
                
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read()
            print(f"\n  [WARN] Native tar returned exit code {proc.returncode} for {archive_path.name}: {err}", flush=True)
            
    except Exception as exc:
        print(f"\n  [WARN] Error invoking native tar for {archive_path.name}: {exc}", flush=True)
        
    return rows


def scan_directory(dir_path: Path, pretraining_root: Path) -> list[dict]:
    """Recursively walk an extracted directory."""
    rows: list[dict] = []
    source_name = dir_path.name
    n_img = 0
    t0 = time.time()
    pbar = tqdm.tqdm(desc=f"  Scan {dir_path.name[:20]}", unit="file", leave=False, dynamic_ncols=True)
    for root, dirs, files in os.walk(dir_path):
        # Skip evaluation sub-dirs, .git, __MACOSX
        dirs[:] = [d for d in dirs if d not in ("Evaluation", ".git", "__MACOSX")]
        for f in files:
            pbar.update(1)
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
    pbar.close()
    return rows


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_catalog(pretraining_root: Path, output_path: Path) -> None:
    pretraining_root = pretraining_root.resolve()
    print(f"\n{'=' * 72}")
    print(f" Pretraining Catalog Builder v2  (clean rebuild)")
    print(f"   Root  : {pretraining_root}")
    print(f"   Output: {output_path}")
    print(f"   Excluded datasets: {len(KNOWN_EXCLUDES)}")
    print(f"   Duplicate skips : {len(KNOWN_DUPLICATES)}")
    print(f"{'=' * 72}\n", flush=True)

    all_rows: list[dict] = []
    skipped_dups: list[str] = []
    skipped_excl: list[str] = []
    format_counts: Counter = Counter()
    source_counts: dict[str, int] = {}

    zip_files = sorted(pretraining_root.glob("*.zip"))
    tar_plain  = sorted(pretraining_root.glob("*.tar"))
    tar_gz     = sorted(pretraining_root.glob("*.tar.gz"))
    # Exclude FAIR1M (remote sensing/satellite) and standard non-data dirs
    EXCLUDED_DIRS = {"Evaluation", ".git", "__pycache__", "FAIR1M"}
    dirs       = [d for d in sorted(pretraining_root.iterdir())
                  if d.is_dir() and d.name not in EXCLUDED_DIRS]

    print(f"Discovered:")
    print(f"  {len(zip_files)} ZIP archives")
    print(f"  {len(tar_plain)} plain .tar archives")
    print(f"  {len(tar_gz)} .tar.gz archives")
    print(f"  {len(dirs)} directories")
    print(flush=True)

    # ---- ZIPs ----
    print(f"{'─' * 72}")
    print(f"Phase 1/3: ZIP files")
    print(f"{'─' * 72}", flush=True)
    for i, zp in enumerate(zip_files, 1):
        name = zp.name

        if name in KNOWN_EXCLUDES:
            print(f"  [{i:02}/{len(zip_files)}] EXCLUDE: {name}")
            skipped_excl.append(name)
            continue

        if name in KNOWN_DUPLICATES:
            preferred = KNOWN_DUPLICATES[name]
            print(f"  [{i:02}/{len(zip_files)}] SKIP duplicate: {name}")
            print(f"         (use {preferred} instead)", flush=True)
            skipped_dups.append(name)
            continue

        source_name = zp.stem
        size_gb = zp.stat().st_size / 1e9
        t0 = time.time()
        print(f"  [{i:02}/{len(zip_files)}] {name}  ({size_gb:.1f} GB)", flush=True)
        rows = scan_zip(zp, source_name)
        elapsed = time.time() - t0
        n = len(rows)
        print(f"         -> {n:,} images  ({elapsed:.1f}s)", flush=True)
        all_rows.extend(rows)
        source_counts[source_name] = n
        for r in rows:
            format_counts[Path(r["path"]).suffix.lower()] += 1

    # ---- TARs ----
    print(f"\n{'─' * 72}")
    print(f"Phase 2/3: TAR files (streaming, no extraction)")
    print(f"{'─' * 72}", flush=True)
    all_tars = tar_plain + tar_gz
    for i, tp in enumerate(all_tars, 1):
        name = tp.name

        if name in KNOWN_EXCLUDES:
            print(f"  [{i:02}/{len(all_tars)}] EXCLUDE: {name}")
            skipped_excl.append(name)
            continue

        source_name = name.replace(".tar.gz", "").replace(".tar", "")
        size_gb = tp.stat().st_size / 1e9
        t0 = time.time()
        print(f"  [{i:02}/{len(all_tars)}] {name}  ({size_gb:.1f} GB)", flush=True)
        rows = scan_tar_streaming(tp, source_name)
        elapsed = time.time() - t0
        n = len(rows)
        print(f"         -> {n:,} images  ({elapsed:.1f}s total)", flush=True)
        all_rows.extend(rows)
        source_counts[source_name] = n
        for r in rows:
            format_counts[Path(r["path"]).suffix.lower()] += 1

    # ---- Directories ----
    print(f"\n{'─' * 72}")
    print(f"Phase 3/3: Extracted directories")
    print(f"{'─' * 72}", flush=True)
    for i, dp in enumerate(dirs, 1):
        t0 = time.time()
        print(f"  [{i:02}/{len(dirs)}] {dp.name}/", flush=True)
        rows = scan_directory(dp, pretraining_root)
        elapsed = time.time() - t0
        n = len(rows)
        print(f"         -> {n:,} images  ({elapsed:.1f}s)", flush=True)
        all_rows.extend(rows)
        source_counts[dp.name] = n
        for r in rows:
            format_counts[Path(r["path"]).suffix.lower()] += 1

    # ---- Shuffle for better shard diversity ----
    import random
    random.seed(42)
    random.shuffle(all_rows)

    # ---- Write catalog ----
    print(f"\n[Catalog] Writing {len(all_rows):,} records to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "group", "source_name"])
        writer.writeheader()
        writer.writerows(all_rows)

    # ---- Summary ----
    total = len(all_rows)
    print(f"\n{'=' * 72}")
    print(f" CATALOG v2 COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Total pretraining images : {total:>12,}")
    print(f"  Excluded datasets        : {len(skipped_excl):>12} ({', '.join(skipped_excl[:3])}...)")
    print(f"  Duplicate archives skipped: {len(skipped_dups):>11} ({', '.join(skipped_dups)})")
    print(f"  Output file              : {output_path}")
    print(f"\n  Format breakdown:")
    for ext, cnt in format_counts.most_common():
        pct = 100 * cnt / max(total, 1)
        bar = "#" * int(pct / 2)
        print(f"    {ext.ljust(6)}  {cnt:>12,}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Per-source image counts (largest first):")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(total, 1)
        try:
            safe_src = src.encode("ascii", errors="replace").decode("ascii")
        except Exception:
            safe_src = repr(src)
        print(f"    {cnt:>12,}  ({pct:4.1f}%)  {safe_src}")
    print(flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean pretraining catalog v2 (with exclusions).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pretraining-root",
        type=Path,
        default=Path(r"E:\AG_Dataset\AG-Foundational-Model\Pretraining"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <pretraining-root>/catalog_v2.csv)",
    )
    args = parser.parse_args()
    output = args.output or (args.pretraining_root / "catalog_v2.csv")
    build_catalog(args.pretraining_root, output)


if __name__ == "__main__":
    main()
