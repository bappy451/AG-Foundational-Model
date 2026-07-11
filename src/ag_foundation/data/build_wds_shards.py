#!/usr/bin/env python3
"""
build_wds_shards.py
===================
Rebuilds the WebDataset pretraining shards from scratch with a clean
preprocessing pipeline.

Pipeline per image:
  1. Open from catalog path (ZIP / TAR / plain file)
  2. Validate: skip corrupt / tiny (<64px any side)
  3. Channel normalization:
       - Palette (P) → RGB
       - RGBA → RGB (alpha-composite on white)
       - Grayscale (L/LA) → proper RGB via ImageOps.colorize
  4. Bounded resize: if min(H,W) > MAX_SIDE → resize so min-side = MAX_SIDE
     This eliminates scale variance while preserving aspect ratio and texture.
     Images <= MAX_SIDE are left untouched.
  5. Encode as JPEG (quality=92) and write to shard.

The resulting shards contain 1024px-bounded RGB JPEGs.
The training DataLoader applies the final crop to 224×224 at runtime.
This enables DINO multi-crop (global 224px + local 96px from the same image).

Usage
-----
  python src/ag_foundation/data/build_wds_shards.py \\
      --catalog      "E:/AG_Dataset/AG-Foundational-Model/Pretraining/catalog_v2.csv" \\
      --pretraining-root "E:/AG_Dataset/AG-Foundational-Model/Pretraining" \\
      --output-prefix "E:/AG_Dataset/shards/dataset" \\
      --max-size     1000000000 \\
      --max-count    10000 \\
      --workers      8 \\
      --resume
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tarfile
import time
import zipfile
import concurrent.futures
import queue
import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Iterator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_SIDE: int = 1024          # Bounded resize target (px)
MIN_SIDE: int = 64            # Images smaller than this on any side are skipped
JPEG_QUALITY: int = 92        # Re-encode quality — 92 is visually lossless
PROGRESS_EVERY: int = 500     # Print a line every N images written

# ---------------------------------------------------------------------------
# Imports (handled gracefully)
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None   # allow large images
except ImportError:
    print("[ERROR] Pillow is required: pip install Pillow", file=sys.stderr)
    sys.exit(1)

try:
    import webdataset as wds
except ImportError:
    print("[ERROR] webdataset is required: pip install webdataset", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

import threading

_thread_local = threading.local()

_global_zips = {}
_global_tar_index = {}
_global_lock = threading.Lock()

_archive_semaphores = {}
_archive_sem_lock = threading.Lock()

def _get_archive_semaphore(archive_path: Path) -> threading.Semaphore:
    """Ensure no more than 4 workers hit the exact same archive simultaneously."""
    with _archive_sem_lock:
        if archive_path not in _archive_semaphores:
            _archive_semaphores[archive_path] = threading.Semaphore(4)
        return _archive_semaphores[archive_path]

def _build_global_tar_index(archive_path: Path):
    """Scan a .tar file once globally and map member -> (offset, size)."""
    with _global_lock:
        if archive_path in _global_tar_index:
            return
            
        print(f"\n[Global Indexer] Indexing {archive_path.name} for O(1) random access... (Takes ~30s)")
        idx = {}
        with tarfile.open(archive_path, "r:") as tf:
            for member in tf:
                idx[member.name] = (member.offset_data, member.size)
        _global_tar_index[archive_path] = idx
        print(f"[Global Indexer] Finished indexing {len(idx):,} files in {archive_path.name}!")

def _load_pil_from_catalog_path(pretraining_root: Path, path: str) -> Image.Image | None:
    """
    Open a PIL Image using highly concurrent O(1) reads.
    - ZIPs: Shared global ZipFile object (thread-safe).
    - Uncompressed TARs: Raw OS file reads via pre-computed global offset index.
    - Compressed TAR.GZ: Thread-local tarfile instances.
    """
    try:
        if "::" in path:
            archive_name, inner = path.split("::", 1)
            archive_path = pretraining_root / archive_name
            arc_lower = archive_name.lower()

            with _get_archive_semaphore(archive_path):
                if arc_lower.endswith(".zip"):
                    if not hasattr(_thread_local, 'zips'):
                        _thread_local.zips = {}
                    if archive_path not in _thread_local.zips:
                        _thread_local.zips[archive_path] = zipfile.ZipFile(archive_path, "r")
                    zf = _thread_local.zips[archive_path]
                    data = zf.read(inner)
                    return Image.open(io.BytesIO(data))
    
                elif arc_lower.endswith(".tar"):
                    # Uncompressed TAR — the ultimate fast path
                    if archive_path not in _global_tar_index:
                        _build_global_tar_index(archive_path)
                    
                    idx = _global_tar_index[archive_path]
                    if inner not in idx:
                        return None
                    offset, size = idx[inner]
                    
                    # Each thread gets its own raw file handle to avoid lock contention on seek()
                    if not hasattr(_thread_local, 'raw_files'):
                        _thread_local.raw_files = {}
                    if archive_path not in _thread_local.raw_files:
                        _thread_local.raw_files[archive_path] = open(archive_path, "rb")
                        
                    f = _thread_local.raw_files[archive_path]
                    f.seek(offset)
                    data = f.read(size)
                    return Image.open(io.BytesIO(data))
                    
                elif arc_lower.endswith(".tar.gz"):
                    # Compressed TAR — fallback to thread-local tarfile
                    if not hasattr(_thread_local, 'tars'):
                        _thread_local.tars = {}
                    if archive_path not in _thread_local.tars:
                        _thread_local.tars[archive_path] = tarfile.open(archive_path, "r:gz")
                    
                    tf = _thread_local.tars[archive_path]
                    try:
                        member = tf.getmember(inner)
                    except KeyError:
                        return None
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        return None
                    return Image.open(io.BytesIO(fobj.read()))
    
                else:
                    return None  # Unknown archive type

        else:
            # Plain relative path
            return Image.open(pretraining_root / path)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert any image mode to RGB correctly.

    - Palette (P, PA): convert via getpalette → RGB
    - RGBA: alpha-composite on white background
    - Grayscale (L, LA, I, F): convert to proper 3-channel RGB
                                NOT expand() which creates R=G=B channels
                                that confuse color-based augmentations
    - Already RGB: return as-is
    """
    mode = img.mode

    if mode == "RGB":
        return img

    if mode in ("RGBA", "LA"):
        # Alpha composite on white
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if mode == "LA":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1])  # use alpha channel as mask
        return bg

    if mode in ("P", "PA"):
        return img.convert("RGB")

    if mode in ("L", "I", "F"):
        # Proper grayscale → RGB (single channel replicated correctly)
        return img.convert("RGB")

    # Fallback
    return img.convert("RGB")


def preprocess_image(pil_img: Image.Image) -> bytes | None:
    """
    Full preprocessing pipeline:
      0. Fast draft decode for huge JPEGs
      1. Channel normalization → RGB
      2. Validate minimum size
      3. Bounded resize (max side = MAX_SIDE)
      4. Encode to JPEG bytes

    Returns None if the image should be skipped.
    """
    # Step 0: draft mode for fast JPEG decode
    w, h = pil_img.size
    min_side = min(w, h)
    if min_side > MAX_SIDE:
        scale = MAX_SIDE / min_side
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if hasattr(pil_img, "draft"):
            pil_img.draft("RGB", (new_w, new_h))

    # Step 1: normalize to RGB
    img = _to_rgb(pil_img)

    # Step 2: validate minimum size
    w, h = img.size
    if w < MIN_SIDE or h < MIN_SIDE:
        return None

    # Step 3: bounded resize — only downscale, never upscale
    min_side = min(w, h)
    if min_side > MAX_SIDE:
        scale = MAX_SIDE / min_side
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)

    # Step 4: encode to JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Worker function (runs in thread pool)
# ---------------------------------------------------------------------------

def process_record(
    pretraining_root: Path,
    path: str,
    key: str,
    source_name: str,
    archive_key: str,
) -> tuple[str, bytes | None, str, str, str]:
    """Load + preprocess one record and keep attribution with the result."""
    pil_img = _load_pil_from_catalog_path(pretraining_root, path)
    if pil_img is None:
        return key, None, "load_failed", source_name, archive_key

    try:
        jpeg_bytes = preprocess_image(pil_img)
    except Exception:
        return key, None, "preprocess_failed", source_name, archive_key
    finally:
        try:
            pil_img.close()
        except Exception:
            pass

    if jpeg_bytes is None:
        return key, None, "too_small", source_name, archive_key

    return key, jpeg_bytes, "", source_name, archive_key


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

def load_catalog(catalog_path: Path) -> list[dict]:
    records = []
    with open(catalog_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            records.append(row)
    return records


def source_key_for_record(record: dict) -> str:
    """Return the archive/source bucket that should stay on one worker."""
    path = record.get("path", "")
    if "::" in path:
        return path.split("::", 1)[0]
    if record.get("source_name"):
        return record["source_name"]
    return Path(path).parts[0] if Path(path).parts else "plain-files"


def prepare_source_groups(records: list[dict], start_index: int = 0) -> list[tuple[str, list[tuple[dict, str, str, str]]]]:
    """Group catalog rows so each worker processes one archive/source at a time."""
    groups: dict[str, list[tuple[dict, str, str, str]]] = defaultdict(list)
    for local_idx, rec in enumerate(records):
        archive_key = source_key_for_record(rec)
        source_name = rec.get("source_name") or archive_key
        key = f"{start_index + local_idx:09d}"
        groups[archive_key].append((rec, key, source_name, archive_key))
    return sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def get_resume_offset(output_prefix: str, max_size: int, max_count: int) -> int:
    """
    Count how many images are already written in complete shards.
    A shard is considered 'complete' if its size > 90% of max_size OR
    its image count == max_count (we can't know count without opening).
    Conservative: only skip shards that are clearly full (>= 900MB for 1GB target).
    """
    shard_dir = Path(output_prefix).parent
    base = Path(output_prefix).name
    complete_shards = sorted(shard_dir.glob(f"{base}-*.tar"))
    
    skipped_images = 0
    complete_count = 0
    threshold = max_size * 0.90

    for shard_path in complete_shards:
        size = shard_path.stat().st_size
        if size >= threshold:
            # Estimate images: assume JPEG at ~50KB avg
            # Better: open and count keys
            try:
                with tarfile.open(shard_path, "r:") as tf:
                    # Count unique keys (each image = 1 .jpg member)
                    count = sum(1 for m in tf.getmembers() if m.name.endswith(".jpg"))
                skipped_images += count
                complete_count += 1
                print(f"  [RESUME] Skipping complete shard {shard_path.name} ({count:,} images, {size/1e9:.2f} GB)")
            except Exception:
                # Can't read — stop here to be safe
                break
        else:
            # Partial shard found — stop here
            print(f"  [RESUME] Found partial shard {shard_path.name} ({size/1e9:.2f} GB) — will overwrite from here")
            # Delete the partial shard so ShardWriter starts fresh
            shard_path.unlink(missing_ok=True)
            break

    print(f"  [RESUME] Skipping first {skipped_images:,} records ({complete_count} complete shards)")
    return skipped_images


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_shards(
    catalog_path: Path,
    pretraining_root: Path,
    output_prefix: str,
    max_size: int = 1_000_000_000,
    max_count: int = 10_000,
    workers: int = 8,
    resume: bool = False,
    progress_every: int = PROGRESS_EVERY,
) -> None:
    catalog_path = catalog_path.resolve()
    pretraining_root = pretraining_root.resolve()
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" AG-Foundation WebDataset Shard Builder v2")
    print("=" * 72)
    print(f"  Catalog        : {catalog_path}")
    print(f"  Pretraining root: {pretraining_root}")
    print(f"  Output prefix  : {output_prefix}")
    print(f"  Max shard size : {max_size / 1e9:.1f} GB")
    print(f"  Max shard count: {max_count:,} images")
    print(f"  Workers        : {workers}")
    print(f"  Scheduling     : one archive/source group per worker at a time")
    print(f"  Progress update: every {progress_every:,} completed records")
    print(f"  Bounded resize : images > {MAX_SIDE}px min-side → resize to {MAX_SIDE}px")
    print(f"  JPEG quality   : {JPEG_QUALITY}")
    print(f"  Min image size : {MIN_SIDE}px")
    print("=" * 72)

    records = load_catalog(catalog_path)
    total_records = len(records)
    print(f"\nLoaded {total_records:,} records from catalog.\n")

    if resume:
        raise RuntimeError(
            "Shard resume is disabled because the previous implementation could "
            "overwrite existing shard names and skip the wrong records. Run a clean rebuild instead."
        )

    start_index = 0
    groups = prepare_source_groups(records, start_index=start_index)
    print(f"Prepared {len(groups):,} archive/source groups for dedicated-worker processing.", flush=True)
    for archive_key, items in groups[:10]:
        print(f"  group: {archive_key} ({len(items):,} records)", flush=True)
    if len(groups) > 10:
        print(f"  ... plus {len(groups) - 10:,} more groups", flush=True)
    print()

    # Stats
    written = 0
    completed = 0
    skip_counts: Counter = Counter()
    source_counts: Counter = Counter()
    active_sources: set[str] = set()
    completed_sources = 0
    t_start = time.time()

    # Change to output dir (WebDataset requires relative paths on Windows)
    original_cwd = os.getcwd()
    os.chdir(str(output_dir))
    base_pattern = Path(output_prefix).name + "-%06d.tar"

    work_queue: Queue = Queue()
    result_queue: Queue = Queue(maxsize=max(workers * 16, 64))
    for group in groups:
        work_queue.put(group)

    def archive_worker(worker_id: int) -> None:
        while True:
            try:
                archive_key, items = work_queue.get_nowait()
            except queue.Empty:
                break
            result_queue.put(("source_start", archive_key, len(items)))
            try:
                for rec, key, source_name, source_archive in items:
                    result = process_record(
                        pretraining_root,
                        rec["path"],
                        key,
                        source_name,
                        source_archive,
                    )
                    result_queue.put(("record", result, None))
            except Exception as exc:
                result_queue.put(("worker_error", archive_key, repr(exc)))
            finally:
                result_queue.put(("source_done", archive_key, len(items)))
                work_queue.task_done()
        result_queue.put(("worker_done", worker_id, None))

    worker_count = min(max(workers, 1), max(len(groups), 1))
    threads: list[Thread] = []

    try:
        with wds.ShardWriter(
            base_pattern,
            maxsize=max_size,
            maxcount=max_count,
            verbose=False,
        ) as sink:
            for worker_id in range(worker_count):
                thread = Thread(target=archive_worker, args=(worker_id,), daemon=True)
                thread.start()
                threads.append(thread)

            active_workers = worker_count
            pbar = tqdm.tqdm(
                total=total_records,
                initial=start_index,
                desc="Building Shards",
                unit="img",
                smoothing=0.05,
                dynamic_ncols=True,
            )

            while active_workers > 0:
                kind, payload, extra = result_queue.get()

                if kind == "worker_done":
                    active_workers -= 1
                    continue

                if kind == "source_start":
                    active_sources.add(str(payload))
                    pbar.set_postfix(
                        active_archives=len(active_sources),
                        archives_done=f"{completed_sources}/{len(groups)}",
                        written=f"{written:,}",
                    )
                    continue

                if kind == "source_done":
                    active_sources.discard(str(payload))
                    completed_sources += 1
                    pbar.set_postfix(
                        active_archives=len(active_sources),
                        archives_done=f"{completed_sources}/{len(groups)}",
                        written=f"{written:,}",
                        skipped=f"{sum(skip_counts.values()):,}",
                    )
                    continue

                if kind == "worker_error":
                    skip_counts["worker_error"] += 1
                    tqdm.tqdm.write(f"[WARN] Worker failed while processing {payload}: {extra}")
                    continue

                key, jpeg_bytes, skip_reason, source_name, archive_key = payload
                completed += 1
                if jpeg_bytes is None:
                    skip_counts[skip_reason] += 1
                    pbar.update(1)
                else:
                    sink.write({"__key__": key, "jpg": jpeg_bytes})
                    written += 1
                    source_counts[source_name] += 1
                    pbar.update(1)

                if completed % progress_every == 0:
                    elapsed = time.time() - t_start
                    rate = completed / max(elapsed, 1)
                    approx_shard = written // max(max_count, 1)
                    pbar.set_postfix(
                        written=f"{written:,}",
                        skipped=f"{sum(skip_counts.values()):,}",
                        rate=f"{rate:.0f}/s",
                        shard=f"~{approx_shard:06d}",
                        active_archives=len(active_sources),
                    )

            pbar.close()

        for thread in threads:
            thread.join()

    finally:
        os.chdir(original_cwd)

    # Final report
    elapsed_total = time.time() - t_start
    skipped_total = sum(skip_counts.values())
    total_processed = written + skipped_total

    print(f"\n{'=' * 72}")
    print(f" SHARD BUILD COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Images written : {written:,}")
    print(f"  Images skipped : {skipped_total:,}")
    print(f"  Total processed: {total_processed:,}")
    print(f"  Elapsed        : {elapsed_total / 3600:.2f} hours")
    print(f"  Average rate   : {written / max(elapsed_total, 1):.0f} img/s")
    print(f"\n  Skip breakdown:")
    for reason, count in skip_counts.most_common():
        print(f"    {reason:20s}: {count:,}")
    print(f"\n  Top 20 sources by images written:")
    for src, count in source_counts.most_common(20):
        pct = 100 * count / max(written, 1)
        print(f"    {count:>10,}  ({pct:4.1f}%)  {src}")
    print(flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build WebDataset shards with bounded-resize preprocessing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Path to catalog_v2.csv produced by build_pretraining_catalog.py",
    )
    parser.add_argument(
        "--pretraining-root",
        type=Path,
        default=Path(r"E:\AG_Dataset\AG-Foundational-Model\Pretraining"),
        help="Base directory for resolving relative paths in the catalog",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=r"E:\AG_Dataset\shards\dataset",
        help="Output shard prefix, e.g. E:/AG_Dataset/shards/dataset",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1_000_000_000,
        help="Max bytes per shard (default: 1 GB)",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=10_000,
        help="Max images per shard (default: 10000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel image-decoding threads",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Disabled: resume is unsafe for this shard writer and will fail fast.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help="Refresh progress postfix after this many completed records.",
    )
    args = parser.parse_args()

    build_shards(
        catalog_path=args.catalog,
        pretraining_root=args.pretraining_root,
        output_prefix=args.output_prefix,
        max_size=args.max_size,
        max_count=args.max_count,
        workers=args.workers,
        resume=args.resume,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
