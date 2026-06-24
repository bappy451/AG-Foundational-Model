#!/usr/bin/env python3
"""Build a master pretraining catalog from all ZIP/directory sources.

Usage::

    python scripts/build_pretraining_catalog.py \\
        --pretraining-root ../Pretraining \\
        --output-path catalogs/pretraining_master.csv \\
        --exclude-sources "Toxic Plant Classification" "Edible wild plants" \\
            "Pea Plant dataset" "Indian Medicinal Plant Image Dataset" \\
            "Agriculture crop images" "Paddy Doctor- Paddy Disease Classification" \\
            "GeoPlant_ Spatial Plant Species Prediction Dataset-008" \\
            "Pumpkin Leaf Diseases Dataset From Bangladesh" \\
            "PlantSeg_ A Large-Scale In-the-wild Dataset for Plant Disease Segmentation" \\
            "corn-kernel-counting" "longitudinal-nutrient-deficiency"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from the scripts/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ag_foundation.data.dataset import AgricultureImageDataset  # noqa: E402
from ag_foundation.data.multi_source_dataset import (  # noqa: E402
    scan_pretraining_directory,
)


def build_catalog(
    pretraining_root: Path,
    output_path: Path,
    *,
    exclude_sources: set[str] | None = None,
    skip_known_duplicates: bool = True,
    crop_size: int = 1,
) -> None:
    """Scan all sources and write a master CSV catalog with source tracking."""
    import pandas as pd

    sources = scan_pretraining_directory(
        pretraining_root,
        exclude_sources=exclude_sources,
        skip_known_duplicates=skip_known_duplicates,
    )
    print(f"Discovered {len(sources)} sources under {pretraining_root}")

    all_rows: list[dict[str, str]] = []
    for source_path in sources:
        source_name = source_path.stem
        print(f"  Scanning: {source_path.name} ... ", end="", flush=True)
        t0 = time.time()

        try:
            dataset = AgricultureImageDataset(
                source_path, crop_size=crop_size, augment=False,
            )
            for record in dataset.records:
                # Make path portable: relative to pretraining root
                try:
                    rel = record.source_path.relative_to(pretraining_root.resolve())
                    source_text = rel.as_posix()
                except ValueError:
                    source_text = str(record.source_path)

                if record.archive_chain:
                    portable_path = "::".join((source_text, *record.archive_chain))
                else:
                    portable_path = source_text

                all_rows.append({
                    "path": portable_path,
                    "group": record.group,
                    "source_dataset": source_name,
                })
            dataset.close()
            elapsed = time.time() - t0
            print(f"{len(dataset)} images ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"SKIPPED ({exc})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_path, index=False)
    print(f"\nWrote {len(frame)} records to {output_path}")
    print(f"Sources: {frame['source_dataset'].nunique()}")
    print(f"Groups:  {frame['group'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pretraining catalog")
    parser.add_argument(
        "--pretraining-root", type=Path, required=True,
        help="Path to Pretraining directory",
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("catalogs/pretraining_master.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--exclude-sources", nargs="*", default=[],
        help="Source stems to exclude (held-out for evaluation)",
    )
    parser.add_argument(
        "--skip-duplicates", action="store_true", default=True,
        help="Skip known duplicate ZIP files",
    )
    args = parser.parse_args()
    build_catalog(
        args.pretraining_root,
        args.output_path,
        exclude_sources=set(args.exclude_sources),
        skip_known_duplicates=args.skip_duplicates,
    )


if __name__ == "__main__":
    main()
