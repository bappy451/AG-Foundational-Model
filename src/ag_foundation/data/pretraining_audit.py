from __future__ import annotations

import argparse
import csv
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".heic",
    ".heif",
    ".jpe",
    ".jpeg",
    ".jfif",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
ARRAY_EXTENSIONS = {".npy"}
DATA_CONTAINER_EXTENSIONS = {".h5", ".hdf5", ".npz"}
ANNOTATION_EXTENSIONS = {
    ".csv",
    ".dbf",
    ".geojson",
    ".json",
    ".kml",
    ".prj",
    ".shp",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
ARCHIVE_EXTENSIONS = {".zip"}
IGNORED_NAMES = {".DS_Store", "Thumbs.db"}
IGNORED_PREFIXES = ("__MACOSX/",)
URL_PATTERN = re.compile(r"https?://[^\s)]+")
WORD_PATTERN = re.compile(r"[a-z0-9]+")

GENERIC_LABEL_PARTS = {
    "archive",
    "augmented",
    "color",
    "content",
    "crop_images",
    "data",
    "dataset",
    "datasets",
    "files",
    "image",
    "images",
    "images_test",
    "images_train",
    "images_val",
    "input",
    "jpeg",
    "jpg",
    "leaf",
    "leaves",
    "original",
    "output",
    "plantnet_300k",
    "png",
    "segmented",
    "test",
    "testing",
    "train",
    "training",
    "val",
    "valid",
    "validation",
}

MATCH_STOPWORDS = {
    "and",
    "classification",
    "data",
    "dataset",
    "datasets",
    "detection",
    "for",
    "from",
    "image",
    "images",
    "published",
    "the",
    "version",
}

THEME_KEYWORDS = {
    "plant disease / stress": {
        "bacterial",
        "blight",
        "disease",
        "diseases",
        "downy",
        "fungal",
        "healthy",
        "mildew",
        "mosaic",
        "pathogen",
        "powdery",
        "rust",
        "smut",
        "spot",
        "stress",
        "virus",
    },
    "species / plant identification": {
        "edible",
        "house",
        "medicinal",
        "plantify",
        "plantnet",
        "species",
        "toxic",
        "type",
        "types",
        "wild",
    },
    "crop-specific agronomy": {
        "agriculture",
        "chili",
        "cotton",
        "crop",
        "ghana",
        "paddy",
        "pea",
        "pumpkin",
        "rice",
        "soybean",
        "wheat",
    },
    "seedling / early growth": {"seedling", "seedlings"},
    "weed / crop discrimination": {"broadleaf", "grass", "weed", "weeds"},
    "geospatial / remote sensing": {
        "geoplant",
        "geotiff",
        "landsat",
        "nir",
        "remote",
        "satellite",
        "sentinel",
        "uav",
    },
    "detection / annotation": {
        "annotation",
        "bbox",
        "bounding",
        "box",
        "detect",
        "detection",
        "label",
        "labels",
        "yolo",
    },
}


@dataclass(frozen=True)
class ManifestEntry:
    index: int
    name: str
    urls: tuple[str, ...]
    providers: tuple[str, ...]
    line: str


@dataclass(frozen=True)
class ImageSample:
    path: str
    extension: str
    width: int | None
    height: int | None
    mode: str | None
    bands: int | None
    dtype: str | None
    error: str | None = None


@dataclass(frozen=True)
class LocalDatasetAudit:
    name: str
    path: str
    kind: str
    size_bytes: int
    matched_manifest_name: str | None
    manifest_urls: tuple[str, ...]
    manifest_providers: tuple[str, ...]
    total_files: int
    image_count: int
    annotation_count: int
    nested_archive_count: int
    extension_counts: dict[str, int]
    annotation_extension_counts: dict[str, int]
    top_label_folders: list[dict[str, int | str]]
    inferred_label_folder_count: int
    likely_themes: list[str]
    likely_modalities: list[str]
    sample_images: list[ImageSample]
    warnings: list[str]


def parse_manifest(dataset_list: str | Path) -> list[ManifestEntry]:
    path = Path(dataset_list)
    entries: list[ManifestEntry] = []
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        urls = tuple(URL_PATTERN.findall(line))
        if not urls:
            continue
        first_url_start = URL_PATTERN.search(line)
        if first_url_start is None:
            continue
        prefix = line[: first_url_start.start()].strip(" :-")
        name = _clean_display_name(prefix) or f"dataset-line-{index}"
        entries.append(
            ManifestEntry(
                index=index,
                name=name,
                urls=urls,
                providers=_providers_from_urls(urls),
                line=line,
            )
        )
    return entries


def run_audit(
    pretraining_root: str | Path,
    *,
    dataset_list: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_limit: int = 24,
    inspect_samples: bool = True,
) -> dict[str, Any]:
    root = Path(pretraining_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Pretraining root does not exist: {root}")
    manifest_path = Path(dataset_list).expanduser().resolve() if dataset_list else root / "Dataset.txt"
    entries = parse_manifest(manifest_path) if manifest_path.exists() else []

    local_items = _discover_local_items(root)
    audits = [
        _audit_local_item(item, entries=entries, sample_limit=sample_limit, inspect_samples=inspect_samples)
        for item in local_items
    ]
    summary = _build_summary(root, manifest_path, entries, audits)

    if output_dir is not None:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        summary["outputs"] = _output_map(output_path)
        _write_json(output_path / "pretraining_dataset_audit.json", summary)
        _write_csv(output_path / "pretraining_dataset_audit.csv", audits)
        (output_path / "pretraining_dataset_audit.md").write_text(_render_markdown(summary), encoding="utf-8")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a local agricultural pretraining dataset folder. The scanner reads ZIP central "
            "directories without extracting archives and writes JSON, CSV, and Markdown reports."
        )
    )
    parser.add_argument(
        "--pretraining-root",
        default="../Pretraining",
        help="Folder containing downloaded archives/directories. Defaults to ../Pretraining from the project root.",
    )
    parser.add_argument(
        "--dataset-list",
        default=None,
        help="Dataset manifest text file. Defaults to <pretraining-root>/Dataset.txt when present.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/pretraining_dataset_audit",
        help="Directory for JSON, CSV, and Markdown reports.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=24,
        help="Maximum image files per local dataset to inspect for dimensions/modes/bands.",
    )
    parser.add_argument(
        "--no-sample-inspection",
        action="store_true",
        help="Only count files and extensions; do not open image samples.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_audit(
        args.pretraining_root,
        dataset_list=args.dataset_list,
        output_dir=args.output_dir,
        sample_limit=max(args.sample_limit, 0),
        inspect_samples=not args.no_sample_inspection,
    )
    aggregate = summary["aggregate"]
    outputs = summary["outputs"]
    print(f"Scanned local datasets: {aggregate['local_dataset_count']}")
    print(f"Counted image-like files: {aggregate['image_count']:,}")
    print(f"Manifest sources listed: {aggregate['manifest_entry_count']}")
    print(f"Manifest sources not found locally: {len(summary['missing_manifest_entries'])}")
    print(f"Wrote Markdown report: {outputs['markdown']}")
    print(f"Wrote JSON report: {outputs['json']}")
    print(f"Wrote CSV report: {outputs['csv']}")


def _discover_local_items(root: Path) -> list[Path]:
    items: list[Path] = []
    for item in sorted(root.iterdir(), key=lambda value: value.name.lower()):
        if item.name in IGNORED_NAMES or item.name.startswith("."):
            continue
        if item.is_dir() or item.suffix.lower() in ARCHIVE_EXTENSIONS:
            items.append(item)
    return items


def _audit_local_item(
    item: Path,
    *,
    entries: list[ManifestEntry],
    sample_limit: int,
    inspect_samples: bool,
) -> LocalDatasetAudit:
    match = _match_manifest_entry(item.name, entries)
    counters: Counter[str] = Counter()
    annotation_counters: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    sample_candidates: list[tuple[str, bytes | Path]] = []
    warnings: list[str] = []
    total_files = 0
    image_count = 0
    annotation_count = 0
    nested_archive_count = 0
    kind = "directory" if item.is_dir() else "zip"

    if item.is_dir():
        for file_path in sorted((path for path in item.rglob("*") if path.is_file()), key=lambda value: str(value)):
            if _is_ignored_path(file_path.relative_to(item).as_posix()):
                continue
            relative = file_path.relative_to(item).as_posix()
            ext = _normalized_suffix(relative)
            total_files += 1
            counters[ext] += 1
            if ext in IMAGE_EXTENSIONS or ext in ARRAY_EXTENSIONS:
                image_count += 1
                label = _infer_label_from_member(relative)
                if label:
                    labels[label] += 1
                if len(sample_candidates) < sample_limit:
                    sample_candidates.append((relative, file_path))
            elif ext in ANNOTATION_EXTENSIONS:
                annotation_count += 1
                annotation_counters[ext] += 1
            elif ext in ARCHIVE_EXTENSIONS:
                nested_archive_count += 1
            elif ext in DATA_CONTAINER_EXTENSIONS:
                warnings.append(f"Found {ext} container {relative}; internal arrays were not expanded.")
    else:
        try:
            with zipfile.ZipFile(item) as archive:
                for info in archive.infolist():
                    if info.is_dir() or _is_ignored_path(info.filename):
                        continue
                    ext = _normalized_suffix(info.filename)
                    total_files += 1
                    counters[ext] += 1
                    if ext in IMAGE_EXTENSIONS or ext in ARRAY_EXTENSIONS:
                        image_count += 1
                        label = _infer_label_from_member(info.filename)
                        if label:
                            labels[label] += 1
                        if inspect_samples and len(sample_candidates) < sample_limit:
                            try:
                                sample_candidates.append((info.filename, archive.read(info)))
                            except RuntimeError as exc:
                                warnings.append(f"Could not read sample {info.filename}: {exc}")
                    elif ext in ANNOTATION_EXTENSIONS:
                        annotation_count += 1
                        annotation_counters[ext] += 1
                    elif ext in ARCHIVE_EXTENSIONS:
                        nested_archive_count += 1
                    elif ext in DATA_CONTAINER_EXTENSIONS:
                        warnings.append(f"Found {ext} container {info.filename}; internal arrays were not expanded.")
        except zipfile.BadZipFile:
            warnings.append("Archive could not be opened as a ZIP file.")
        except OSError as exc:
            warnings.append(f"Archive scan failed: {exc}")

    samples = (
        [_inspect_sample(sample_path, payload) for sample_path, payload in sample_candidates]
        if inspect_samples and sample_limit > 0
        else []
    )
    if nested_archive_count:
        warnings.append(
            f"Found {nested_archive_count} nested archive(s). Image counts do not include files inside nested archives."
        )

    context_text = " ".join([item.stem, match.name if match else "", " ".join(labels.keys())])
    extension_counts = dict(sorted(counters.items(), key=lambda value: (-value[1], value[0])))
    annotation_extension_counts = dict(sorted(annotation_counters.items(), key=lambda value: (-value[1], value[0])))

    return LocalDatasetAudit(
        name=item.name,
        path=str(item),
        kind=kind,
        size_bytes=_path_size(item),
        matched_manifest_name=match.name if match else None,
        manifest_urls=match.urls if match else (),
        manifest_providers=match.providers if match else (),
        total_files=total_files,
        image_count=image_count,
        annotation_count=annotation_count,
        nested_archive_count=nested_archive_count,
        extension_counts=extension_counts,
        annotation_extension_counts=annotation_extension_counts,
        top_label_folders=[
            {"label": label, "count": count} for label, count in labels.most_common(15)
        ],
        inferred_label_folder_count=len(labels),
        likely_themes=_infer_themes(context_text),
        likely_modalities=_infer_modalities(extension_counts, samples, context_text),
        sample_images=samples,
        warnings=warnings,
    )


def _build_summary(
    root: Path,
    manifest_path: Path,
    entries: list[ManifestEntry],
    audits: list[LocalDatasetAudit],
) -> dict[str, Any]:
    matched_names = {audit.matched_manifest_name for audit in audits if audit.matched_manifest_name}
    missing_entries = [entry for entry in entries if entry.name not in matched_names]
    extension_counts: Counter[str] = Counter()
    annotation_counts: Counter[str] = Counter()
    theme_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    for audit in audits:
        extension_counts.update(audit.extension_counts)
        annotation_counts.update(audit.annotation_extension_counts)
        theme_counts.update(audit.likely_themes)
        modality_counts.update(audit.likely_modalities)

    total_images = sum(audit.image_count for audit in audits)
    total_annotations = sum(audit.annotation_count for audit in audits)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pretraining_root": str(root),
        "dataset_list": str(manifest_path) if manifest_path else None,
        "outputs": {
            "json": "",
            "csv": "",
            "markdown": "",
        },
        "aggregate": {
            "local_dataset_count": len(audits),
            "manifest_entry_count": len(entries),
            "matched_manifest_entry_count": len(matched_names),
            "unmatched_local_dataset_count": len([audit for audit in audits if not audit.matched_manifest_name]),
            "total_files": sum(audit.total_files for audit in audits),
            "image_count": total_images,
            "annotation_count": total_annotations,
            "nested_archive_count": sum(audit.nested_archive_count for audit in audits),
            "size_bytes": sum(audit.size_bytes for audit in audits),
            "extension_counts": dict(sorted(extension_counts.items(), key=lambda value: (-value[1], value[0]))),
            "annotation_extension_counts": dict(
                sorted(annotation_counts.items(), key=lambda value: (-value[1], value[0]))
            ),
            "theme_counts": dict(sorted(theme_counts.items(), key=lambda value: (-value[1], value[0]))),
            "modality_counts": dict(sorted(modality_counts.items(), key=lambda value: (-value[1], value[0]))),
        },
        "local_datasets": [asdict(audit) for audit in audits],
        "missing_manifest_entries": [asdict(entry) for entry in missing_entries],
        "unmatched_local_datasets": [asdict(audit) for audit in audits if not audit.matched_manifest_name],
    }
    summary["recommendations"] = _build_recommendations(summary)
    return summary


def _write_json(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, audits: list[LocalDatasetAudit]) -> None:
    fieldnames = [
        "name",
        "kind",
        "size_gb",
        "matched_manifest_name",
        "image_count",
        "annotation_count",
        "inferred_label_folder_count",
        "top_extensions",
        "likely_themes",
        "likely_modalities",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for audit in audits:
            writer.writerow(
                {
                    "name": audit.name,
                    "kind": audit.kind,
                    "size_gb": f"{audit.size_bytes / (1024**3):.3f}",
                    "matched_manifest_name": audit.matched_manifest_name or "",
                    "image_count": audit.image_count,
                    "annotation_count": audit.annotation_count,
                    "inferred_label_folder_count": audit.inferred_label_folder_count,
                    "top_extensions": _format_counter(audit.extension_counts, limit=5),
                    "likely_themes": "; ".join(audit.likely_themes),
                    "likely_modalities": "; ".join(audit.likely_modalities),
                    "warnings": "; ".join(audit.warnings),
                }
            )


def _render_markdown(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Pretraining Dataset Audit",
        "",
        f"Generated at: `{summary['generated_at_utc']}`",
        "",
        f"Pretraining root: `{summary['pretraining_root']}`",
        "",
        "## Executive Summary",
        "",
        f"- Local datasets scanned: **{aggregate['local_dataset_count']}**",
        f"- Image-like files counted: **{aggregate['image_count']:,}**",
        f"- Files counted across archives/directories: **{aggregate['total_files']:,}**",
        f"- Annotation/metadata files counted: **{aggregate['annotation_count']:,}**",
        f"- Manifest sources listed: **{aggregate['manifest_entry_count']}**",
        f"- Manifest sources matched locally: **{aggregate['matched_manifest_entry_count']}**",
        f"- Manifest sources not found locally: **{len(summary['missing_manifest_entries'])}**",
        f"- Local storage scanned: **{aggregate['size_bytes'] / (1024**3):.2f} GB**",
        "",
        "## Format Mix",
        "",
        _markdown_counter_table(aggregate["extension_counts"], ("Extension", "Count"), limit=15),
        "",
        "## Dataset Inventory",
        "",
        (
            "| Local dataset | Matched manifest source | Images | Labels/classes inferred "
            "| Type summary | Modality summary |"
        ),
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for audit in summary["local_datasets"]:
        lines.append(
            "| {name} | {match} | {images} | {labels} | {themes} | {modalities} |".format(
                name=_escape_md(audit["name"]),
                match=_escape_md(audit["matched_manifest_name"] or "not listed / not matched"),
                images=f"{audit['image_count']:,}",
                labels=audit["inferred_label_folder_count"],
                themes=_escape_md(", ".join(audit["likely_themes"]) or "unknown"),
                modalities=_escape_md(", ".join(audit["likely_modalities"]) or "unknown"),
            )
        )

    lines.extend(
        [
            "",
            "## What Is Present",
            "",
            _markdown_counter_table(aggregate["theme_counts"], ("Likely image/task type", "Dataset count"), limit=20),
            "",
            "## What Is Missing Or Underrepresented",
            "",
        ]
    )
    for gap in _gap_findings(summary):
        lines.append(f"- {gap}")

    lines.extend(
        [
            "",
            "## Recommended Additions",
            "",
        ]
    )
    for recommendation in summary["recommendations"]:
        lines.append(f"- {recommendation}")

    lines.extend(
        [
            "",
            "## Listed Sources Not Found Locally",
            "",
        ]
    )
    if summary["missing_manifest_entries"]:
        for entry in summary["missing_manifest_entries"]:
            urls = ", ".join(entry["urls"])
            lines.append(f"- {entry['name']}: {urls}")
    else:
        lines.append("- Every listed source had a local match.")

    lines.extend(
        [
            "",
            "## Local Datasets Not Matched To The Manifest",
            "",
        ]
    )
    if summary["unmatched_local_datasets"]:
        for audit in summary["unmatched_local_datasets"]:
            lines.append(f"- {audit['name']} ({audit['image_count']:,} image-like files)")
    else:
        lines.append("- Every local dataset matched a manifest entry.")

    lines.extend(
        [
            "",
            "## Re-run Command",
            "",
            "```bash",
            "python scripts/analyze_pretraining_dataset.py \\",
            "  --pretraining-root ../Pretraining \\",
            "  --dataset-list ../Pretraining/Dataset.txt \\",
            "  --output-dir reports/pretraining_dataset_audit",
            "```",
            "",
            (
                "The scanner does not extract archives. Image counts are based on file extensions, while dimensions, "
                "modes, "
            ),
            "and band counts are sampled from a small number of files per dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_recommendations(summary: dict[str, Any]) -> list[str]:
    aggregate = summary["aggregate"]
    ext_counts = aggregate["extension_counts"]
    theme_counts = aggregate["theme_counts"]
    annotation_count = aggregate["annotation_count"]
    recommendations = [
        (
            "Add true multispectral and GeoTIFF pretraining data, ideally Sentinel-2/Landsat/UAV tiles with NIR "
            "and red-edge bands, because the current corpus is dominated by standard RGB formats."
        ),
        (
            "Add georeferenced field-scale imagery with location, date, crop stage, climate, soil, and management "
            "metadata so the foundation model learns agronomic context instead of only leaf appearance."
        ),
        (
            "Add segmentation masks and object-detection labels for weeds, crops, pests, disease lesions, canopy "
            "rows, and field boundaries to support dense downstream tasks."
        ),
        (
            "Add external held-out benchmark datasets grouped by geography, crop, farm, and season to measure "
            "cross-domain generalization for CVPR or Computers and Electronics in Agriculture quality."
        ),
        (
            "Run duplicate and near-duplicate detection before pretraining, especially for augmented disease "
            "datasets, so SSL does not overfit repeated transformations."
        ),
        (
            "Track license, citation, source URL, and download date per dataset before publication; this is "
            "essential for a top-tier reproducible dataset release."
        ),
    ]
    if ext_counts.get(".tif", 0) + ext_counts.get(".tiff", 0) + ext_counts.get(".npy", 0) > 0:
        recommendations[0] = (
            "Expand the existing multispectral/GeoTIFF subset with more crops, sensors, seasons, and geographies; "
            "keep RGB-only and multiband splits separate for ablation."
        )
    if annotation_count > 0:
        recommendations[2] = (
            "Standardize the existing annotation files into a single schema and add segmentation/detection coverage "
            "for datasets that currently provide only class folders."
        )
    if "geospatial / remote sensing" in theme_counts:
        recommendations[1] = (
            "Broaden geospatial coverage beyond the currently listed sources by adding field-scale UAV and satellite "
            "tiles with explicit train/validation/test geography splits."
        )
    return recommendations


def _gap_findings(summary: dict[str, Any]) -> list[str]:
    aggregate = summary["aggregate"]
    ext_counts = aggregate["extension_counts"]
    theme_counts = aggregate["theme_counts"]
    gaps: list[str] = []
    if ext_counts.get(".tif", 0) + ext_counts.get(".tiff", 0) + ext_counts.get(".npy", 0) == 0:
        gaps.append(
            "No TIFF/GeoTIFF or NPY multispectral files were found locally, so the current downloaded corpus "
            "appears RGB-only."
        )
    else:
        gaps.append(
            "Multiband-capable formats are present, but they should be audited separately from RGB images for "
            "sensor, band, and georeferencing coverage."
        )
    if "geospatial / remote sensing" not in theme_counts:
        gaps.append(
            "No local dataset was clearly identified as geospatial, satellite, UAV, NIR, or remote-sensing imagery."
        )
    if aggregate["annotation_count"] == 0:
        gaps.append(
            "No annotation/metadata files were counted, which means detection, segmentation, and metadata tasks "
            "are underrepresented."
        )
    else:
        gaps.append(
            "Annotation/metadata files exist, but the corpus is still mostly class-folder style; convert labels "
            "into a shared schema before multi-task training."
        )
    if len(summary["missing_manifest_entries"]) > 0:
        gaps.append(
            "Several sources listed in the manifest are not downloaded locally yet, so the corpus is incomplete "
            "relative to the current plan."
        )
    if "plant disease / stress" in theme_counts and "species / plant identification" in theme_counts:
        gaps.append(
            "The corpus has useful disease and species coverage, but it is likely biased toward leaf close-ups "
            "and classification labels."
        )
    return gaps


def _inspect_sample(path: str, payload: bytes | Path) -> ImageSample:
    ext = _normalized_suffix(path)
    if ext == ".npy":
        return _inspect_npy_sample(path, payload)
    try:
        if isinstance(payload, Path):
            with Image.open(payload) as image:
                return _sample_from_image(path, ext, image)
        with Image.open(io.BytesIO(payload)) as image:
            return _sample_from_image(path, ext, image)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return ImageSample(
            path=path,
            extension=ext,
            width=None,
            height=None,
            mode=None,
            bands=None,
            dtype=None,
            error=str(exc),
        )


def _inspect_npy_sample(path: str, payload: bytes | Path) -> ImageSample:
    try:
        import numpy as np

        array = np.load(
            payload if isinstance(payload, Path) else io.BytesIO(payload),
            mmap_mode="r" if isinstance(payload, Path) else None,
        )
        shape = tuple(int(dim) for dim in array.shape)
        bands = _infer_array_bands(shape)
        height, width = _infer_array_hw(shape)
        return ImageSample(
            path=path,
            extension=".npy",
            width=width,
            height=height,
            mode=f"array{shape}",
            bands=bands,
            dtype=str(array.dtype),
        )
    except Exception as exc:  # noqa: BLE001 - sample inspection should never fail the whole audit.
        return ImageSample(
            path=path,
            extension=".npy",
            width=None,
            height=None,
            mode=None,
            bands=None,
            dtype=None,
            error=str(exc),
        )


def _sample_from_image(path: str, ext: str, image: Image.Image) -> ImageSample:
    return ImageSample(
        path=path,
        extension=ext,
        width=image.size[0],
        height=image.size[1],
        mode=image.mode,
        bands=len(image.getbands()),
        dtype=None,
    )


def _infer_array_bands(shape: tuple[int, ...]) -> int | None:
    if len(shape) == 2:
        return 1
    if len(shape) != 3:
        return None
    if shape[0] <= 32:
        return shape[0]
    if shape[-1] <= 32:
        return shape[-1]
    return None


def _infer_array_hw(shape: tuple[int, ...]) -> tuple[int | None, int | None]:
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) != 3:
        return None, None
    if shape[0] <= 32:
        return shape[1], shape[2]
    return shape[0], shape[1]


def _infer_label_from_member(member_name: str) -> str | None:
    parts = [part for part in PurePosixPath(member_name).parts[:-1] if part not in {"", ".", "/"}]
    for part in reversed(parts):
        normalized = "_".join(_tokens(part, stopwords=set()))
        if not normalized or normalized in GENERIC_LABEL_PARTS:
            continue
        return part
    return None


def _infer_themes(text: str) -> list[str]:
    tokens = set(_tokens(text, stopwords=set()))
    themes = [theme for theme, keywords in THEME_KEYWORDS.items() if tokens & keywords]
    return themes or ["general plant imagery"]


def _infer_modalities(extension_counts: dict[str, int], samples: list[ImageSample], context_text: str) -> list[str]:
    modalities: list[str] = []
    if any(extension_counts.get(ext, 0) for ext in IMAGE_EXTENSIONS - {".tif", ".tiff"}):
        modalities.append("standard RGB/image files")
    if extension_counts.get(".tif", 0) or extension_counts.get(".tiff", 0):
        modalities.append("TIFF/GeoTIFF candidate")
    if extension_counts.get(".npy", 0):
        modalities.append("NumPy array candidate")
    if any(extension_counts.get(ext, 0) for ext in DATA_CONTAINER_EXTENSIONS):
        modalities.append("HDF5/NPZ container candidate (not expanded)")
    if any(sample.bands and sample.bands > 4 for sample in samples):
        modalities.append("sampled multiband imagery")
    elif any(sample.bands == 4 for sample in samples):
        modalities.append("sampled RGBA/four-band imagery")
    elif any(sample.bands == 3 or sample.mode == "RGB" for sample in samples):
        modalities.append("sampled RGB imagery")
    elif any(sample.bands == 1 or sample.mode == "L" for sample in samples):
        modalities.append("sampled grayscale imagery")
    if set(_tokens(context_text, stopwords=set())) & THEME_KEYWORDS["geospatial / remote sensing"]:
        modalities.append("geospatial source indicated by name/labels")
    return _dedupe(modalities) or ["unknown modality"]


def _match_manifest_entry(local_name: str, entries: list[ManifestEntry]) -> ManifestEntry | None:
    if not entries:
        return None
    local_tokens = set(_tokens(Path(local_name).stem, stopwords=MATCH_STOPWORDS))
    if not local_tokens:
        return None
    best_entry: ManifestEntry | None = None
    best_score = 0.0
    local_joined = " ".join(sorted(local_tokens))
    for entry in entries:
        entry_tokens = set(_tokens(entry.name, stopwords=MATCH_STOPWORDS))
        if not entry_tokens:
            continue
        score = len(local_tokens & entry_tokens) / len(local_tokens | entry_tokens)
        entry_joined = " ".join(sorted(entry_tokens))
        if local_joined in entry_joined or entry_joined in local_joined:
            score += 0.25
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry if best_score >= 0.45 else None


def _tokens(text: str, *, stopwords: set[str]) -> list[str]:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return [word for word in WORD_PATTERN.findall(ascii_text.lower()) if word and word not in stopwords]


def _providers_from_urls(urls: Iterable[str]) -> tuple[str, ...]:
    providers: set[str] = set()
    for url in urls:
        if "kaggle.com" in url:
            providers.add("Kaggle")
        elif "mendeley.com" in url:
            providers.add("Mendeley Data")
        elif "zenodo.org" in url:
            providers.add("Zenodo")
        elif "github.com" in url:
            providers.add("GitHub")
        else:
            providers.add("Other")
    return tuple(sorted(providers))


def _clean_display_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip(" :-")
    return ascii_name


def _normalized_suffix(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    return suffix or "[no extension]"


def _is_ignored_path(path: str) -> bool:
    if PurePosixPath(path).name in IGNORED_NAMES:
        return True
    return any(path.startswith(prefix) for prefix in IGNORED_PREFIXES)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _format_counter(counter: dict[str, int], *, limit: int) -> str:
    return ", ".join(f"{key}:{value}" for key, value in list(counter.items())[:limit])


def _markdown_counter_table(counter: dict[str, int], headers: tuple[str, str], *, limit: int) -> str:
    if not counter:
        return "_None found._"
    rows = [f"| {headers[0]} | {headers[1]} |", "| --- | ---: |"]
    for key, value in list(counter.items())[:limit]:
        rows.append(f"| {_escape_md(str(key))} | {value:,} |")
    return "\n".join(rows)


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|")


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _output_map(output_dir: Path) -> dict[str, str]:
    return {
        "json": str(output_dir / "pretraining_dataset_audit.json"),
        "csv": str(output_dir / "pretraining_dataset_audit.csv"),
        "markdown": str(output_dir / "pretraining_dataset_audit.md"),
    }


if __name__ == "__main__":
    main()
