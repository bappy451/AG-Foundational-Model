"""Multi-source pretraining dataset for loading from many ZIP archives and directories.

This module composes the existing ``AgricultureImageDataset`` to support
pretraining across dozens of heterogeneous data sources simultaneously.
It provides source-level weighting, held-out exclusion, and provenance
tracking while reusing all existing image-loading infrastructure.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset import (
    AgricultureImageDataset,
    ImageRecord,
    _seed_worker,
    _split_records_by_group,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInfo:
    """Metadata about a single data source (ZIP or directory)."""

    source_id: str
    local_path: Path
    record_count: int
    weight: float = 1.0


@dataclass
class MultiSourceSummary:
    """Summary statistics for a multi-source dataset build."""

    total_records: int = 0
    source_count: int = 0
    excluded_count: int = 0
    duplicate_skipped: int = 0
    sources: list[SourceInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Duplicate detection helper
# ---------------------------------------------------------------------------

_KNOWN_DUPLICATE_SUFFIXES: set[str] = {
    # These are known byte-identical duplicates in the Pretraining folder.
    # We skip the copy without the numeric suffix.
    "Plant Disease Expert.zip",
    "Plant Leaves for Image Classification.zip",
}


def _is_known_duplicate(filename: str) -> bool:
    """Return True if *filename* is a known duplicate that should be skipped."""
    return filename in _KNOWN_DUPLICATE_SUFFIXES


# ---------------------------------------------------------------------------
# Core multi-source dataset
# ---------------------------------------------------------------------------


class MultiSourcePretrainingDataset(Dataset[dict[str, Any]]):
    """Dataset that unifies records from multiple imagery roots.

    Parameters
    ----------
    source_roots : list of str or Path
        Each entry is a path to a directory or ZIP file containing imagery.
    crop_size : int
        Target crop size in pixels (square).
    channels : int or None
        Expected channel count (``None`` for auto-detect from first image).
    precision : str
        One of ``'fp32'``, ``'fp16'``, ``'bf16'``.
    augment : bool
        Whether to apply random augmentation.
    exclude_sources : set of str or None
        Set of source base-names (stem of ZIP or directory name) to exclude.
        Used to hold out entire datasets for evaluation.
    skip_known_duplicates : bool
        If True, automatically skip known duplicate ZIP files.
    catalog_path : str or Path or None
        If given, load from this pre-built catalog instead of scanning roots.
    """

    def __init__(
        self,
        source_roots: Sequence[str | Path],
        *,
        crop_size: int = 224,
        channels: int | None = None,
        precision: str = "fp32",
        augment: bool = True,
        exclude_sources: set[str] | None = None,
        skip_known_duplicates: bool = True,
        catalog_path: str | Path | None = None,
    ) -> None:
        if not source_roots:
            raise ValueError("source_roots must contain at least one path.")
        self.crop_size = crop_size
        self.channels = channels
        self.precision = precision
        self.augment = augment
        self.exclude_sources = exclude_sources or set()
        self.skip_known_duplicates = skip_known_duplicates

        if catalog_path is not None:
            self._records, self._source_map, self.summary = self._load_from_catalog(
                Path(catalog_path), crop_size=crop_size, channels=channels,
                precision=precision, augment=augment,
            )
        else:
            self._records, self._source_map, self.summary = self._build_from_roots(
                source_roots, crop_size=crop_size, channels=channels,
                precision=precision, augment=augment,
            )

        # Build per-source datasets for __getitem__ delegation
        self._datasets: dict[str, AgricultureImageDataset] = {}
        self._record_to_source: list[str] = []
        self._record_to_local_index: list[int] = []
        self._build_index()

    @property
    def records(self) -> list[ImageRecord]:
        return self._records

    def _build_from_roots(
        self,
        source_roots: Sequence[str | Path],
        *,
        crop_size: int,
        channels: int | None,
        precision: str,
        augment: bool,
    ) -> tuple[list[ImageRecord], dict[str, list[int]], MultiSourceSummary]:
        all_records: list[ImageRecord] = []
        source_map: dict[str, list[int]] = {}
        summary = MultiSourceSummary()

        for root_path in source_roots:
            root = Path(root_path).expanduser().resolve()
            if not root.exists():
                continue

            source_name = root.stem
            # Check exclusions
            if source_name in self.exclude_sources:
                summary.excluded_count += 1
                continue
            # Check duplicates
            if self.skip_known_duplicates and _is_known_duplicate(root.name):
                summary.duplicate_skipped += 1
                continue

            try:
                dataset = AgricultureImageDataset(
                    root,
                    crop_size=crop_size,
                    channels=channels,
                    precision=precision,
                    augment=augment,
                )
            except (ValueError, FileNotFoundError):
                # Skip sources with no supported images or other issues
                continue

                continue
            indices = []
            for record in dataset.records:
                idx = len(all_records)
                all_records.append(record)
                indices.append(idx)
            source_map[source_name] = indices
            dataset.close()

            summary.sources.append(
                SourceInfo(
                    source_id=source_name,
                    local_path=root,
                    record_count=len(dataset.records),
                )
            )
            summary.source_count += 1

        summary.total_records = len(all_records)
        return all_records, source_map, summary

    def _load_from_catalog(
        self,
        catalog_path: Path,
        *,
        crop_size: int,
        channels: int | None,
        precision: str,
        augment: bool,
    ) -> tuple[list[ImageRecord], dict[str, list[int]], MultiSourceSummary]:
        import pandas as pd

        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")
        frame = pd.read_csv(catalog_path)
        required = {"path", "group", "source_dataset"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Catalog must contain columns: {required}")

        all_records: list[ImageRecord] = []
        source_map: dict[str, list[int]] = {}
        summary = MultiSourceSummary()

        for _, row in frame.iterrows():
            source_name = str(row["source_dataset"])
            if source_name in self.exclude_sources:
                continue

            record = ImageRecord(
                uri=str(row["path"]),
                group=str(row["group"]),
                source_path=Path(str(row["path"]).split("::")[0]),
            )
            idx = len(all_records)
            all_records.append(record)
            source_map.setdefault(source_name, []).append(idx)

        for source_name, indices in source_map.items():
            summary.sources.append(
                SourceInfo(
                    source_id=source_name,
                    local_path=Path("."),
                    record_count=len(indices),
                )
            )
            summary.source_count += 1

        summary.total_records = len(all_records)
        return all_records, source_map, summary

    def _build_index(self) -> None:
        """Build per-record source tracking for efficient __getitem__."""
        self._record_to_source = []
        self._record_to_local_index = []
        for source_name, indices in self._source_map.items():
            for local_idx, global_idx in enumerate(indices):
                # We store mappings at the global index position
                pass

        # Simpler: store source for each record
        source_for_record: dict[int, str] = {}
        for source_name, indices in self._source_map.items():
            for idx in indices:
                source_for_record[idx] = source_name

        self._record_to_source = [
            source_for_record.get(i, "unknown") for i in range(len(self._records))
        ]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        source_name = self._record_to_source[index]

        # Lazy-load dataset for this source's root if not cached
        if source_name not in self._datasets:
            source_info = next(
                (s for s in self.summary.sources if s.source_id == source_name), None
            )
            if source_info is not None:
                try:
                    ds = AgricultureImageDataset(
                        source_info.local_path,
                        crop_size=self.crop_size,
                        channels=self.channels,
                        precision=self.precision,
                        augment=self.augment,
                    )
                    self._datasets[source_name] = ds
                except (ValueError, FileNotFoundError):
                    raise

        # Use the underlying dataset's loading infrastructure
        ds = self._datasets.get(source_name)
        if ds is not None:
            # Find the record in the underlying dataset and load it
            image = ds._load_image(record)
            image = ds._load_image(record)

            _, height, width = image.shape
            if height < self.crop_size or width < self.crop_size:
                raise ValueError(
                    f"crop_size={self.crop_size} exceeds image dimensions "
                    f"{height}x{width} for '{record.uri}'."
                )

            from .dataset import PRECISION_DTYPES, _color_jitter, _random_bool

            if self.augment:
                image = ds._random_crop(image)
                if _random_bool():
                    image = torch.flip(image, dims=(2,))
                if _random_bool():
                    image = torch.flip(image, dims=(1,))
                image = _color_jitter(image)
            else:
                image = ds._center_crop(image)

            return {
                "image": image.to(dtype=PRECISION_DTYPES[self.precision]),
                "path": record.uri,
                "group": record.group,
                "source_dataset": source_name,
            }

        raise RuntimeError(f"No dataset available for source '{source_name}'")

    def close(self) -> None:
        """Close all cached dataset handles."""
        for ds in self._datasets.values():
            ds.close()
        self._datasets.clear()

    def __del__(self) -> None:
        if hasattr(self, "_datasets"):
            self.close()

    def get_source_weights(self, strategy: str = "balanced") -> list[float]:
        """Compute per-sample weights for source-balanced sampling.

        Parameters
        ----------
        strategy : str
            - ``'balanced'``: each source contributes equally
              (weight inversely proportional to source size).
            - ``'sqrt'``: square-root rebalancing (moderate downweighting
              of large sources).
            - ``'uniform'``: all samples weighted equally (no rebalancing).
        """
        if strategy == "uniform":
            return [1.0] * len(self._records)

        source_sizes = {
            name: len(indices) for name, indices in self._source_map.items()
        }

        weights = []
        for i in range(len(self._records)):
            source = self._record_to_source[i]
            size = source_sizes.get(source, 1)
            if strategy == "balanced":
                weights.append(1.0 / size)
            elif strategy == "sqrt":
                weights.append(1.0 / math.sqrt(size))
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
        return weights


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def get_pretraining_dataloaders(
    source_roots: Sequence[str | Path],
    *,
    batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 27,
    crop_size: int = 224,
    channels: int | None = None,
    precision: str = "fp32",
    num_workers: int = 4,
    prefetch_factor: int = 2,
    exclude_sources: set[str] | None = None,
    skip_known_duplicates: bool = True,
    sampling_strategy: str = "sqrt",
    train_augment: bool = True,
    val_augment: bool = False,
) -> tuple[DataLoader, DataLoader, MultiSourceSummary]:
    """Create train and validation DataLoaders from multiple pretraining sources.

    Returns
    -------
    train_loader, val_loader, summary
        The two DataLoaders and a summary of what was loaded.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    # Discover all records across all sources
    catalog_dataset = MultiSourcePretrainingDataset(
        source_roots,
        crop_size=crop_size,
        channels=channels,
        precision=precision,
        augment=False,
        exclude_sources=exclude_sources,
        skip_known_duplicates=skip_known_duplicates,
    )
    summary = catalog_dataset.summary
    all_records = catalog_dataset.records
    source_map = catalog_dataset._source_map

    if len(all_records) == 0:
        raise ValueError("No images found across any source roots.")

    # Split by group (existing logic)
    train_records, val_records = _split_records_by_group(
        all_records, val_fraction=val_fraction, seed=seed,
    )

    catalog_dataset.close()

    # Build train and val datasets with correct source maps
    train_source_map: dict[str, list[int]] = {}
    for i, rec in enumerate(train_records):
        for source_name, indices in source_map.items():
            # Check if this record's uri matches any from this source
            source_records = [all_records[idx] for idx in indices]
            if any(sr.uri == rec.uri for sr in source_records):
                train_source_map.setdefault(source_name, []).append(i)
                break

    # Create separate AgricultureImageDatasets for train and val
    # We use the first available source root as a common root
    # but pass explicit file lists
    source_roots_resolved = [
        Path(r).expanduser().resolve() for r in source_roots if Path(r).expanduser().resolve().exists()
    ]

    # Build individual datasets per source for train and val
    train_datasets: list[AgricultureImageDataset] = []
    val_datasets: list[AgricultureImageDataset] = []

    # Group records by their source_path
    train_by_root: dict[Path, list[ImageRecord]] = {}
    for rec in train_records:
        root_path = rec.source_path
        # Find the actual root (ZIP file or directory root)
        if root_path.suffix.lower() == ".zip":
            actual_root = root_path
        else:
            # Find which source root this belongs to
            actual_root = root_path
            for sr in source_roots_resolved:
                try:
                    root_path.relative_to(sr)
                    actual_root = sr
                    break
                except ValueError:
                    continue
        train_by_root.setdefault(actual_root, []).append(rec)

    val_by_root: dict[Path, list[ImageRecord]] = {}
    for rec in val_records:
        root_path = rec.source_path
        if root_path.suffix.lower() == ".zip":
            actual_root = root_path
        else:
            actual_root = root_path
            for sr in source_roots_resolved:
                try:
                    root_path.relative_to(sr)
                    actual_root = sr
                    break
                except ValueError:
                    continue
        val_by_root.setdefault(actual_root, []).append(rec)

    for root, records in train_by_root.items():
        ds = AgricultureImageDataset(
            root, files=records, crop_size=crop_size, channels=channels,
            precision=precision, augment=train_augment,
        )
        train_datasets.append(ds)

    for root, records in val_by_root.items():
        ds = AgricultureImageDataset(
            root, files=records, crop_size=crop_size, channels=channels,
            precision=precision, augment=val_augment,
        )
        val_datasets.append(ds)

    # Concatenate datasets
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    if len(val_datasets) == 1:
        val_dataset = val_datasets[0]
    else:
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    # Build sampler for source-balanced training
    if sampling_strategy != "uniform" and len(train_source_map) > 1:
        source_sizes = {}
        for name, indices in train_source_map.items():
            source_sizes[name] = len(indices)

        weights = []
        for i in range(len(train_records)):
            for name, indices in train_source_map.items():
                if i in indices:
                    size = source_sizes[name]
                    if sampling_strategy == "balanced":
                        weights.append(1.0 / size)
                    elif sampling_strategy == "sqrt":
                        weights.append(1.0 / math.sqrt(size))
                    else:
                        weights.append(1.0)
                    break
            else:
                weights.append(1.0)

        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(train_records), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_shuffle,
        generator=generator if train_sampler is None else None,
        worker_init_fn=_seed_worker,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        worker_init_fn=_seed_worker,
        **loader_kwargs,
    )
    return train_loader, val_loader, summary


def scan_pretraining_directory(
    pretraining_root: str | Path,
    *,
    exclude_sources: set[str] | None = None,
    skip_known_duplicates: bool = True,
) -> list[Path]:
    """Discover all loadable data sources (ZIPs and directories) under *pretraining_root*.

    Returns a sorted list of absolute paths to each source. Excludes
    directories that are clearly not image datasets (e.g., ``__MACOSX``).
    """
    root = Path(pretraining_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Pretraining root does not exist: {root}")

    exclude = exclude_sources or set()
    sources: list[Path] = []

    for item in sorted(root.iterdir()):
        if item.name.startswith(".") or item.name.startswith("__"):
            continue
        if item.suffix.lower() == ".yml" or item.suffix.lower() == ".txt":
            continue
        if item.suffix.lower() == ".pdf":
            continue

        stem = item.stem
        if stem in exclude:
            continue
        if skip_known_duplicates and _is_known_duplicate(item.name):
            continue

        # Only include ZIPs and directories (skip tar.gz for now)
        if item.suffix.lower() == ".zip":
            sources.append(item.resolve())
        elif item.is_dir():
            sources.append(item.resolve())

    return sources
