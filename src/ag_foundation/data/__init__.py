"""Data loading and GeoTIFF utilities for agricultural foundation-model training."""

from .demo import create_demo_dataset
from .pretraining_audit import parse_manifest, run_audit

__all__ = [
    "AgricultureImageDataset",
    "MultiSourcePretrainingDataset",
    "create_demo_dataset",
    "create_dataset_catalog",
    "get_dataloaders",
    "get_pretraining_dataloaders",
    "parse_manifest",
    "run_audit",
    "scan_pretraining_directory",
]


def __getattr__(name: str):
    if name in {"AgricultureImageDataset", "create_dataset_catalog", "get_dataloaders"}:
        from .dataset import AgricultureImageDataset, create_dataset_catalog, get_dataloaders

        exports = {
            "AgricultureImageDataset": AgricultureImageDataset,
            "create_dataset_catalog": create_dataset_catalog,
            "get_dataloaders": get_dataloaders,
        }
        return exports[name]
    if name in {
        "MultiSourcePretrainingDataset",
        "get_pretraining_dataloaders",
        "scan_pretraining_directory",
    }:
        from .multi_source_dataset import (
            MultiSourcePretrainingDataset,
            get_pretraining_dataloaders,
            scan_pretraining_directory,
        )

        exports = {
            "MultiSourcePretrainingDataset": MultiSourcePretrainingDataset,
            "get_pretraining_dataloaders": get_pretraining_dataloaders,
            "scan_pretraining_directory": scan_pretraining_directory,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
