"""Data loading and GeoTIFF utilities for agricultural foundation-model training."""

from .demo import create_demo_dataset
from .pretraining_audit import parse_manifest, run_audit

__all__ = [
    "AgricultureImageDataset",
    "create_demo_dataset",
    "create_dataset_catalog",
    "get_dataloaders",
    "parse_manifest",
    "run_audit",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
