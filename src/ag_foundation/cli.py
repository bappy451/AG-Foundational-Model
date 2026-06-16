from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext

from .command_logging import command_log_context, parse_command_logging


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ag-foundation",
        description=(
            "Agricultural foundation-model utilities for dataset cataloging, GeoTIFF slicing, "
            "masked image modeling pretraining, DINO-style continual pretraining, command logging, "
            "and run-manifest capture."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("train-mim", "train-dino", "create-catalog", "create-demo-data", "slice-geotiffs"),
        help="Command to run.",
    )
    parser.add_argument("--log-file", help="Append stdout/stderr to this file.")
    parser.add_argument("--no-log", action="store_true", help="Disable command logging.")
    return parser


def _run_create_catalog(argv: list[str]) -> None:
    from ag_foundation.data.dataset import create_dataset_catalog

    parser = argparse.ArgumentParser(description="Create a CSV catalog for an agricultural image dataset.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args(argv)
    create_dataset_catalog(args.data_root, args.output_path)


def _run_slice_geotiffs(argv: list[str]) -> None:
    try:
        from ag_foundation.data.geotiff import slice_geotiff_collection_to_files
    except ImportError as exc:
        raise SystemExit("GeoTIFF slicing requires rasterio. Install the optional ML dependencies first.") from exc

    parser = argparse.ArgumentParser(description="Slice GeoTIFF files into pretraining tiles.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size", required=True, type=int)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--output-format", default="tif", choices=("tif", "png", "jpg", "jpeg"))
    parser.add_argument("--workers", default="auto")
    args = parser.parse_args(argv)
    slice_geotiff_collection_to_files(
        args.input_path,
        args.output_dir,
        tile_size=args.tile_size,
        stride=args.stride,
        output_format=args.output_format,
        workers=args.workers,
    )


def _run_create_demo_data(argv: list[str]) -> None:
    from ag_foundation.data.demo import create_demo_dataset

    parser = argparse.ArgumentParser(description="Create deterministic RGB and multispectral demo data.")
    parser.add_argument("--output-dir", default="data/demo")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--samples-per-group", type=int, default=6)
    parser.add_argument("--multispectral-channels", type=int, default=5)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args(argv)
    summary = create_demo_dataset(
        args.output_dir,
        image_size=args.image_size,
        samples_per_group=args.samples_per_group,
        multispectral_channels=args.multispectral_channels,
        seed=args.seed,
    )
    print(f"Created demo RGB data: {summary['rgb_root']}")
    print(f"Created demo multispectral data: {summary['multispectral_root']}")


def _run_train_dino(argv: list[str]) -> None:
    from ag_foundation.training.dino_runner import main as train_dino_main

    train_dino_main(argv)


def _dispatch(argv: list[str]) -> None:
    parser = _build_root_parser()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    command = argv[0]
    remainder = argv[1:]
    if command == "train-mim":
        from ag_foundation.training.mim_runner import main as train_mim_main

        train_mim_main(remainder)
        return
    if command == "train-dino":
        _run_train_dino(remainder)
        return
    if command == "create-catalog":
        _run_create_catalog(remainder)
        return
    if command == "create-demo-data":
        _run_create_demo_data(remainder)
        return
    if command == "slice-geotiffs":
        _run_slice_geotiffs(remainder)
        return
    parser.error(f"Unsupported command: {command}")


def main(
    argv: list[str] | None = None,
    *,
    enable_logging: bool | None = None,
) -> None:
    invoked_from_cli = argv is None
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    clean_argv, logging_config = parse_command_logging(raw_argv)
    if enable_logging is None:
        enable_logging = invoked_from_cli
    wrapper_is_logging = os.environ.get("AG_FOUNDATION_WRAPPER_LOGGING") == "1"
    context = (
        command_log_context(raw_argv, config=logging_config)
        if enable_logging and not wrapper_is_logging
        else nullcontext()
    )
    with context:
        _dispatch(clean_argv)
