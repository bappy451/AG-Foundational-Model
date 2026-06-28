from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ag_foundation.data.dataset import get_dataloaders
from ag_foundation.models.mim import RemoteSensingMIMModel
from ag_foundation.models.vit import (
    SUPPORTED_PRETRAINED_SOURCES,
    VIT_CONFIGS,
    resolve_backbone_spec,
)

from .artifacts import load_training_checkpoint
from .experiment_metadata import build_run_manifest, resolve_config_paths, write_run_manifest
from .ssl_trainer import SSLTrainer, select_torch_device

TRAIN_MIM_DEFAULTS: dict[str, Any] = {
    "data_root": None,
    "output_dir": None,
    "catalog_path": None,
    "batch_size": 8,
    "epochs": 1,
    "seed": 27,
    "crop_size": 224,
    "channels": 3,
    "prefetch_factor": 4,
    "gradient_accumulation_steps": 1,
    "model_name": "S",
    "pretrained_backbone": True,
    "pretrained_source": "imagenet",
    "pretrained_cfg": None,
    "mask_ratio": 0.75,
    "gradient_checkpointing": False,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0,
    "precision": "bf16",
    "num_workers": 8,
    "compile": False,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 0,
    "device": None,
    "resume": False,
    "resume_from": None,
    "initialize_from": None,
    "log_every": 10,
    "save_visualizations": True,
    "visualization_every": 1,
    "visualization_samples": 4,
    "val_fraction": 0.2,
}


TRAIN_MIM_SECTION_MAP: dict[str, dict[str, str]] = {
    "data": {
        "data_root": "data_root",
        "catalog_path": "catalog_path",
        "crop_size": "crop_size",
        "channels": "channels",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "prefetch_factor": "prefetch_factor",
        "val_fraction": "val_fraction",
    },
    "runtime": {
        "output_dir": "output_dir",
        "epochs": "epochs",
        "seed": "seed",
        "precision": "precision",
        "device": "device",
        "warmup_epochs": "warmup_epochs",
        "resume": "resume",
        "resume_from": "resume_from",
        "initialize_from": "initialize_from",
        "log_every": "log_every",
        "save_visualizations": "save_visualizations",
        "visualization_every": "visualization_every",
        "visualization_samples": "visualization_samples",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
    },
    "model": {
        "model_name": "model_name",
        "pretrained_backbone": "pretrained_backbone",
        "pretrained_source": "pretrained_source",
        "pretrained_cfg": "pretrained_cfg",
        "mask_ratio": "mask_ratio",
        "gradient_checkpointing": "gradient_checkpointing",
        "drop_rate": "drop_rate",
        "attn_drop_rate": "attn_drop_rate",
        "drop_path_rate": "drop_path_rate",
    },
    "optimizer": {
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
    },
}


def load_train_mim_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must decode to a mapping: {config_path}")

    unknown_sections = sorted(set(payload) - set(TRAIN_MIM_SECTION_MAP))
    if unknown_sections:
        raise ValueError(
            f"Unknown config section(s) in {config_path}: {', '.join(unknown_sections)}."
        )

    flat: dict[str, Any] = {}
    for section_name, mapping in TRAIN_MIM_SECTION_MAP.items():
        section = payload.get(section_name, {})
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"Section '{section_name}' in {config_path} must be a mapping.")
        unknown_keys = sorted(set(section) - set(mapping))
        if unknown_keys:
            raise ValueError(
                f"Unknown key(s) in section '{section_name}' of {config_path}: "
                f"{', '.join(unknown_keys)}."
            )
        for yaml_key, arg_key in mapping.items():
            if yaml_key in section:
                flat[arg_key] = section[yaml_key]
    return resolve_config_paths(flat, config_path=config_path)


def build_train_mim_parser(config_defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = dict(TRAIN_MIM_DEFAULTS)
    if config_defaults:
        defaults.update({key: value for key, value in config_defaults.items() if value is not None})

    parser = argparse.ArgumentParser(description="Train the agricultural ViT masked image model.")
    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--data-root", default=defaults["data_root"], help="Dataset root directory or ZIP archive.")
    parser.add_argument("--output-dir", default=defaults["output_dir"], help="Directory for checkpoints and metrics.")
    parser.add_argument(
        "--catalog-path",
        default=defaults["catalog_path"],
        help="Optional CSV catalog created by create_catalog.py.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--crop-size", type=int, default=defaults["crop_size"])
    parser.add_argument("--channels", type=int, default=defaults["channels"])
    parser.add_argument("--prefetch-factor", type=int, default=defaults["prefetch_factor"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults["gradient_accumulation_steps"])
    parser.add_argument("--model-name", choices=tuple(VIT_CONFIGS), default=defaults["model_name"])
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=defaults["pretrained_backbone"],
    )
    parser.add_argument(
        "--pretrained-source",
        choices=SUPPORTED_PRETRAINED_SOURCES,
        default=defaults["pretrained_source"],
        help=(
            "Official ViT checkpoint family to use for initialization and patch-size matching. "
            "Choose imagenet, dinov2, dinov3, or mae."
        ),
    )
    parser.add_argument("--pretrained-cfg", default=defaults["pretrained_cfg"])
    parser.add_argument("--mask-ratio", type=float, default=defaults["mask_ratio"])
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=defaults["gradient_checkpointing"],
    )
    parser.add_argument("--drop-rate", type=float, default=defaults["drop_rate"])
    parser.add_argument("--attn-drop-rate", type=float, default=defaults["attn_drop_rate"])
    parser.add_argument("--drop-path-rate", type=float, default=defaults["drop_path_rate"])
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=defaults["precision"])
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=defaults.get("compile", False), help="Compile model using torch.compile")
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int, default=defaults["warmup_epochs"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=defaults["resume"])
    parser.add_argument("--resume-from", default=defaults["resume_from"])
    parser.add_argument(
        "--initialize-from",
        default=defaults["initialize_from"],
        help="Initialize weights from a previous SSL checkpoint without restoring optimizer state.",
    )
    parser.add_argument("--log-every", type=int, default=defaults["log_every"])
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=defaults["save_visualizations"],
    )
    parser.add_argument("--visualization-every", type=int, default=defaults["visualization_every"])
    parser.add_argument("--visualization-samples", type=int, default=defaults["visualization_samples"])
    parser.add_argument("--val-fraction", type=float, default=defaults["val_fraction"])
    return parser


def parse_train_mim_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None)
    known, _ = bootstrap.parse_known_args(argv)
    config_defaults = load_train_mim_config(known.config) if known.config else {}
    parser = build_train_mim_parser(config_defaults)
    args = parser.parse_args(argv)

    missing = [
        flag
        for flag, value in {
            "--data-root": args.data_root,
            "--output-dir": args.output_dir,
        }.items()
        if value in {None, ""}
    ]
    if missing:
        parser.error(f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} required.")
    _validate_common_training_args(args, parser)
    _validate_model_dimensions(args, parser)
    if not 0.0 <= args.mask_ratio <= 1.0:
        parser.error("--mask-ratio must be between 0 and 1.")
    return args


def _resolve_device(requested_device: str | None) -> str:
    if requested_device is None:
        return select_torch_device()
    normalized = str(requested_device).strip().lower()
    if normalized in {"", "auto"}:
        return select_torch_device()
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device '{requested_device}'.") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; "
                f"detected {torch.cuda.device_count()} CUDA device(s)."
            )
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
    elif device.type != "cpu":
        raise ValueError("device must be auto, cpu, cuda, cuda:<index>, or mps.")
    return str(device)


def _validate_common_training_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    positive_integer_fields = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--crop-size": args.crop_size,
        "--channels": args.channels,
        "--prefetch-factor": args.prefetch_factor,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--log-every": args.log_every,
        "--visualization-every": args.visualization_every,
        "--visualization-samples": args.visualization_samples,
    }
    for flag, value in positive_integer_fields.items():
        if int(value) <= 0:
            parser.error(f"{flag} must be a positive integer.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if int(args.gradient_accumulation_steps) <= 0:
        parser.error("--gradient-accumulation-steps must be a positive integer.")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive.")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay cannot be negative.")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        parser.error("--warmup-epochs must be between 0 and --epochs.")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1.")
    if args.initialize_from not in {None, ""} and (args.resume or args.resume_from not in {None, ""}):
        parser.error("--initialize-from cannot be combined with --resume or --resume-from.")
    for flag, value in {
        "--drop-rate": args.drop_rate,
        "--attn-drop-rate": args.attn_drop_rate,
        "--drop-path-rate": args.drop_path_rate,
    }.items():
        if not 0.0 <= value < 1.0:
            parser.error(f"{flag} must be in [0, 1).")


def _resolve_model_dimensions(args: argparse.Namespace) -> dict[str, Any]:
    spec = resolve_backbone_spec(args.model_name, pretrained_source=args.pretrained_source)
    return {
        "model_name": spec.model_name,
        "embed_dim": spec.embed_dim,
        "patch_size": spec.patch_size,
    }


def _validate_model_dimensions(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        spec = resolve_backbone_spec(args.model_name, pretrained_source=args.pretrained_source)
    except ValueError as exc:
        parser.error(str(exc))
    if args.crop_size % spec.patch_size != 0:
        parser.error(
            f"--crop-size must be divisible by the ViT patch size ({spec.patch_size}) for "
            f"{spec.model_name}."
        )
    has_pretrained_cfg = args.pretrained_cfg is not None and args.pretrained_cfg != ""
    if not args.pretrained_backbone and has_pretrained_cfg:
        parser.error("--pretrained-cfg requires --pretrained-backbone.")
    if args.pretrained_source != "imagenet" and has_pretrained_cfg:
        parser.error(
            "--pretrained-cfg is only supported with --pretrained-source imagenet. "
            "Official DINOv2, DINOv3, and MAE checkpoints are selected by name."
        )


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_epoch_lr_schedule(*, warmup_epochs: int) -> Any:
    warmup_epochs = max(0, int(warmup_epochs))

    def schedule(epoch_index: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return 1.0
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        decay_epochs = max(1, total_epochs - warmup_epochs)
        progress = float(epoch_index - warmup_epochs) / float(decay_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def _resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_from not in {None, ""}:
        return Path(args.resume_from).expanduser().resolve()
    if not args.resume:
        return None
    candidate = Path(args.output_dir).expanduser().resolve() / "last.pt"
    return candidate if candidate.exists() else None


def _resolve_initialize_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.initialize_from in {None, ""}:
        return None
    candidate = Path(args.initialize_from).expanduser()
    if candidate.is_dir():
        nested = candidate / "last.pt"
        if nested.exists():
            return nested.resolve()
    return candidate.resolve()


def _build_progress_callback(tag: str):
    def _callback(completed_work: int, total_work: int, *, detail: str = "") -> None:
        if total_work <= 0:
            return
        print(f"[{tag}] progress {completed_work + 1}/{total_work} | {detail}")

    return _callback


def run_train_mim(args: argparse.Namespace, *, command_argv: list[str] | None = None):
    set_global_seed(args.seed)
    resume_checkpoint = _resolve_resume_checkpoint(args)
    initialize_checkpoint = _resolve_initialize_checkpoint(args)
    print("[train-mim] Scanning dataset directories and catalog (this may take several minutes on slow storage)...", flush=True)
    train_loader, val_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        crop_size=args.crop_size,
        channels=args.channels,
        precision=args.precision,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        catalog_path=args.catalog_path,
        train_augment=True,
        val_augment=False,
    )
    print(f"[train-mim] Constructing RemoteSensingMIMModel (ViT-{args.model_name}) and loading weights...", flush=True)
    model = RemoteSensingMIMModel(
        in_channels=args.channels,
        image_size=args.crop_size,
        model_name=args.model_name,
        precision=args.precision,
        pretrained_backbone=args.pretrained_backbone and resume_checkpoint is None and initialize_checkpoint is None,
        pretrained_source=args.pretrained_source,
        pretrained_cfg=args.pretrained_cfg,
        mask_ratio=args.mask_ratio,
        gradient_checkpointing=args.gradient_checkpointing,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
    )
    if initialize_checkpoint is not None:
        checkpoint = load_training_checkpoint(initialize_checkpoint)
        model.initialize_from_state_dict(checkpoint.get("model_state_dict", checkpoint))
        
    if getattr(args, "compile", False):
        print(f"[train-mim] Compiling model with torch.compile...", flush=True)
        model = torch.compile(model)
        
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_config = dict(vars(args))
    run_config["resolved_resume_checkpoint"] = (
        None if resume_checkpoint is None else str(resume_checkpoint)
    )
    run_config["initialize_from"] = None if initialize_checkpoint is None else str(initialize_checkpoint)
    run_config["backbone_initialized_from_timm"] = bool(
        args.pretrained_backbone and resume_checkpoint is None and initialize_checkpoint is None
    )
    run_config["compile"] = getattr(args, "compile", False)
    run_config["effective_batch_size"] = int(args.batch_size) * int(args.gradient_accumulation_steps)
    print("[train-mim] Assembling trainer, optimizers, and schedulers...", flush=True)
    trainer = SSLTrainer(
        model,
        train_loader,
        optimizer,
        val_loader=val_loader,
        device=_resolve_device(args.device),
        precision=args.precision,
        epoch_lr_schedule=build_epoch_lr_schedule(warmup_epochs=args.warmup_epochs),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_every=args.log_every,
        progress_callback=_build_progress_callback("train-mim"),
        save_visualizations=args.save_visualizations,
        visualization_every=args.visualization_every,
        visualization_samples=args.visualization_samples,
        run_config=run_config,
    )
    manifest = build_run_manifest(
        command_name="train-mim",
        args=vars(args),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        command_argv=command_argv,
    )
    manifest["model"]["initialization"] = {
        "pretrained_requested": bool(args.pretrained_backbone),
        "pretrained_source": args.pretrained_source,
        "timm_pretrained_loaded": bool(
            args.pretrained_backbone and resume_checkpoint is None and initialize_checkpoint is None
        ),
        "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "initialize_from": None if initialize_checkpoint is None else str(initialize_checkpoint),
    }
    manifest_path = write_run_manifest(args.output_dir, manifest)
    print(f"[metadata] Saved run manifest to {manifest_path}")
    return trainer.fit(
        args.epochs,
        Path(args.output_dir),
        resume_from=resume_checkpoint,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_train_mim_args(argv)
    summary = run_train_mim(args, command_argv=list(argv or []))
    print(summary)
