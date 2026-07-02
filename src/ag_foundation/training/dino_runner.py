from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from ag_foundation.data.dataset import get_dataloaders
from ag_foundation.models.dino import RemoteSensingDINOModel
from ag_foundation.models.vit import (
    SUPPORTED_PRETRAINED_SOURCES,
    VIT_CONFIGS,
    resolve_backbone_spec,
)

from .artifacts import load_training_checkpoint
from .dino_trainer import DINOAugmentationConfig, DINOTrainer
from .experiment_metadata import build_run_manifest, resolve_config_paths, write_run_manifest
from .mim_runner import (
    _resolve_device,
    _validate_common_training_args,
    build_epoch_lr_schedule,
    set_global_seed,
)

TRAIN_DINO_DEFAULTS: dict[str, Any] = {
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
    "dino_out_dim": 65536,
    "dino_hidden_dim": 2048,
    "dino_bottleneck_dim": 256,
    "head_nlayers": 3,
    "num_global_crops": 2,
    "num_local_crops": 2,
    "global_crop_scale": (0.6, 1.0),
    "local_crop_scale": (0.3, 0.6),
    "student_temperature": 0.1,
    "teacher_temperature": 0.04,
    "teacher_momentum_start": 0.996,
    "teacher_momentum_end": 1.0,
    "center_momentum": 0.9,
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
    "epoch_batches": 100000,
    "use_dali": False,
}


TRAIN_DINO_SECTION_MAP: dict[str, dict[str, str]] = {
    "data": {
        "data_root": "data_root",
        "catalog_path": "catalog_path",
        "crop_size": "crop_size",
        "channels": "channels",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "prefetch_factor": "prefetch_factor",
        "val_fraction": "val_fraction",
        "epoch_batches": "epoch_batches",
        "use_dali": "use_dali",
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
        "dino_out_dim": "dino_out_dim",
        "dino_hidden_dim": "dino_hidden_dim",
        "dino_bottleneck_dim": "dino_bottleneck_dim",
        "head_nlayers": "head_nlayers",
        "num_global_crops": "num_global_crops",
        "num_local_crops": "num_local_crops",
        "global_crop_scale": "global_crop_scale",
        "local_crop_scale": "local_crop_scale",
        "student_temperature": "student_temperature",
        "teacher_temperature": "teacher_temperature",
        "teacher_momentum_start": "teacher_momentum_start",
        "teacher_momentum_end": "teacher_momentum_end",
        "center_momentum": "center_momentum",
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


def load_train_dino_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must decode to a mapping: {config_path}")

    unknown_sections = sorted(set(payload) - set(TRAIN_DINO_SECTION_MAP))
    if unknown_sections:
        raise ValueError(
            f"Unknown config section(s) in {config_path}: {', '.join(unknown_sections)}."
        )

    flat: dict[str, Any] = {}
    for section_name, mapping in TRAIN_DINO_SECTION_MAP.items():
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


def build_train_dino_parser(config_defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = dict(TRAIN_DINO_DEFAULTS)
    if config_defaults:
        defaults.update({key: value for key, value in config_defaults.items() if value is not None})

    parser = argparse.ArgumentParser(description="Train the agricultural ViT with DINOv2-style self-distillation.")
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
    parser.add_argument("--dino-out-dim", type=int, default=defaults["dino_out_dim"])
    parser.add_argument("--dino-hidden-dim", type=int, default=defaults["dino_hidden_dim"])
    parser.add_argument("--dino-bottleneck-dim", type=int, default=defaults["dino_bottleneck_dim"])
    parser.add_argument("--head-nlayers", type=int, default=defaults["head_nlayers"])
    parser.add_argument("--num-global-crops", type=int, default=defaults["num_global_crops"])
    parser.add_argument("--num-local-crops", type=int, default=defaults["num_local_crops"])
    parser.add_argument("--global-crop-scale", nargs=2, type=float, default=defaults["global_crop_scale"])
    parser.add_argument("--local-crop-scale", nargs=2, type=float, default=defaults["local_crop_scale"])
    parser.add_argument("--student-temperature", type=float, default=defaults["student_temperature"])
    parser.add_argument("--teacher-temperature", type=float, default=defaults["teacher_temperature"])
    parser.add_argument("--teacher-momentum-start", type=float, default=defaults["teacher_momentum_start"])
    parser.add_argument("--teacher-momentum-end", type=float, default=defaults["teacher_momentum_end"])
    parser.add_argument("--center-momentum", type=float, default=defaults["center_momentum"])
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
    parser.add_argument("--epoch-batches", type=int, default=defaults["epoch_batches"], help="Number of batches per epoch for WebDataset.")
    parser.add_argument("--use-dali", action="store_true", help="Use NVIDIA DALI for WebDataset decoding (Colab/Linux only).")
    parser.add_argument("--no-use-dali", action="store_false", dest="use_dali")
    parser.set_defaults(use_dali=defaults["use_dali"])
    return parser


def parse_train_dino_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None)
    known, _ = bootstrap.parse_known_args(argv)
    config_defaults = load_train_dino_config(known.config) if known.config else {}
    parser = build_train_dino_parser(config_defaults)
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
    _validate_dino_args(args, parser)
    return args


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


def _pair(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError("crop scale values must contain exactly two numbers.")


def _validate_dino_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    for flag, value in {
        "--dino-out-dim": args.dino_out_dim,
        "--dino-hidden-dim": args.dino_hidden_dim,
        "--dino-bottleneck-dim": args.dino_bottleneck_dim,
        "--head-nlayers": args.head_nlayers,
    }.items():
        if int(value) <= 0:
            parser.error(f"{flag} must be a positive integer.")
    if args.num_global_crops < 2:
        parser.error("--num-global-crops must be at least 2.")
    if args.num_local_crops < 0:
        parser.error("--num-local-crops cannot be negative.")
    if args.student_temperature <= 0.0 or args.teacher_temperature <= 0.0:
        parser.error("DINO student and teacher temperatures must be positive.")
    if not 0.0 <= args.center_momentum < 1.0:
        parser.error("--center-momentum must be in [0, 1).")
    if not 0.0 <= args.teacher_momentum_start <= args.teacher_momentum_end <= 1.0:
        parser.error(
            "Teacher momenta must satisfy 0 <= --teacher-momentum-start "
            "<= --teacher-momentum-end <= 1."
        )
    for flag, value in {
        "--global-crop-scale": args.global_crop_scale,
        "--local-crop-scale": args.local_crop_scale,
    }.items():
        try:
            minimum, maximum = _pair(value)
        except ValueError as exc:
            parser.error(str(exc))
        if not 0.0 < minimum <= maximum <= 1.0:
            parser.error(f"{flag} must satisfy 0 < min <= max <= 1.")


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


def run_train_dino(args: argparse.Namespace, *, command_argv: list[str] | None = None, progress_callback = None):
    set_global_seed(args.seed)
    resume_checkpoint = _resolve_resume_checkpoint(args)
    initialize_checkpoint = _resolve_initialize_checkpoint(args)
    data_root_str = str(args.data_root)
    if "*" in data_root_str and data_root_str.endswith(".tar"):
        import glob
        import os
        import sys

        resolved_tars = []
        if "*" in data_root_str or "?" in data_root_str:
            resolved_tars.extend(glob.glob(data_root_str))
        else:
            resolved_tars.append(data_root_str)

        if not resolved_tars:
            raise ValueError(f"No tar files found matching: {data_root_str}")

        if sys.platform == "win32":
            from webdataset.gopen import gopen_schemes

            gopen_schemes["winfile"] = lambda url, mode="rb", bufsize=8192, **kw: open(
                url.replace("winfile://", ""), mode, buffering=bufsize
            )
            resolved_tars = [
                f"winfile://{os.path.abspath(t).replace(os.sep, '/')}" for t in resolved_tars
            ]

        if args.use_dali:
            print("[train-dino] Detected WebDataset tarballs. Using NVIDIA DALI GPU loader.", flush=True)
            from ag_foundation.data.dali_wds_loader import build_dali_wds_dataloader

            train_loader = build_dali_wds_dataloader(
                tar_urls=resolved_tars,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epoch_batches=args.epoch_batches,
                crop_size=args.crop_size,
            )
        else:
            print("[train-dino] Detected WebDataset tarballs in config. Using high-performance WDS CPU loader.", flush=True)
            from ag_foundation.data.wds_loader import build_wds_dataloader

            train_loader = build_wds_dataloader(
                tar_urls=resolved_tars,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epoch_batches=args.epoch_batches,
                crop_size=args.crop_size,
            )
        val_loader = None
    else:
        print("[train-dino] Scanning dataset directories and catalog (this may take several minutes on slow storage)...", flush=True)
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
            train_augment=False,
            val_augment=False,
        )
    augmentation_config = DINOAugmentationConfig(
        image_size=(args.crop_size, args.crop_size),
        num_global_crops=args.num_global_crops,
        num_local_crops=args.num_local_crops,
        global_crop_scale=_pair(args.global_crop_scale),
        local_crop_scale=_pair(args.local_crop_scale),
    )
    print(f"[train-dino] Constructing RemoteSensingDINOModel (ViT-{args.model_name}) and loading weights...", flush=True)
    model = RemoteSensingDINOModel(
        in_channels=args.channels,
        image_size=args.crop_size,
        model_name=args.model_name,
        precision=args.precision,
        pretrained_backbone=args.pretrained_backbone and resume_checkpoint is None and initialize_checkpoint is None,
        pretrained_source=args.pretrained_source,
        pretrained_cfg=args.pretrained_cfg,
        dino_out_dim=args.dino_out_dim,
        dino_hidden_dim=args.dino_hidden_dim,
        dino_bottleneck_dim=args.dino_bottleneck_dim,
        head_nlayers=args.head_nlayers,
        gradient_checkpointing=args.gradient_checkpointing,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        teacher_temperature=args.teacher_temperature,
    )
    if initialize_checkpoint is not None:
        checkpoint = load_training_checkpoint(initialize_checkpoint)
        model.initialize_from_state_dict(checkpoint.get("model_state_dict", checkpoint))
        
    if getattr(args, "compile", False):
        print(f"[train-dino] Compiling model with torch.compile...", flush=True)
        model = torch.compile(model)
        
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
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
    print("[train-dino] Assembling trainer, optimizers, and schedulers...", flush=True)
    trainer = DINOTrainer(
        model,
        train_loader,
        optimizer,
        val_loader=val_loader,
        device=_resolve_device(args.device),
        precision=args.precision,
        epoch_lr_schedule=build_epoch_lr_schedule(warmup_epochs=args.warmup_epochs),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_every=args.log_every,
        num_global_crops=args.num_global_crops,
        num_local_crops=args.num_local_crops,
        student_temperature=args.student_temperature,
        center_momentum=args.center_momentum,
        teacher_momentum_start=args.teacher_momentum_start,
        teacher_momentum_end=args.teacher_momentum_end,
        augmentation_config=augmentation_config,
        progress_callback=progress_callback,
        save_visualizations=args.save_visualizations,
        visualization_every=args.visualization_every,
        visualization_samples=args.visualization_samples,
        run_config=run_config,
    )
    manifest = build_run_manifest(
        command_name="train-dino",
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
    args = parse_train_dino_args(argv)
    from ag_foundation.progress import command_progress_context
    with command_progress_context(argv) as progress:
        progress_cb = progress.update if progress else None
        summary = run_train_dino(args, command_argv=list(argv or []), progress_callback=progress_cb)
        print(summary)
