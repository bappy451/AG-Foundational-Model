from __future__ import annotations

from pathlib import Path

import pytest

from ag_foundation.training.dino_runner import load_train_dino_config, parse_train_dino_args
from ag_foundation.training.mim_runner import (
    _resolve_device,
    _resolve_model_dimensions,
    build_epoch_lr_schedule,
    load_train_mim_config,
    parse_train_mim_args,
)


def test_load_train_mim_config_flattens_nested_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
data:
  data_root: ../data.zip
  catalog_path: ../catalog.csv
  crop_size: 128
  channels: 4
  batch_size: 16
  num_workers: 8
  prefetch_factor: 3
  val_fraction: 0.15
runtime:
  output_dir: ../run
  epochs: 5
  seed: 19
  precision: fp16
  device: auto
  warmup_epochs: 2
  resume: true
  resume_from: ../run/last.pt
  log_every: 7
model:
  model_name: B
  pretrained_backbone: true
  pretrained_cfg: augreg_in1k
  mask_ratio: 0.8
  gradient_checkpointing: true
  drop_rate: 0.1
  attn_drop_rate: 0.05
  drop_path_rate: 0.1
optimizer:
  learning_rate: 0.0002
  weight_decay: 0.05
""".strip(),
        encoding="utf-8",
    )

    flat = load_train_mim_config(config_path)

    assert flat == {
        "data_root": str((tmp_path / "data.zip").resolve()),
        "catalog_path": str((tmp_path / "catalog.csv").resolve()),
        "crop_size": 128,
        "channels": 4,
        "batch_size": 16,
        "num_workers": 8,
        "prefetch_factor": 3,
        "val_fraction": 0.15,
        "output_dir": str((tmp_path / "run").resolve()),
        "epochs": 5,
        "seed": 19,
        "precision": "fp16",
        "device": "auto",
        "warmup_epochs": 2,
        "resume": True,
        "resume_from": str((tmp_path / "run" / "last.pt").resolve()),
        "log_every": 7,
        "model_name": "B",
        "pretrained_backbone": True,
        "pretrained_cfg": "augreg_in1k",
        "mask_ratio": 0.8,
        "gradient_checkpointing": True,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.05,
        "drop_path_rate": 0.1,
        "learning_rate": 0.0002,
        "weight_decay": 0.05,
    }


def test_parse_train_mim_args_lets_cli_override_yaml_values(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
data:
  data_root: /tmp/from-config.zip
  channels: 3
runtime:
  output_dir: /tmp/output
  precision: fp16
model:
  model_name: B
  pretrained_backbone: false
""".strip(),
        encoding="utf-8",
    )

    args = parse_train_mim_args(
        [
            "--config",
            str(config_path),
            "--data-root",
            "/tmp/from-cli.zip",
            "--channels",
            "5",
            "--epochs",
            "2",
            "--resume",
            "--prefetch-factor",
            "5",
            "--pretrained-backbone",
            "--drop-rate",
            "0.1",
            "--attn-drop-rate",
            "0.2",
        ]
    )

    assert args.config == str(config_path)
    assert args.data_root == "/tmp/from-cli.zip"
    assert args.output_dir == "/tmp/output"
    assert args.precision == "fp16"
    assert args.channels == 5
    assert args.epochs == 2
    assert args.resume is True
    assert args.prefetch_factor == 5
    assert args.model_name == "B"
    assert args.pretrained_backbone is True
    assert args.drop_rate == 0.1
    assert args.attn_drop_rate == 0.2


def test_resolve_device_maps_auto_to_supported_backend(monkeypatch) -> None:
    monkeypatch.setattr("ag_foundation.training.mim_runner.select_torch_device", lambda: "cpu")
    monkeypatch.setattr("ag_foundation.training.mim_runner.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("ag_foundation.training.mim_runner.torch.cuda.device_count", lambda: 1)
    assert _resolve_device("auto") == "cpu"
    assert _resolve_device("") == "cpu"
    assert _resolve_device(None) == "cpu"
    assert _resolve_device("CUDA") == "cuda"


def test_resolve_model_dimensions_uses_preset_and_validates_variants() -> None:
    args = parse_train_mim_args(
        [
            "--data-root",
            "/tmp/data.zip",
            "--output-dir",
            "/tmp/run",
            "--model-name",
            "B",
        ]
    )

    resolved = _resolve_model_dimensions(args)

    assert resolved == {
        "model_name": "vit_base_patch16_224",
        "embed_dim": 768,
        "patch_size": 16,
    }


def test_build_epoch_lr_schedule_warms_up_then_decays() -> None:
    schedule = build_epoch_lr_schedule(warmup_epochs=2)

    assert schedule(0, 5) == 0.5
    assert schedule(1, 5) == 1.0
    assert schedule(4, 5) == pytest.approx(0.25)
    assert build_epoch_lr_schedule(warmup_epochs=0)(1, 2) == pytest.approx(0.5)


def test_train_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
data:
  data_root: /tmp/data
runtime:
  output_dir: /tmp/run
model:
  model_name: S
  embed_dim: 192
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embed_dim"):
        load_train_mim_config(config_path)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--epochs", "0"],
        ["--crop-size", "30"],
        ["--warmup-epochs", "2", "--epochs", "1"],
        ["--learning-rate", "0"],
    ],
)
def test_parse_train_mim_args_rejects_invalid_values(extra_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_train_mim_args(
            [
                "--data-root",
                "/tmp/data",
                "--output-dir",
                "/tmp/run",
                *extra_args,
            ]
        )


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--num-global-crops", "1"],
        ["--student-temperature", "0"],
        [
            "--teacher-momentum-start",
            "1",
            "--teacher-momentum-end",
            "0.9",
            "--teacher-momentum-schedule",
            "cosine",
        ],
        ["--local-crop-scale", "0.8", "0.2"],
    ],
)
def test_parse_train_dino_args_rejects_invalid_values(extra_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_train_dino_args(
            [
                "--data-root",
                "/tmp/data",
                "--output-dir",
                "/tmp/run",
                *extra_args,
            ]
        )


def test_load_train_dino_config_flattens_nested_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "train_dino.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
data:
  data_root: ../data.zip
  catalog_path: ../catalog.csv
  crop_size: 160
  channels: 4
  batch_size: 12
  num_workers: 6
  prefetch_factor: 2
  val_fraction: 0.2
runtime:
  output_dir: ../run-dino
  epochs: 6
  seed: 31
  precision: bf16
  device: auto
  warmup_epochs: 1
  resume: true
  resume_from: ../run-dino/last.pt
  log_every: 9
model:
  model_name: S
  pretrained_backbone: true
  pretrained_cfg: augreg_in1k
  dino_out_dim: 256
  dino_hidden_dim: 1024
  dino_bottleneck_dim: 128
  head_nlayers: 2
  num_global_crops: 2
  num_local_crops: 3
  global_crop_scale: [0.7, 1.0]
  local_crop_scale: [0.4, 0.6]
  student_temperature: 0.1
  teacher_temperature: 0.04
  teacher_momentum_start: 0.996
  teacher_momentum_end: 0.996
  teacher_momentum_schedule: constant
  center_momentum: 0.9
  gram_anchor_weight: 0.25
  gram_anchor_max_tokens: 64
  gradient_checkpointing: false
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1
optimizer:
  learning_rate: 0.0003
  weight_decay: 0.02
""".strip(),
        encoding="utf-8",
    )

    flat = load_train_dino_config(config_path)

    assert flat == {
        "data_root": str((tmp_path / "data.zip").resolve()),
        "catalog_path": str((tmp_path / "catalog.csv").resolve()),
        "crop_size": 160,
        "channels": 4,
        "batch_size": 12,
        "num_workers": 6,
        "prefetch_factor": 2,
        "val_fraction": 0.2,
        "output_dir": str((tmp_path / "run-dino").resolve()),
        "epochs": 6,
        "seed": 31,
        "precision": "bf16",
        "device": "auto",
        "warmup_epochs": 1,
        "resume": True,
        "resume_from": str((tmp_path / "run-dino" / "last.pt").resolve()),
        "log_every": 9,
        "model_name": "S",
        "pretrained_backbone": True,
        "pretrained_cfg": "augreg_in1k",
        "dino_out_dim": 256,
        "dino_hidden_dim": 1024,
        "dino_bottleneck_dim": 128,
        "head_nlayers": 2,
        "num_global_crops": 2,
        "num_local_crops": 3,
        "global_crop_scale": [0.7, 1.0],
        "local_crop_scale": [0.4, 0.6],
        "student_temperature": 0.1,
        "teacher_temperature": 0.04,
        "teacher_momentum_start": 0.996,
        "teacher_momentum_end": 0.996,
        "teacher_momentum_schedule": "constant",
        "center_momentum": 0.9,
        "gram_anchor_weight": 0.25,
        "gram_anchor_max_tokens": 64,
        "gradient_checkpointing": False,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "learning_rate": 0.0003,
        "weight_decay": 0.02,
    }


def test_parse_train_dino_args_lets_cli_override_yaml_values(tmp_path: Path) -> None:
    config_path = tmp_path / "train_dino.yaml"
    config_path.write_text(
        """
data:
  data_root: /tmp/from-config.zip
runtime:
  output_dir: /tmp/output-dino
model:
  model_name: B
  num_local_crops: 1
  global_crop_scale: [0.8, 1.0]
  gram_anchor_weight: 0.1
""".strip(),
        encoding="utf-8",
    )

    args = parse_train_dino_args(
        [
            "--config",
            str(config_path),
            "--data-root",
            "/tmp/from-cli.zip",
            "--num-local-crops",
            "3",
            "--global-crop-scale",
            "0.7",
            "1.0",
            "--pretrained-backbone",
            "--dino-out-dim",
            "128",
            "--gram-anchor-weight",
            "0.35",
            "--teacher-momentum-schedule",
            "constant",
        ]
    )

    assert args.data_root == "/tmp/from-cli.zip"
    assert args.output_dir == "/tmp/output-dino"
    assert args.model_name == "B"
    assert args.num_local_crops == 3
    assert args.global_crop_scale == [0.7, 1.0]
    assert args.pretrained_backbone is True
    assert args.dino_out_dim == 128
    assert args.gram_anchor_weight == 0.35
    assert args.teacher_momentum_schedule == "constant"
