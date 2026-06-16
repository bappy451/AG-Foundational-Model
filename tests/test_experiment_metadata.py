from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ag_foundation.models.mim import RemoteSensingMIMModel
from ag_foundation.training.experiment_metadata import build_run_manifest, write_run_manifest


class _Loader:
    def __init__(self, *, root: Path, batches: int, groups: list[str]) -> None:
        self.dataset = SimpleNamespace(
            root=root,
            records=[SimpleNamespace(group=group) for group in groups],
        )
        self._batches = [object() for _ in range(batches)]

    def __len__(self) -> int:
        return len(self._batches)


def test_run_manifest_writes_reproducibility_artifacts(fake_timm, tmp_path: Path) -> None:
    fake_timm()
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    train_loader = _Loader(root=tmp_path / "train", batches=3, groups=["farm-a", "farm-b"])
    val_loader = _Loader(root=tmp_path / "val", batches=1, groups=["farm-c"])
    manifest = build_run_manifest(
        command_name="train-mim",
        args={
            "data_root": str(tmp_path / "data.zip"),
            "output_dir": str(tmp_path / "run"),
            "channels": 4,
        },
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=tmp_path / "run",
        command_argv=["--data-root", str(tmp_path / "data.zip"), "--output-dir", str(tmp_path / "run")],
    )

    manifest_path = write_run_manifest(tmp_path / "run", manifest)

    assert manifest_path.exists()
    assert (tmp_path / "run" / "resolved_config.yaml").exists()
    assert (tmp_path / "run" / "model_summary.txt").exists()
    assert (tmp_path / "run" / "command.txt").exists()
    attempt_dirs = list((tmp_path / "run" / "attempts").iterdir())
    assert len(attempt_dirs) == 1
    assert (attempt_dirs[0] / "run_manifest.json").exists()
    assert (attempt_dirs[0] / "resolved_config.yaml").exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["command"]["name"] == "train-mim"
    assert payload["data"]["train"]["batches"] == 3
    assert payload["data"]["train"]["group_count"] == 2
    assert payload["data"]["val"]["group_count"] == 1
    assert payload["model"]["total_parameters"] > 0
    assert payload["resolved_args"]["channels"] == 4
