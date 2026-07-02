from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ag_foundation.models.dino import RemoteSensingDINOModel
from ag_foundation.models.mim import RemoteSensingMIMModel
from ag_foundation.training.artifacts import save_training_checkpoint
from ag_foundation.training.dino_trainer import DINOTrainer
from ag_foundation.training.ssl_trainer import (
    SSLTrainer,
    _build_grad_scaler,
    select_torch_device,
)


class _TrackingSampler:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _SimpleLoader(list):
    def __init__(self, batches: list[dict[str, object]], sampler: object | None = None) -> None:
        super().__init__(batches)
        self.sampler = sampler


def _make_batch(batch_size: int = 2, channels: int = 4, size: int = 32, *, dtype=torch.float32) -> dict[str, object]:
    return {
        "image": torch.randn(batch_size, channels, size, size, dtype=dtype),
        "path": [f"a_{i}.npy" for i in range(batch_size)],
        "group": ["g1" for _ in range(batch_size)],
    }


def test_ssl_trainer_fit_writes_checkpoints_metrics_and_sets_epoch(fake_timm, tmp_path: Path) -> None:
    fake_timm()
    batch = _make_batch()
    sampler = _TrackingSampler()
    train_loader = _SimpleLoader([batch, batch], sampler=sampler)
    val_loader = _SimpleLoader([batch])
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SSLTrainer(
        model,
        train_loader,
        optimizer,
        val_loader=val_loader,
        device="cpu",
        precision="fp32",
        log_every=100,
    )

    summary = trainer.fit(epochs=1, output_dir=tmp_path / "mim-run")

    assert sampler.epochs == [0]
    assert (tmp_path / "mim-run" / "last.pt").exists()
    assert (tmp_path / "mim-run" / "best.pt").exists()
    assert (tmp_path / "mim-run" / "figures" / "training_metrics.png").exists()
    assert (tmp_path / "mim-run" / "figures" / "mim_reconstruction_epoch_0001.png").exists()
    metrics_path = tmp_path / "mim-run" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 1
    assert "learning_rate" in metrics["history"][0]
    assert metrics["system_info"]["resumed_from"] is None
    assert summary.train_batches == 2
    assert summary.val_batches == 1
    assert summary.final_train_loss is not None
    assert summary.final_val_loss is not None
    checkpoint = torch.load(tmp_path / "mim-run" / "last.pt", map_location="cpu", weights_only=False)
    assert "rng_state" in checkpoint
    assert "run_config" in checkpoint


def test_ssl_trainer_accumulates_gradients_before_stepping(fake_timm, tmp_path: Path) -> None:
    fake_timm()
    batch = _make_batch()
    train_loader = _SimpleLoader([batch, batch, batch])
    val_loader = _SimpleLoader([batch])
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SSLTrainer(
        model,
        train_loader,
        optimizer,
        val_loader=val_loader,
        device="cpu",
        precision="fp32",
        gradient_accumulation_steps=2,
        log_every=100,
    )

    summary = trainer.fit(epochs=1, output_dir=tmp_path / "mim-accum")

    metrics = json.loads((tmp_path / "mim-accum" / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(tmp_path / "mim-accum" / "last.pt", map_location="cpu", weights_only=False)
    assert metrics["history"][0]["optimizer_steps"] == 2
    assert metrics["system_info"]["gradient_accumulation_steps"] == 2
    assert metrics["system_info"]["optimizer_steps_completed"] == 2
    assert checkpoint["optimizer_step_count"] == 2
    assert summary.gradient_accumulation_steps == 2
    assert summary.optimizer_steps == 2


def test_ssl_trainer_resume_continues_history(fake_timm, tmp_path: Path) -> None:
    fake_timm()
    batch = _make_batch()
    train_loader = _SimpleLoader([batch, batch])
    val_loader = _SimpleLoader([batch])
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output_dir = tmp_path / "mim-resume"

    trainer = SSLTrainer(
        model,
        train_loader,
        optimizer,
        val_loader=val_loader,
        device="cpu",
        precision="fp32",
        log_every=100,
    )
    trainer.fit(epochs=1, output_dir=output_dir)

    resumed_model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_trainer = SSLTrainer(
        resumed_model,
        train_loader,
        resumed_optimizer,
        val_loader=val_loader,
        device="cpu",
        precision="fp32",
        log_every=100,
    )

    summary = resumed_trainer.fit(
        epochs=2,
        output_dir=output_dir,
        resume_from=output_dir / "last.pt",
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 2
    assert metrics["system_info"]["starting_epoch"] == 1
    assert metrics["system_info"]["resumed_from"].endswith("last.pt")
    assert summary.epochs == 2


def test_ssl_trainer_fp16_cpu_step_has_finite_loss_and_gradients(fake_timm) -> None:
    fake_timm()
    batch = _make_batch(dtype=torch.float16)
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp16",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SSLTrainer(
        model,
        _SimpleLoader([batch]),
        optimizer,
        device="cpu",
        precision="fp16",
    )

    metrics = trainer.train_step(batch)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_dino_trainer_fit_and_resume(fake_timm, tmp_path: Path) -> None:
    fake_timm()
    batch = _make_batch()
    train_loader = _SimpleLoader([batch, batch])
    val_loader = _SimpleLoader([batch])
    model = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output_dir = tmp_path / "dino-run"

    trainer = DINOTrainer(
        model,
        train_loader,
        optimizer,
        val_loader=val_loader,
        device="cpu",
        precision="fp32",
        log_every=100,
        num_global_crops=2,
        num_local_crops=1,
        student_temperature=0.1,
    )
    summary = trainer.fit(epochs=1, output_dir=output_dir)

    assert (output_dir / "last.pt").exists()
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "figures" / "training_metrics.png").exists()
    assert (output_dir / "figures" / "dino_views_epoch_0001.png").exists()
    assert (output_dir / "figures" / "dino_similarity_epoch_0001.png").exists()
    assert (output_dir / "diagnostics" / "dino_similarity_epoch_0001.csv").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 1
    assert summary.train_batches == 2
    assert summary.val_batches == 1
    assert summary.final_train_loss is not None
    assert summary.final_val_loss is not None

    resumed_model = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_trainer = DINOTrainer(
        resumed_model,
        train_loader,
        resumed_optimizer,
        val_loader=val_loader,
        device="cpu",
        precision="fp32",
        log_every=100,
        num_global_crops=2,
        num_local_crops=1,
        student_temperature=0.1,
    )
    resumed_summary = resumed_trainer.fit(
        epochs=2,
        output_dir=output_dir,
        resume_from=output_dir / "last.pt",
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 2
    assert metrics["system_info"]["starting_epoch"] == 1
    assert metrics["system_info"]["resumed_from"].endswith("last.pt")
    assert resumed_summary.epochs == 2


def test_dino_trainer_accumulates_gradients_before_teacher_updates(fake_timm, tmp_path: Path) -> None:
    fake_timm()
    batch = _make_batch()
    train_loader = _SimpleLoader([batch, batch, batch])
    val_loader = _SimpleLoader([batch])
    model = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output_dir = tmp_path / "dino-accum"

    trainer = DINOTrainer(
        model,
        train_loader,
        optimizer,
        val_loader=val_loader,
        device="cpu",
        precision="fp32",
        gradient_accumulation_steps=2,
        log_every=100,
        num_global_crops=2,
        num_local_crops=1,
        student_temperature=0.1,
    )
    summary = trainer.fit(epochs=1, output_dir=output_dir)

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert metrics["history"][0]["optimizer_steps"] == 2
    assert metrics["system_info"]["gradient_accumulation_steps"] == 2
    assert metrics["system_info"]["optimizer_steps_completed"] == 2
    assert checkpoint["optimizer_step_count"] == 2
    assert summary.gradient_accumulation_steps == 2
    assert summary.optimizer_steps == 2


def test_dino_student_and_teacher_receive_identically_augmented_global_views(fake_timm) -> None:
    fake_timm()
    torch.manual_seed(11)
    model = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    trainer = DINOTrainer(
        model,
        _SimpleLoader([_make_batch()]),
        optimizer,
        device="cpu",
        precision="fp32",
        num_global_crops=2,
        num_local_crops=1,
    )

    student_views, teacher_views = trainer._augment_batch(torch.rand(2, 4, 32, 32))

    assert len(student_views) == 3
    assert len(teacher_views) == 2
    assert torch.equal(student_views[0], teacher_views[0])
    assert torch.equal(student_views[1], teacher_views[1])


def test_checkpoint_snapshot_serializes_payload_once(tmp_path: Path, monkeypatch) -> None:
    calls = 0
    original_save = torch.save

    def tracking_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", tracking_save)
    last_path, best_path = save_training_checkpoint(
        {"epoch": 1, "tensor": torch.ones(2)},
        tmp_path,
        improved=True,
    )

    assert calls == 1
    assert last_path.exists()
    assert best_path is not None and best_path.exists()


def test_grad_scaler_falls_back_to_legacy_cuda_api() -> None:
    class _LegacyGradScaler:
        def __init__(self, *, enabled: bool) -> None:
            self.enabled = enabled

    fake_torch = type(
        "FakeTorch",
        (),
        {
            "amp": object(),
            "cuda": type("Cuda", (), {"amp": type("Amp", (), {"GradScaler": _LegacyGradScaler})})(),
        },
    )()

    scaler = _build_grad_scaler(fake_torch, enabled=True)

    assert scaler.enabled is True


def test_select_torch_device_prefers_cuda_over_mps(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert select_torch_device() == "cuda"


def test_select_torch_device_uses_cuda_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_torch_device() == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_torch_device() == "cpu"
