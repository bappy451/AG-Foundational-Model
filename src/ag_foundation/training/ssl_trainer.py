from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .artifacts import (
    atomic_write_text,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .state import (
    capture_loader_generator_state,
    capture_rng_state,
    restore_loader_generator_state,
    restore_rng_state,
)
from .visualization import save_mim_preview, save_training_curves


def select_torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SSLTrainingSummary:
    output_dir: str
    epochs: int
    train_batches: int
    val_batches: int
    final_train_loss: float | None
    final_val_loss: float | None


def _move_ssl_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved = dict(batch)
    image = moved.get("image")
    if hasattr(image, "to"):
        moved["image"] = image.to(device=device, non_blocking=True)
    return moved


def _move_optimizer_state_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device=device)


def _build_grad_scaler(torch_module: Any, *, enabled: bool) -> Any:
    grad_scaler = getattr(getattr(torch_module, "amp", None), "GradScaler", None)
    if grad_scaler is not None:
        try:
            return grad_scaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch_module.cuda.amp.GradScaler(enabled=enabled)


class SSLTrainer:
    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        *,
        val_loader=None,
        device=None,
        precision: str = "fp32",
        scheduler=None,
        epoch_lr_schedule: Callable[[int, int], float] | None = None,
        grad_clip_norm: float | None = None,
        log_every: int = 10,
        progress_callback: Any | None = None,
        save_visualizations: bool = True,
        visualization_every: int = 1,
        visualization_samples: int = 4,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        import torch

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_lr_schedule = epoch_lr_schedule
        self.grad_clip_norm = grad_clip_norm
        self.log_every = max(1, int(log_every))
        self.requested_precision = precision
        self.progress_callback = progress_callback
        self.save_visualizations = bool(save_visualizations)
        self.visualization_every = max(1, int(visualization_every))
        self.visualization_samples = max(1, int(visualization_samples))
        self.run_config = dict(run_config or {})
        self.device = torch.device(device) if device is not None else torch.device(select_torch_device())
        self.model = self.model.to(self.device)
        self._trainable_parameter_items = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.grad_scaler = _build_grad_scaler(
            torch,
            enabled=self.device.type == "cuda" and precision == "fp16",
        )

    def _autocast_context(self):
        import contextlib

        import torch

        if self.requested_precision == "fp16":
            enabled = self.device.type in {"cuda", "mps"}
            return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=enabled)
        if self.requested_precision == "bf16":
            enabled = self.device.type in {"cpu", "cuda"}
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=enabled)
        return contextlib.nullcontext()

    def _trainable_parameters(self):
        return [parameter for _, parameter in self._trainable_parameter_items]

    def _validate_gradients(self) -> None:
        import torch

        for name, parameter in self._trainable_parameter_items:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Encountered a non-finite gradient in parameter '{name}'.")

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply_epoch_learning_rate(self, epoch_index: int, total_epochs: int) -> None:
        if self.epoch_lr_schedule is None:
            return
        scale = float(self.epoch_lr_schedule(epoch_index, total_epochs))
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = float(base_lr * scale)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        import torch

        self.model.train()
        moved_batch = _move_ssl_batch_to_device(batch, self.device)
        images = moved_batch["image"]
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            loss = self.model(images)
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite SSL loss.")

        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self._validate_gradients()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_parameters(),
                    self.grad_clip_norm,
                    error_if_nonfinite=True,
                )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self._validate_gradients()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_parameters(),
                    self.grad_clip_norm,
                    error_if_nonfinite=True,
                )
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return {"loss": float(loss.detach().cpu())}

    def train_epoch(self, epoch_index: int, total_epochs: int = 1) -> dict[str, float]:
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_index)

        total_loss = 0.0
        total_batches = 0
        num_batches = len(self.train_loader)
        last_loss = 0.0

        for step_index, batch in enumerate(self.train_loader, start=1):
            metrics = self.train_step(batch)
            current_loss = metrics["loss"]
            total_loss += current_loss
            total_batches += 1
            last_loss = current_loss
            if self.progress_callback is not None:
                total_work = total_epochs * num_batches
                completed_work = (epoch_index * num_batches) + step_index - 1
                avg_loss = total_loss / total_batches
                lr = self._current_learning_rate()
                detail = (
                    f"ep {epoch_index + 1}/{total_epochs} | "
                    f"batch {step_index}/{num_batches} | "
                    f"loss {last_loss:.4f} (avg {avg_loss:.4f}) | "
                    f"lr {lr:.6f}"
                )
                self.progress_callback(completed_work, total_work, detail=detail)
            if step_index % self.log_every == 0:
                print(
                    f"train-mim epoch={epoch_index + 1} step={step_index}/{num_batches} "
                    f"loss={current_loss:.6f}"
                )

        return {
            "loss": float(total_loss / total_batches) if total_batches > 0 else float("nan"),
            "batches": total_batches,
        }

    def evaluate(self, epoch_index: int) -> dict[str, float]:
        import torch

        if self.val_loader is None:
            return {"loss": float("nan"), "batches": 0}

        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in self.val_loader:
                moved_batch = _move_ssl_batch_to_device(batch, self.device)
                with self._autocast_context():
                    loss = self.model(moved_batch["image"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Encountered a non-finite SSL validation loss.")
                losses.append(float(loss.detach().cpu()))

        mean_loss = float(sum(losses) / len(losses)) if losses else float("nan")
        print(f"train-mim epoch={epoch_index + 1} val_loss={mean_loss:.6f}")
        return {"loss": mean_loss, "batches": len(losses)}

    def _best_metric_from_history(self, history: list[dict[str, Any]]) -> float:
        best_metric = float("inf")
        for record in history:
            candidate = record.get("val_loss")
            if candidate is None:
                candidate = record.get("train_loss")
            if candidate is None:
                continue
            candidate_value = float(candidate)
            if candidate_value < best_metric:
                best_metric = candidate_value
        return best_metric

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint = load_training_checkpoint(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(self.optimizer, self.device)
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        grad_scaler_state = checkpoint.get("grad_scaler_state_dict")
        if grad_scaler_state is not None and self.grad_scaler.is_enabled():
            self.grad_scaler.load_state_dict(grad_scaler_state)
        restore_rng_state(checkpoint.get("rng_state"))
        restore_loader_generator_state(
            self.train_loader,
            checkpoint.get("train_loader_generator_state"),
        )
        return checkpoint

    def _save_visualization(self, output_dir: Path, epoch: int) -> list[Path]:
        import torch

        if not self.save_visualizations or epoch % self.visualization_every != 0:
            return []

        loader = self.val_loader if self.val_loader is not None else self.train_loader
        rng_state = capture_rng_state()
        loader_state = capture_loader_generator_state(loader)
        try:
            batch = next(iter(loader))
            moved_batch = _move_ssl_batch_to_device(batch, self.device)
            self.model.eval()
            with torch.no_grad(), self._autocast_context():
                outputs = self.model.forward_with_intermediates(moved_batch["image"])
            return save_mim_preview(
                outputs,
                output_dir,
                epoch=epoch,
                patch_size=self.model.patch_size,
                max_samples=self.visualization_samples,
            )
        finally:
            restore_rng_state(rng_state)
            restore_loader_generator_state(loader, loader_state)

    def fit(
        self,
        epochs: int,
        output_dir: str | Path,
        *,
        resume_from: str | Path | None = None,
    ) -> SSLTrainingSummary:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "device": str(self.device),
            "start_time": start_time,
        }

        history: list[dict[str, Any]] = []
        best_metric = float("inf")
        final_train_loss: float | None = None
        final_val_loss: float | None = None
        total_epochs = int(epochs)
        start_epoch = 0
        resumed_from: str | None = None

        if resume_from is not None:
            checkpoint_path = Path(resume_from)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            checkpoint = self.load_checkpoint(checkpoint_path)
            resumed_from = str(checkpoint_path)
            history = list(checkpoint.get("history", []))
            start_epoch = int(checkpoint.get("epoch", 0))
            best_metric = float(checkpoint.get("best_metric", self._best_metric_from_history(history)))
            if history:
                final_train_loss = history[-1].get("train_loss")
                final_val_loss = history[-1].get("val_loss")

        system_info["resumed_from"] = resumed_from
        system_info["starting_epoch"] = start_epoch

        for epoch_index in range(start_epoch, total_epochs):
            self._apply_epoch_learning_rate(epoch_index, total_epochs)
            epoch_start_time = time.time()
            train_metrics = self.train_epoch(epoch_index, total_epochs=total_epochs)
            val_metrics = self.evaluate(epoch_index) if self.val_loader is not None else {"loss": None, "batches": 0}
            epoch_duration = time.time() - epoch_start_time

            final_train_loss = train_metrics["loss"]
            final_val_loss = val_metrics["loss"]
            metric = final_val_loss if final_val_loss is not None else final_train_loss
            if metric is None:
                metric = final_train_loss

            history.append(
                {
                    "epoch": epoch_index + 1,
                    "train_loss": final_train_loss,
                    "val_loss": final_val_loss,
                    "epoch_duration_seconds": epoch_duration,
                    "learning_rate": self._current_learning_rate(),
                }
            )
            print(
                f"train-mim epoch={epoch_index + 1} summary "
                f"train_loss={final_train_loss:.6f} "
                f"val_loss={float('nan') if final_val_loss is None else final_val_loss:.6f} "
                f"lr={self._current_learning_rate():.6f} "
                f"duration={epoch_duration:.2f}s"
            )
            improved = metric is not None and metric < best_metric
            if improved:
                best_metric = metric
            system_info["last_completed_epoch"] = epoch_index + 1
            _write_training_metrics(
                output_dir,
                history=history,
                system_info=system_info,
                requested_precision=getattr(self.model, "requested_precision", self.requested_precision),
                resolved_precision=getattr(self.model, "resolved_precision", self.requested_precision),
                best_metric=best_metric,
            )
            curve_path = save_training_curves(history, output_dir, method_name="MIM")
            checkpoint = {
                "epoch": epoch_index + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
                "grad_scaler_state_dict": self.grad_scaler.state_dict() if self.grad_scaler.is_enabled() else None,
                "history": history,
                "best_metric": best_metric,
                "run_config": self.run_config,
                "rng_state": capture_rng_state(),
                "train_loader_generator_state": capture_loader_generator_state(self.train_loader),
            }
            save_training_checkpoint(checkpoint, output_dir, improved=improved)
            visualization_paths = self._save_visualization(output_dir, epoch_index + 1)
            print(f"[artifacts] Training curve: {curve_path}")
            for path in visualization_paths:
                if not path.name.endswith("_latest.png"):
                    print(f"[artifacts] Model output: {path}")

        end_time = time.time()
        system_info["end_time"] = end_time
        system_info["total_duration_seconds"] = end_time - start_time

        summary = SSLTrainingSummary(
            output_dir=str(output_dir),
            epochs=total_epochs,
            train_batches=len(self.train_loader),
            val_batches=0 if self.val_loader is None else len(self.val_loader),
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
        )
        _write_training_metrics(
            output_dir,
            history=history,
            system_info=system_info,
            requested_precision=getattr(self.model, "requested_precision", self.requested_precision),
            resolved_precision=getattr(self.model, "resolved_precision", self.requested_precision),
            best_metric=best_metric,
            summary=summary,
        )
        return summary


def _write_training_metrics(
    output_dir: Path,
    *,
    history: list[dict[str, Any]],
    system_info: dict[str, Any],
    requested_precision: str,
    resolved_precision: str,
    best_metric: float,
    summary: SSLTrainingSummary | None = None,
) -> None:
    atomic_write_text(output_dir / "metrics.csv", pd.DataFrame(history).to_csv(index=False))
    payload: dict[str, Any] = {
        "system_info": system_info,
        "history": history,
        "requested_precision": requested_precision,
        "resolved_precision": resolved_precision,
        "best_metric": None if best_metric == float("inf") else best_metric,
        "summary": None if summary is None else asdict(summary),
    }
    atomic_write_text(output_dir / "metrics.json", json.dumps(payload, indent=2))
