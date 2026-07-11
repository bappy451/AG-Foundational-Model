from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any, Callable

from .ssl_trainer import (
    _optimizer_steps_for_batches,
    _is_accumulation_boundary,
    select_torch_device,
    _move_optimizer_state_to_device,
    _build_grad_scaler,
)


def _accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class ClassificationTrainer:
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
        gradient_accumulation_steps: int = 1,
        log_every: int = 10,
        progress_callback: Any | None = None,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_lr_schedule = epoch_lr_schedule
        self.grad_clip_norm = grad_clip_norm
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.log_every = max(1, int(log_every))
        self.requested_precision = precision
        self.progress_callback = progress_callback
        self.label_smoothing = label_smoothing
        self.device = torch.device(device) if device is not None else torch.device(select_torch_device())
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        
        self.model = self.model.to(self.device)
        self._trainable_parameter_items = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.optimizer_step_count = 0
        self.grad_scaler = _build_grad_scaler(
            torch,
            enabled=self.device.type == "cuda" and precision == "fp16",
        )
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

    def _autocast_context(self):
        import contextlib
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
        for name, parameter in self._trainable_parameter_items:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Encountered a non-finite gradient in parameter '{name}'.")

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _finalize_optimizer_step(self) -> None:
        if self.grad_scaler.is_enabled():
            self.grad_scaler.unscale_(self.optimizer)
        self._validate_gradients()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._trainable_parameters(),
                self.grad_clip_norm,
                error_if_nonfinite=True,
            )
        if self.grad_scaler.is_enabled():
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = dict(batch)
        if "image" in moved and hasattr(moved["image"], "to"):
            moved["image"] = moved["image"].to(device=self.device, non_blocking=True)
        if "label" in moved and hasattr(moved["label"], "to"):
            moved["label"] = moved["label"].to(device=self.device, non_blocking=True)
        return moved

    def _apply_epoch_learning_rate(self, epoch_index: int, total_epochs: int) -> None:
        if self.epoch_lr_schedule is None:
            return
        multiplier = self.epoch_lr_schedule(epoch_index, total_epochs)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * multiplier

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        moved_batch = self._move_batch_to_device(batch)
        images = moved_batch["image"]
        labels = moved_batch["label"]

        with self._autocast_context():
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing, weight=self.class_weights)

        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite classification loss.")

        scaled_loss = loss / float(self.gradient_accumulation_steps)
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._finalize_optimizer_step()
        self.optimizer_step_count += 1
        
        acc1, acc5 = _accuracy(logits, labels, topk=(1, min(5, logits.size(1))))
        
        return {
            "loss": float(loss.detach().cpu()),
            "acc1": acc1,
            "acc5": acc5,
        }

    def train_epoch(self, epoch_index: int, total_epochs: int = 1) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        total_acc1 = 0.0
        total_batches = 0
        num_batches = len(self.train_loader)
        num_optimizer_steps = _optimizer_steps_for_batches(num_batches, self.gradient_accumulation_steps)
        optimizer_steps = 0

        for step_index, batch in enumerate(self.train_loader, start=1):
            moved_batch = self._move_batch_to_device(batch)
            images = moved_batch["image"]
            labels = moved_batch["label"]

            with self._autocast_context():
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing, weight=self.class_weights)

            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered a non-finite classification loss.")

            current_loss = float(loss.detach().cpu())
            acc1, _ = _accuracy(logits, labels, topk=(1, min(5, logits.size(1))))
            
            total_loss += current_loss
            total_acc1 += acc1
            total_batches += 1
            
            avg_loss = total_loss / total_batches
            avg_acc1 = total_acc1 / total_batches

            scaled_loss = loss / float(self.gradient_accumulation_steps)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = _is_accumulation_boundary(step_index, num_batches, self.gradient_accumulation_steps)
            if should_step:
                self._finalize_optimizer_step()
                self.optimizer_step_count += 1
                optimizer_steps += 1
                self.optimizer.zero_grad(set_to_none=True)

            lr = self._current_learning_rate()

            if self.progress_callback is not None:
                self.progress_callback(
                    step_index, 
                    num_batches, 
                    description=f"train-cls Epoch [{epoch_index + 1}/{total_epochs}]",
                    detail=f"loss: {current_loss:.4f} | acc1: {acc1:.1f}% | lr: {lr:.6f}"
                )
            elif step_index % self.log_every == 0:
                print(
                    f"train-cls epoch={epoch_index + 1}/{total_epochs} | "
                    f"batch={step_index}/{num_batches} | "
                    f"loss={current_loss:.4f} (avg={avg_loss:.4f}) | "
                    f"acc1={acc1:.1f}% (avg={avg_acc1:.1f}%) | "
                    f"lr={lr:.6f}"
                )

        return {
            "loss": float(total_loss / total_batches) if total_batches > 0 else float("nan"),
            "acc1": float(total_acc1 / total_batches) if total_batches > 0 else float("nan"),
            "batches": total_batches,
            "optimizer_steps": optimizer_steps,
        }

    def evaluate(self, epoch_index: int) -> dict[str, float]:
        if self.val_loader is None:
            return {"loss": float("nan"), "acc1": float("nan"), "batches": 0}

        self.model.eval()
        losses: list[float] = []
        acc1s: list[float] = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                moved_batch = self._move_batch_to_device(batch)
                images = moved_batch["image"]
                labels = moved_batch["label"]

                with self._autocast_context():
                    logits = self.model(images)
                    loss = F.cross_entropy(logits, labels, weight=self.class_weights)

                if not torch.isfinite(loss):
                    raise FloatingPointError("Encountered a non-finite classification validation loss.")
                
                losses.append(float(loss.detach().cpu()))
                acc1, _ = _accuracy(logits, labels, topk=(1, min(5, logits.size(1))))
                acc1s.append(acc1)

        mean_loss = float(sum(losses) / len(losses)) if losses else float("nan")
        mean_acc1 = float(sum(acc1s) / len(acc1s)) if acc1s else float("nan")
        print(f"train-cls epoch={epoch_index + 1} val_loss={mean_loss:.4f} val_acc1={mean_acc1:.1f}%")
        return {"loss": mean_loss, "acc1": mean_acc1, "batches": len(losses)}
