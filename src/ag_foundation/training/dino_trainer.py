from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ag_foundation.training.artifacts import load_training_checkpoint, save_training_checkpoint
from ag_foundation.training.ssl_trainer import (
    SSLTrainingSummary,
    _build_grad_scaler,
    _move_optimizer_state_to_device,
    _move_ssl_batch_to_device,
    _write_training_metrics,
    select_torch_device,
)
from ag_foundation.training.state import (
    capture_loader_generator_state,
    capture_rng_state,
    restore_loader_generator_state,
    restore_rng_state,
)
from ag_foundation.training.visualization import save_dino_preview, save_training_curves


@dataclass(frozen=True)
class DINOAugmentationConfig:
    image_size: tuple[int, int]
    num_global_crops: int = 2
    num_local_crops: int = 2
    global_crop_scale: tuple[float, float] = (0.6, 1.0)
    local_crop_scale: tuple[float, float] = (0.3, 0.6)
    grayscale_prob: float = 0.2
    color_jitter_strength: float = 0.4


class DINOMultiCropAugmenter:
    def __init__(self, config: DINOAugmentationConfig, *, deterministic: bool = False) -> None:
        self.config = config
        self.deterministic = bool(deterministic)

    def __call__(self, images):
        import torch

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINOMultiCropAugmenter expects RGB image batches with shape [B, 3, H, W].")

        images = images.to(dtype=torch.float32)
        if self.deterministic:
            base = images.clamp(0.0, 1.0)
            return [base.clone() for _ in range(self.config.num_global_crops)]

        views = []
        for view_index in range(self.config.num_global_crops):
            blur_prob = 0.8 if view_index == 0 else 0.1
            solarize_prob = 0.2 if view_index == 1 else 0.0
            views.append(
                self._augment_batch(
                    images,
                    self.config.global_crop_scale,
                    blur_prob=blur_prob,
                    solarize_prob=solarize_prob,
                )
            )
        for _ in range(self.config.num_local_crops):
            views.append(
                self._augment_batch(
                    images,
                    self.config.local_crop_scale,
                    blur_prob=0.5,
                    solarize_prob=0.0,
                )
            )
        return views

    def global_views(self, images):
        import torch

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINOMultiCropAugmenter expects RGB image batches with shape [B, 3, H, W].")

        images = images.to(dtype=torch.float32)
        if self.deterministic:
            base = images.clamp(0.0, 1.0)
            return [base.clone() for _ in range(self.config.num_global_crops)]

        views = []
        for view_index in range(self.config.num_global_crops):
            blur_prob = 0.8 if view_index == 0 else 0.1
            solarize_prob = 0.2 if view_index == 1 else 0.0
            views.append(
                self._augment_batch(
                    images,
                    self.config.global_crop_scale,
                    blur_prob=blur_prob,
                    solarize_prob=solarize_prob,
                )
            )
        return views

    def _augment_batch(
        self,
        images,
        scale_range: tuple[float, float],
        *,
        blur_prob: float,
        solarize_prob: float,
    ):
        import torch

        transformed = [
            self._augment_single_image(image, scale_range, blur_prob=blur_prob, solarize_prob=solarize_prob)
            for image in images
        ]
        return torch.stack(transformed, dim=0)

    def _augment_single_image(
        self,
        image,
        scale_range: tuple[float, float],
        *,
        blur_prob: float,
        solarize_prob: float,
    ):
        import torch
        import torch.nn.functional as F

        image = image.float()
        _, height, width = image.shape
        image = self._random_resized_crop(image, scale_range, output_size=(height, width))
        if bool(torch.rand(()) < 0.5):
            image = torch.flip(image, dims=(2,))
        image = self._color_jitter(image)
        if bool(torch.rand(()) < self.config.grayscale_prob):
            image = image.mean(dim=0, keepdim=True).expand_as(image)
        if bool(torch.rand(()) < blur_prob):
            image = F.avg_pool2d(image.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        if bool(torch.rand(()) < solarize_prob):
            image = torch.where(image < 0.5, image, 1.0 - image)
        return image.clamp(0.0, 1.0)

    def _random_resized_crop(
        self,
        image,
        scale_range: tuple[float, float],
        *,
        output_size: tuple[int, int],
    ):
        import math

        import torch
        import torch.nn.functional as F

        _, height, width = image.shape
        area = float(height * width)
        min_scale, max_scale = scale_range
        for _ in range(10):
            scale = float(torch.empty(()).uniform_(min_scale, max_scale).item())
            aspect_ratio = float(torch.empty(()).uniform_(3.0 / 4.0, 4.0 / 3.0).item())
            target_area = area * scale
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
            if 0 < crop_h <= height and 0 < crop_w <= width:
                top = int(torch.randint(height - crop_h + 1, (1,)).item())
                left = int(torch.randint(width - crop_w + 1, (1,)).item())
                crop = image[:, top : top + crop_h, left : left + crop_w]
                if crop_h != output_size[0] or crop_w != output_size[1]:
                    crop = F.interpolate(
                        crop.unsqueeze(0),
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                return crop
        top = max(0, (height - output_size[0]) // 2)
        left = max(0, (width - output_size[1]) // 2)
        crop = image[:, top : top + output_size[0], left : left + output_size[1]]
        if crop.shape[-2:] != output_size:
            crop = F.interpolate(crop.unsqueeze(0), size=output_size, mode="bilinear", align_corners=False).squeeze(0)
        return crop

    def _color_jitter(self, image):
        import torch

        strength = float(self.config.color_jitter_strength)
        if strength <= 0.0:
            return image

        brightness = 1.0 + float(torch.empty(()).uniform_(-strength, strength).item())
        contrast = 1.0 + float(torch.empty(()).uniform_(-strength, strength).item())
        saturation = 1.0 + float(torch.empty(()).uniform_(-strength, strength).item())

        image = image * brightness
        channel_mean = image.mean(dim=(1, 2), keepdim=True)
        image = (image - channel_mean) * contrast + channel_mean
        grayscale = image.mean(dim=0, keepdim=True)
        image = grayscale + saturation * (image - grayscale)
        return image


class DINOTrainer:
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
        num_global_crops: int = 2,
        num_local_crops: int = 2,
        student_temperature: float = 0.1,
        center_momentum: float = 0.9,
        teacher_momentum_start: float = 0.996,
        teacher_momentum_end: float = 1.0,
        teacher_momentum_schedule: str = "constant",
        gram_anchor_weight: float | None = None,
        gram_anchor_max_tokens: int | None = None,
        augmentation_config: DINOAugmentationConfig | None = None,
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
        self.num_global_crops = max(1, int(num_global_crops))
        self.num_local_crops = max(0, int(num_local_crops))
        if self.num_global_crops + self.num_local_crops < 2:
            raise ValueError("DINO pretraining requires at least two crops in total.")
        self.student_temperature = float(student_temperature)
        self.center_momentum = float(center_momentum)
        self.teacher_momentum_start = float(teacher_momentum_start)
        self.teacher_momentum_end = float(teacher_momentum_end)
        self.teacher_momentum_schedule = str(teacher_momentum_schedule).strip().lower()
        if self.teacher_momentum_schedule not in {"constant", "cosine"}:
            raise ValueError("teacher_momentum_schedule must be 'constant' or 'cosine'.")
        model_gram_weight = float(getattr(self.model, "gram_anchor_weight", 0.0))
        self.gram_anchor_weight = model_gram_weight if gram_anchor_weight is None else float(gram_anchor_weight)
        if self.gram_anchor_weight < 0.0:
            raise ValueError("gram_anchor_weight must be non-negative.")
        model_gram_tokens = getattr(self.model, "gram_anchor_max_tokens", None)
        if gram_anchor_max_tokens is None:
            self.gram_anchor_max_tokens = None if model_gram_tokens is None else int(model_gram_tokens)
        else:
            if gram_anchor_max_tokens <= 0:
                raise ValueError("gram_anchor_max_tokens must be positive when provided.")
            self.gram_anchor_max_tokens = int(gram_anchor_max_tokens)
        if augmentation_config is None:
            augmentation_config = DINOAugmentationConfig(
                image_size=self.model.student_backbone.image_size,
                num_global_crops=self.num_global_crops,
                num_local_crops=self.num_local_crops,
            )
        self.augmenter = DINOMultiCropAugmenter(augmentation_config, deterministic=False)
        self.eval_augmenter = DINOMultiCropAugmenter(
            DINOAugmentationConfig(
                image_size=self.model.student_backbone.image_size,
                num_global_crops=self.num_global_crops,
                num_local_crops=0,
                global_crop_scale=(1.0, 1.0),
                local_crop_scale=(1.0, 1.0),
                grayscale_prob=0.0,
                color_jitter_strength=0.0,
            ),
            deterministic=True,
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

    def _teacher_momentum(self, epoch_index: int, step_index: int, total_epochs: int, num_batches: int) -> float:
        if self.teacher_momentum_schedule == "constant":
            return self.teacher_momentum_start
        total_steps = max(1, total_epochs * num_batches - 1)
        global_step = epoch_index * num_batches + step_index
        progress = min(max(float(global_step) / float(total_steps), 0.0), 1.0)
        blend = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.teacher_momentum_start + (self.teacher_momentum_end - self.teacher_momentum_start) * blend

    def _augment_batch(self, images):
        import torch

        student_adapted = self.model.adapt_student(images)
        teacher_adapted = self.model.adapt_teacher(images)

        initial_rng_state = torch.get_rng_state()
        student_views = self.augmenter(student_adapted)
        final_rng_state = torch.get_rng_state()
        torch.set_rng_state(initial_rng_state)
        try:
            teacher_views = self.augmenter.global_views(teacher_adapted)
        finally:
            torch.set_rng_state(final_rng_state)
        return student_views, teacher_views

    def _eval_views(self, images):
        student_adapted = self.model.adapt_student(images)
        teacher_adapted = self.model.adapt_teacher(images)
        return self.eval_augmenter(student_adapted), self.eval_augmenter.global_views(teacher_adapted)

    def train_step(self, batch: dict[str, Any], *, teacher_momentum: float) -> dict[str, float]:
        import torch

        self.model.train()
        self.model.teacher_adapter.eval()
        self.model.teacher_backbone.eval()
        self.model.teacher_head.eval()
        moved_batch = _move_ssl_batch_to_device(batch, self.device)
        images = moved_batch["image"]
        self.optimizer.zero_grad(set_to_none=True)
        student_views, teacher_views = self._augment_batch(images)

        with self._autocast_context():
            student_outputs = self.model.forward_student_views(student_views)
            teacher_outputs = self.model.forward_teacher_views(teacher_views)
            student_dense_views = None
            teacher_dense_views = None
            if self.gram_anchor_weight > 0.0:
                student_dense_views = self.model.student_dense_views(student_views[: len(teacher_views)])
                teacher_dense_views = self.model.teacher_dense_views(teacher_views)
            losses = self.model.dino_v3_loss(
                student_outputs,
                teacher_outputs,
                student_dense_views=student_dense_views,
                teacher_dense_views=teacher_dense_views,
                student_temperature=self.student_temperature,
                gram_anchor_weight=self.gram_anchor_weight,
                gram_anchor_max_tokens=self.gram_anchor_max_tokens,
            )
            loss = losses["loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite DINOv3 loss.")

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

        self.model.update_teacher(teacher_momentum)
        self.model.update_center(teacher_outputs, self.center_momentum)
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "loss": float(loss.detach().cpu()),
            "cls_loss": float(losses["cls_loss"].detach().cpu()),
            "gram_anchor_loss": float(losses["gram_anchor_loss"].detach().cpu()),
        }

    def train_epoch(self, epoch_index: int, total_epochs: int = 1) -> dict[str, float]:
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_index)

        total_loss = 0.0
        total_cls_loss = 0.0
        total_gram_anchor_loss = 0.0
        total_batches = 0
        num_batches = len(self.train_loader)
        last_loss = 0.0
        last_cls_loss = 0.0
        last_gram_anchor_loss = 0.0
        last_teacher_momentum = self.teacher_momentum_start

        for step_index, batch in enumerate(self.train_loader):
            teacher_momentum = self._teacher_momentum(epoch_index, step_index, total_epochs, num_batches)
            metrics = self.train_step(batch, teacher_momentum=teacher_momentum)
            current_loss = metrics["loss"]
            current_cls_loss = metrics["cls_loss"]
            current_gram_anchor_loss = metrics["gram_anchor_loss"]
            total_loss += current_loss
            total_cls_loss += current_cls_loss
            total_gram_anchor_loss += current_gram_anchor_loss
            total_batches += 1
            last_loss = current_loss
            last_cls_loss = current_cls_loss
            last_gram_anchor_loss = current_gram_anchor_loss
            last_teacher_momentum = teacher_momentum
            if self.progress_callback is not None:
                total_work = total_epochs * num_batches
                completed_work = (epoch_index * num_batches) + step_index
                avg_loss = total_loss / total_batches
                lr = self._current_learning_rate()
                detail = (
                    f"ep {epoch_index + 1}/{total_epochs} | "
                    f"batch {step_index + 1}/{num_batches} | "
                    f"loss {last_loss:.4f} (avg {avg_loss:.4f}) | "
                    f"cls {last_cls_loss:.4f} | gram {last_gram_anchor_loss:.4f} | "
                    f"lr {lr:.6f} | ema {teacher_momentum:.6f}"
                )
                self.progress_callback(completed_work, total_work, detail=detail)
            if (step_index + 1) % self.log_every == 0:
                print(
                    f"train-dino epoch={epoch_index + 1} step={step_index + 1}/{num_batches} "
                    f"loss={current_loss:.6f} cls={current_cls_loss:.6f} gram={current_gram_anchor_loss:.6f}"
                )

        return {
            "loss": float(total_loss / total_batches) if total_batches > 0 else float("nan"),
            "cls_loss": float(total_cls_loss / total_batches) if total_batches > 0 else float("nan"),
            "gram_anchor_loss": float(total_gram_anchor_loss / total_batches) if total_batches > 0 else float("nan"),
            "batches": total_batches,
            "teacher_momentum": float(last_teacher_momentum),
        }

    def evaluate(self, epoch_index: int) -> dict[str, float]:
        import torch

        if self.val_loader is None:
            return {"loss": float("nan"), "batches": 0}

        self.model.eval()
        self.model.teacher_adapter.eval()
        self.model.teacher_backbone.eval()
        self.model.teacher_head.eval()
        losses: list[float] = []
        cls_losses: list[float] = []
        gram_losses: list[float] = []
        with torch.no_grad():
            for batch in self.val_loader:
                moved_batch = _move_ssl_batch_to_device(batch, self.device)
                images = moved_batch["image"]
                student_views, teacher_views = self._eval_views(images)
                with self._autocast_context():
                    student_outputs = self.model.forward_student_views(student_views)
                    teacher_outputs = self.model.forward_teacher_views(teacher_views)
                    student_dense_views = None
                    teacher_dense_views = None
                    if self.gram_anchor_weight > 0.0:
                        student_dense_views = self.model.student_dense_views(student_views[: len(teacher_views)])
                        teacher_dense_views = self.model.teacher_dense_views(teacher_views)
                    losses_dict = self.model.dino_v3_loss(
                        student_outputs,
                        teacher_outputs,
                        student_dense_views=student_dense_views,
                        teacher_dense_views=teacher_dense_views,
                        student_temperature=self.student_temperature,
                        gram_anchor_weight=self.gram_anchor_weight,
                        gram_anchor_max_tokens=self.gram_anchor_max_tokens,
                    )
                    loss = losses_dict["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Encountered a non-finite DINOv3 validation loss.")
                losses.append(float(loss.detach().cpu()))
                cls_losses.append(float(losses_dict["cls_loss"].detach().cpu()))
                gram_losses.append(float(losses_dict["gram_anchor_loss"].detach().cpu()))

        mean_loss = float(sum(losses) / len(losses)) if losses else float("nan")
        mean_cls_loss = float(sum(cls_losses) / len(cls_losses)) if cls_losses else float("nan")
        mean_gram_loss = float(sum(gram_losses) / len(gram_losses)) if gram_losses else float("nan")
        print(
            f"train-dino epoch={epoch_index + 1} val_loss={mean_loss:.6f} "
            f"val_cls={mean_cls_loss:.6f} val_gram={mean_gram_loss:.6f}"
        )
        return {
            "loss": mean_loss,
            "cls_loss": mean_cls_loss,
            "gram_anchor_loss": mean_gram_loss,
            "batches": len(losses),
        }

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
            self.model.teacher_adapter.eval()
            self.model.teacher_backbone.eval()
            self.model.teacher_head.eval()
            with torch.no_grad(), self._autocast_context():
                adapted = self.model.adapt_student(moved_batch["image"])
                views, teacher_views = self._augment_batch(moved_batch["image"])
                student_features = [self.model.student_features(view) for view in views]
                teacher_features = [
                    self.model.teacher_features(view)
                    for view in teacher_views
                ]
            return save_dino_preview(
                adapted=adapted,
                views=views,
                student_features=student_features,
                teacher_features=teacher_features,
                output_dir=output_dir,
                epoch=epoch,
                num_global_crops=self.num_global_crops,
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
                    "loss": final_train_loss,
                    "cls_loss": train_metrics["cls_loss"],
                    "gram_anchor_loss": train_metrics["gram_anchor_loss"],
                    "train_loss": final_train_loss,
                    "train_cls_loss": train_metrics["cls_loss"],
                    "train_gram_anchor_loss": train_metrics["gram_anchor_loss"],
                    "val_loss": final_val_loss,
                    "val_cls_loss": val_metrics.get("cls_loss"),
                    "val_gram_anchor_loss": val_metrics.get("gram_anchor_loss"),
                    "epoch_duration_seconds": epoch_duration,
                    "learning_rate": self._current_learning_rate(),
                    "teacher_momentum": train_metrics["teacher_momentum"],
                }
            )
            val_cls_display = float("nan") if val_metrics.get("cls_loss") is None else val_metrics["cls_loss"]
            val_gram_display = (
                float("nan") if val_metrics.get("gram_anchor_loss") is None else val_metrics["gram_anchor_loss"]
            )
            print(
                f"train-dino epoch={epoch_index + 1} summary "
                f"train_loss={final_train_loss:.6f} "
                f"train_cls={train_metrics['cls_loss']:.6f} "
                f"train_gram={train_metrics['gram_anchor_loss']:.6f} "
                f"val_loss={float('nan') if final_val_loss is None else final_val_loss:.6f} "
                f"val_cls={val_cls_display:.6f} "
                f"val_gram={val_gram_display:.6f} "
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
            curve_path = save_training_curves(history, output_dir, method_name="DINOv3")
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


DINOv3Trainer = DINOTrainer
