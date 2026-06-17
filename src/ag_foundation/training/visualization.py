from __future__ import annotations

import csv
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def save_training_curves(
    history: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    method_name: str,
) -> Path | None:
    if not history:
        return None

    plt = _load_pyplot()
    figures_dir = _figures_dir(output_dir)
    epochs = [int(record.get("epoch", index + 1)) for index, record in enumerate(history)]
    train_losses = [_optional_float(record.get("train_loss")) for record in history]
    val_losses = [_optional_float(record.get("val_loss")) for record in history]
    learning_rates = [_optional_float(record.get("learning_rate")) for record in history]

    fig, (loss_axis, lr_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True,
    )
    loss_axis.plot(epochs, train_losses, color="#0b5d7a", marker="o", label="Train loss")
    if any(value is not None for value in val_losses):
        loss_axis.plot(epochs, val_losses, color="#c84b31", marker="s", label="Validation loss")
    loss_axis.set_title(f"{method_name} pretraining metrics")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend()

    lr_axis.plot(epochs, learning_rates, color="#b7791f", marker=".", label="Learning rate")
    lr_axis.set_xlabel("Epoch")
    lr_axis.set_ylabel("Learning rate")
    lr_axis.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path = figures_dir / "training_metrics.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_mim_preview(
    outputs: dict[str, Any],
    output_dir: str | Path,
    *,
    epoch: int,
    patch_size: tuple[int, int],
    max_samples: int,
) -> list[Path]:
    import torch
    import torch.nn.functional as F

    adapted = outputs["adapted"].detach().float().cpu().clamp(0.0, 1.0)
    target_patches = outputs["target_patches"].detach().float().cpu()
    reconstructed_patches = outputs["reconstructed_patches"].detach().float().cpu()
    mask = outputs["mask"].detach().bool().cpu()
    sample_count = min(max(1, int(max_samples)), adapted.shape[0])
    height, width = adapted.shape[-2:]

    mask_expanded = mask.unsqueeze(-1)
    masked_patches = torch.where(mask_expanded, torch.zeros_like(target_patches), target_patches)
    completed_patches = torch.where(mask_expanded, reconstructed_patches, target_patches)

    def _fold(patches):
        return F.fold(
            patches.transpose(1, 2),
            output_size=(height, width),
            kernel_size=patch_size,
            stride=patch_size,
        ).clamp(0.0, 1.0)

    masked_images = _fold(masked_patches)
    completed_images = _fold(completed_patches)
    plt = _load_pyplot()
    figures_dir = _figures_dir(output_dir)
    fig, axes = plt.subplots(sample_count, 3, figsize=(10, max(3, sample_count * 3)))
    axes = np.asarray(axes, dtype=object).reshape(sample_count, 3)
    column_titles = ("Adapted input", "Masked input", "Reconstruction")

    for row_index in range(sample_count):
        images = (adapted[row_index], masked_images[row_index], completed_images[row_index])
        for column_index, image in enumerate(images):
            axes[row_index, column_index].imshow(_tensor_to_image(image))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(column_titles[column_index])

    fig.suptitle(f"MIM model output after epoch {epoch}")
    fig.tight_layout()
    epoch_path = figures_dir / f"mim_reconstruction_epoch_{epoch:04d}.png"
    latest_path = figures_dir / "mim_reconstruction_latest.png"
    fig.savefig(epoch_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(epoch_path, latest_path)
    return [epoch_path, latest_path]


def save_dino_preview(
    *,
    adapted: Any,
    views: Sequence[Any],
    student_features: Sequence[Any],
    teacher_features: Sequence[Any],
    output_dir: str | Path,
    epoch: int,
    num_global_crops: int,
    max_samples: int,
) -> list[Path]:
    import torch
    import torch.nn.functional as F

    adapted_cpu = adapted.detach().float().cpu().clamp(0.0, 1.0)
    view_tensors = [view.detach().float().cpu().clamp(0.0, 1.0) for view in views]
    sample_count = min(max(1, int(max_samples)), adapted_cpu.shape[0])
    displayed_views = view_tensors[: min(len(view_tensors), 4)]
    columns = 1 + len(displayed_views)

    plt = _load_pyplot()
    figures_dir = _figures_dir(output_dir)
    fig, axes = plt.subplots(sample_count, columns, figsize=(3 * columns, max(3, sample_count * 3)))
    axes = np.asarray(axes, dtype=object).reshape(sample_count, columns)
    titles = ["Adapted input"]
    for view_index in range(len(displayed_views)):
        kind = "Global" if view_index < num_global_crops else "Local"
        titles.append(f"{kind} view {view_index + 1}")

    for row_index in range(sample_count):
        images = [adapted_cpu[row_index], *(view[row_index] for view in displayed_views)]
        for column_index, image in enumerate(images):
            axes[row_index, column_index].imshow(_tensor_to_image(image))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(titles[column_index])

    fig.suptitle(f"DINOv3 multi-crop inputs after epoch {epoch}")
    fig.tight_layout()
    views_path = figures_dir / f"dino_views_epoch_{epoch:04d}.png"
    latest_views_path = figures_dir / "dino_views_latest.png"
    fig.savefig(views_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(views_path, latest_views_path)

    student_matrix = torch.stack([feature[0].detach().float().cpu() for feature in student_features], dim=0)
    teacher_matrix = torch.stack([feature[0].detach().float().cpu() for feature in teacher_features], dim=0)
    similarities = F.normalize(student_matrix, dim=-1) @ F.normalize(teacher_matrix, dim=-1).T
    similarity_array = similarities.numpy()
    student_labels = [f"student_{index + 1}" for index in range(similarity_array.shape[0])]
    teacher_labels = [f"teacher_{index + 1}" for index in range(similarity_array.shape[1])]

    fig, axis = plt.subplots(figsize=(max(5, len(teacher_labels) * 1.3), max(4, len(student_labels) * 0.8)))
    image = axis.imshow(similarity_array, vmin=-1.0, vmax=1.0, cmap="RdYlBu_r", aspect="auto")
    axis.set_xticks(range(len(teacher_labels)), teacher_labels)
    axis.set_yticks(range(len(student_labels)), student_labels)
    axis.set_title(f"DINOv3 feature cosine similarity after epoch {epoch}")
    for row_index in range(similarity_array.shape[0]):
        for column_index in range(similarity_array.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{similarity_array[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="Cosine similarity")
    fig.tight_layout()
    similarity_path = figures_dir / f"dino_similarity_epoch_{epoch:04d}.png"
    latest_similarity_path = figures_dir / "dino_similarity_latest.png"
    fig.savefig(similarity_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(similarity_path, latest_similarity_path)

    diagnostics_dir = Path(output_dir) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics_dir / f"dino_similarity_epoch_{epoch:04d}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["view", *teacher_labels])
        for label, row in zip(student_labels, similarity_array):
            writer.writerow([label, *[f"{value:.8f}" for value in row]])

    return [
        views_path,
        latest_views_path,
        similarity_path,
        latest_similarity_path,
        csv_path,
    ]


def _load_pyplot():
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "ag_foundation_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _figures_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tensor_to_image(tensor: Any) -> np.ndarray:
    array = tensor.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        return array[..., 0]
    return array


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
