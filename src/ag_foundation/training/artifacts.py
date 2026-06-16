from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_snapshot(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination_path)
    try:
        try:
            os.link(source_path, temporary)
        except OSError:
            shutil.copyfile(source_path, temporary)
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


def save_training_checkpoint(
    checkpoint: dict[str, Any],
    output_dir: str | Path,
    *,
    improved: bool,
) -> tuple[Path, Path | None]:
    output_path = Path(output_dir)
    last_path = atomic_torch_save(checkpoint, output_path / "last.pt")
    best_path = atomic_snapshot(last_path, output_path / "best.pt") if improved else None
    return last_path, best_path


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported PyTorch releases
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {checkpoint_path}")
    return checkpoint


def _temporary_sibling(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
