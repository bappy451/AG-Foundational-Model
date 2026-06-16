from __future__ import annotations

import json
import platform
import shlex
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_write_text


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return value


def resolve_config_paths(flat_config: Mapping[str, Any], *, config_path: str | Path | None) -> dict[str, Any]:
    resolved = dict(flat_config)
    if config_path is None:
        return resolved

    base_dir = Path(config_path).expanduser().resolve().parent
    for key in ("data_root", "catalog_path", "output_dir", "resume_from"):
        value = resolved.get(key)
        if value in {None, ""}:
            continue
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            resolved[key] = str((base_dir / candidate).resolve())
    return resolved


def build_run_manifest(
    *,
    command_name: str,
    args: Mapping[str, Any],
    model: Any,
    train_loader: Any,
    val_loader: Any,
    output_dir: str | Path,
    command_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    dataset_summary = {
        "train": _summarize_loader(train_loader),
        "val": _summarize_loader(val_loader),
    }
    model_summary = _summarize_model(model)
    environment = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "device_count": _torch_device_count(),
        "torch": _torch_version("torch"),
        "torchvision": _torch_version("torchvision"),
        "timm": _torch_version("timm"),
    }
    git_info = _collect_git_info(_find_repo_root())
    command_text = shlex.join(["ag-foundation", command_name, *(command_argv or [])]).strip()

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "command": {
            "name": command_name,
            "argv": list(command_argv or []),
            "text": command_text,
        },
        "output_dir": str(output_path),
        "resolved_args": _jsonable(dict(args)),
        "data": dataset_summary,
        "model": model_summary,
        "environment": environment,
        "git": git_info,
    }


def write_run_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    serialized_manifest = json.dumps(_jsonable(dict(manifest)), indent=2, sort_keys=False)
    resolved_args = manifest.get("resolved_args", {})
    serialized_config = yaml.safe_dump(_jsonable(resolved_args), sort_keys=False)
    model_summary = manifest.get("model", {})
    model_text = str(model_summary.get("repr", ""))
    command = manifest.get("command", {})
    command_text = str(command.get("text", "")) + "\n"

    manifest_path = output_path / "run_manifest.json"
    _write_manifest_bundle(
        output_path,
        serialized_manifest=serialized_manifest,
        serialized_config=serialized_config,
        model_text=model_text,
        command_text=command_text,
    )

    generated_at = str(manifest.get("generated_at", datetime.now().astimezone().isoformat()))
    attempt_id = generated_at.replace(":", "-").replace("+", "_")
    _write_manifest_bundle(
        output_path / "attempts" / attempt_id,
        serialized_manifest=serialized_manifest,
        serialized_config=serialized_config,
        model_text=model_text,
        command_text=command_text,
    )

    return manifest_path


def _write_manifest_bundle(
    output_path: Path,
    *,
    serialized_manifest: str,
    serialized_config: str,
    model_text: str,
    command_text: str,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_path / "run_manifest.json", serialized_manifest)
    atomic_write_text(output_path / "resolved_config.yaml", serialized_config)
    atomic_write_text(output_path / "model_summary.txt", model_text)
    atomic_write_text(output_path / "command.txt", command_text)


def _summarize_loader(loader: Any) -> dict[str, Any]:
    if loader is None:
        return {"present": False}

    summary: dict[str, Any] = {
        "present": True,
        "type": type(loader).__name__,
        "batches": len(loader) if hasattr(loader, "__len__") else None,
    }
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        summary["dataset_type"] = type(dataset).__name__
        summary["samples"] = len(dataset) if hasattr(dataset, "__len__") else None
        root = getattr(dataset, "root", None)
        if root is not None:
            summary["root"] = str(root)
        records = getattr(dataset, "records", None)
        if records:
            groups = sorted({str(getattr(record, "group", "")) for record in records})
            summary["group_count"] = len(groups)
            summary["group_preview"] = groups[:20]
    return summary


def _summarize_model(model: Any) -> dict[str, Any]:
    total_parameters = 0
    trainable_parameters = 0
    parameters_fn = getattr(model, "parameters", None)
    if callable(parameters_fn):
        for parameter in parameters_fn():
            count = int(parameter.numel())
            total_parameters += count
            if parameter.requires_grad:
                trainable_parameters += count

    summary: dict[str, Any] = {
        "type": type(model).__name__,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "repr": repr(model),
    }
    if hasattr(model, "adapter"):
        adapter = model.adapter
        summary["adapter"] = {
            "type": type(adapter).__name__,
            "in_channels": getattr(adapter, "in_channels", None),
            "out_channels": getattr(adapter, "out_channels", None),
        }

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        backbone = getattr(model, "student_backbone", None)
    if backbone is not None:
        summary["backbone"] = {
            "type": type(backbone).__name__,
            "model_name": getattr(backbone, "model_name", None),
            "embed_dim": getattr(backbone, "embed_dim", None),
            "patch_size": getattr(backbone, "patch_size", None),
            "image_size": getattr(backbone, "image_size", None),
        }
    return summary


def _collect_git_info(project_root: Path | None) -> dict[str, Any]:
    if project_root is None or not (project_root / ".git").exists():
        return {"available": False}

    def _run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ("git", "-C", str(project_root), *args),
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return completed.stdout.strip() or None

    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--short")
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def _find_repo_root() -> Path | None:
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _torch_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _torch_device_count() -> int | None:
    try:
        import torch
    except Exception:
        return None

    if torch.cuda.is_available():
        return int(torch.cuda.device_count())
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 1
    return 0
