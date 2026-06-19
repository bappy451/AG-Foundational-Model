from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ag_foundation.cli import main
from ag_foundation.command_logging import parse_command_logging


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((24, 24, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def test_cli_help_lists_primary_commands(capsys) -> None:
    main([])

    output = capsys.readouterr().out
    assert "train-mim" in output
    assert "train-dino" in output
    assert "audit-pretraining-data" in output
    assert "create-catalog" in output
    assert "create-demo-data" in output
    assert "slice-geotiffs" in output


def test_cli_create_catalog_writes_csv(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "source_a" / "tile_a.jpg", value=32)
    _write_rgb(tmp_path / "source_b" / "tile_b.jpg", value=64)
    output_path = tmp_path / "catalog.csv"

    main(
        [
            "create-catalog",
            "--data-root",
            str(tmp_path),
            "--output-path",
            str(output_path),
        ]
    )

    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "path,group"
    assert len(lines) == 3


def test_cli_create_demo_data_writes_rgb_and_multispectral_sets(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"

    main(
        [
            "create-demo-data",
            "--output-dir",
            str(output_dir),
            "--image-size",
            "32",
            "--samples-per-group",
            "2",
            "--multispectral-channels",
            "5",
        ]
    )

    assert len(list((output_dir / "rgb").rglob("*.png"))) == 8
    assert len(list((output_dir / "multispectral").rglob("*.npy"))) == 8
    assert (output_dir / "dataset_summary.json").exists()


def test_cli_can_log_direct_invocations(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"
    log_path = tmp_path / "direct-command.log"

    main(
        [
            "create-demo-data",
            "--output-dir",
            str(output_dir),
            "--samples-per-group",
            "2",
            "--log-file",
            str(log_path),
        ],
        enable_logging=True,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert "Command Log" in log_text
    assert "create-demo-data" in log_text
    assert "Created demo RGB data" in log_text


def test_data_package_does_not_eagerly_require_torch() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src_root), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(src_root)]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ag_foundation.data; "
                "assert 'ag_foundation.data.dataset' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_direct_cli_default_log_follows_current_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    _, config = parse_command_logging(["--no-log"])

    assert config.log_file == tmp_path / "command.log"


def test_direct_cli_rejects_missing_log_file_value() -> None:
    with pytest.raises(SystemExit, match="requires a path"):
        parse_command_logging(["--log-file", "--no-log"])


def test_train_wrapper_reports_missing_ml_dependencies(tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env sh\nprintf 'torch, timm\\n'\nexit 1\n", encoding="utf-8")
    fake_python.chmod(0o755)
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            "bash",
            str(project_root / "scripts" / "train_mim.sh"),
            "--python",
            str(fake_python),
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "run"),
            "--no-log",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "missing required module(s): torch, timm" in completed.stderr
    assert "pip install -e '.[dev,ml]'" in completed.stderr
