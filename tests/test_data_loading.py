from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from ag_foundation.data.dataset import (
    AgricultureImageDataset,
    ImageRecord,
    _split_records_by_group,
    create_dataset_catalog,
    get_dataloaders,
)


def _write_rgb(path: Path, *, size: tuple[int, int] = (32, 32), value: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_grouped_rgb_tiles(root: Path, groups: int = 4, tiles_per_group: int = 2) -> None:
    for group_index in range(groups):
        for tile_index in range(tiles_per_group):
            _write_rgb(
                root / f"source_{group_index}" / f"tile_{tile_index}.jpg",
                value=20 + group_index * 20 + tile_index,
            )


def _write_rgb_zip(path: Path, members: list[tuple[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for member_name, array in members:
            buffer = io.BytesIO()
            Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
            archive.writestr(member_name, buffer.getvalue())


def test_dataset_discovers_nested_images_in_deterministic_order(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "z_source" / "b.jpg")
    _write_rgb(tmp_path / "a_source" / "c.png")
    _write_rgb(tmp_path / "a_source" / "a.jpeg")

    dataset = AgricultureImageDataset(tmp_path, crop_size=16, augment=False)

    assert [record.uri for record in dataset.records] == [
        str((tmp_path / "a_source" / "a.jpeg").resolve()),
        str((tmp_path / "a_source" / "c.png").resolve()),
        str((tmp_path / "z_source" / "b.jpg").resolve()),
    ]
    sample = dataset[0]
    assert set(sample) == {"group", "image", "path"}
    assert sample["path"].endswith("a_source/a.jpeg") or sample["path"].endswith("a_source\\a.jpeg")
    assert sample["group"] == "a_source"
    assert sample["image"].shape == (3, 16, 16)
    assert sample["image"].device.type == "cpu"


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ],
)
def test_dataset_supports_requested_floating_precision(
    tmp_path: Path,
    precision: str,
    expected_dtype: torch.dtype,
) -> None:
    _write_rgb(tmp_path / "source" / "tile.jpg", value=128)

    image = AgricultureImageDataset(tmp_path, crop_size=16, precision=precision, augment=False)[0]["image"]

    assert image.dtype == expected_dtype
    assert image.device.type == "cpu"
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_dataset_loads_multiband_tiff_from_directory(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "source" / "cir.tif"
    path.parent.mkdir()
    data = np.stack([np.full((20, 24), value, dtype=np.uint16) for value in (0, 1, 2, 3)])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=24,
        height=20,
        count=4,
        dtype=data.dtype,
        transform=from_origin(0, 20, 1, 1),
    ) as dst:
        dst.write(data)

    image = AgricultureImageDataset(tmp_path, crop_size=16, channels=4, augment=False)[0]["image"]

    assert image.shape == (4, 16, 16)
    assert image.dtype == torch.float32


def test_dataset_discovers_and_loads_images_from_zip_archives(tmp_path: Path) -> None:
    array_a = np.full((24, 20, 3), 40, dtype=np.uint8)
    array_b = np.full((24, 20, 3), 80, dtype=np.uint8)
    archive_path = tmp_path / "tiles.zip"
    _write_rgb_zip(
        archive_path,
        [
            ("source_b/tile_1.png", array_b),
            ("source_a/tile_0.png", array_a),
        ],
    )

    dataset = AgricultureImageDataset(tmp_path, crop_size=16, augment=False)

    assert [record.uri for record in dataset.records] == [
        f"{archive_path.resolve()}::source_a/tile_0.png",
        f"{archive_path.resolve()}::source_b/tile_1.png",
    ]
    sample = dataset[0]
    assert sample["path"] == f"{archive_path.resolve()}::source_a/tile_0.png"
    assert sample["group"] == "source_a"
    assert sample["image"].shape == (3, 16, 16)


def test_dataset_reuses_open_zip_handle_between_samples(tmp_path: Path) -> None:
    archive_path = tmp_path / "tiles.zip"
    _write_rgb_zip(
        archive_path,
        [
            ("source_a/tile_0.png", np.full((24, 20, 3), 40, dtype=np.uint8)),
            ("source_a/tile_1.png", np.full((24, 20, 3), 80, dtype=np.uint8)),
        ],
    )
    dataset = AgricultureImageDataset(archive_path, crop_size=16, augment=False)

    _ = dataset[0]
    first_handle = dataset._zip_handles[archive_path.resolve()]
    _ = dataset[1]

    assert dataset._zip_handles[archive_path.resolve()] is first_handle
    assert first_handle.fp is not None
    dataset.close()
    assert first_handle.fp is None


def test_dataset_ignores_macos_archive_artifacts_inside_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "seedlings.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((24, 20, 3), 80, dtype=np.uint8)
    with zipfile.ZipFile(archive_path, "w") as archive:
        buffer = io.BytesIO()
        Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
        archive.writestr("archive/Cleavers/348.png", buffer.getvalue())
        archive.writestr("__MACOSX/archive/Cleavers/._348.png", b"not an image")

    dataset = AgricultureImageDataset(archive_path, crop_size=16, augment=False)

    assert len(dataset.records) == 1
    assert dataset[0]["path"] == f"{archive_path.resolve()}::archive/Cleavers/348.png"


def test_dataset_loads_multiband_tiff_from_zip_archives(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    data = np.stack([np.full((20, 24), value, dtype=np.uint16) for value in (0, 1, 2, 3)])
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=24,
            height=20,
            count=4,
            dtype=data.dtype,
            transform=from_origin(0, 20, 1, 1),
        ) as dataset:
            dataset.write(data)
        geotiff_bytes = memory_file.read()

    archive_path = tmp_path / "multiband.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("farm_a/cir.tif", geotiff_bytes)

    image = AgricultureImageDataset(archive_path, crop_size=16, channels=4, augment=False)[0]["image"]

    assert image.shape == (4, 16, 16)
    assert image.dtype == torch.float32


def test_get_dataloaders_support_zip_members_and_grouped_split(tmp_path: Path) -> None:
    archive_path = tmp_path / "tiles.zip"
    members: list[tuple[str, np.ndarray]] = []
    for group_index in range(5):
        for tile_index in range(2):
            members.append(
                (
                    f"source_{group_index}/tile_{tile_index}.png",
                    np.full((24, 20, 3), 20 + group_index * 20 + tile_index, dtype=np.uint8),
                )
            )
    _write_rgb_zip(archive_path, members)

    train_loader, val_loader = get_dataloaders(
        archive_path,
        batch_size=2,
        val_fraction=0.4,
        seed=27,
        crop_size=16,
        num_workers=0,
    )

    batch = next(iter(train_loader))
    assert batch["image"].shape == (2, 3, 16, 16)
    train_groups = {record.group for record in train_loader.dataset.records}
    val_groups = {record.group for record in val_loader.dataset.records}
    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)


def test_create_dataset_catalog_includes_zip_members(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "source_a" / "tile.jpg", value=32)
    _write_rgb_zip(
        tmp_path / "tiles.zip",
        [("source_b/tile.png", np.full((24, 20, 3), 48, dtype=np.uint8))],
    )
    catalog_path = tmp_path / "catalog.csv"

    create_dataset_catalog(tmp_path, catalog_path)

    lines = catalog_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "path,group"
    assert any(line.startswith("source_a/tile.jpg,") for line in lines[1:])
    assert any(line.startswith("tiles.zip::source_b/tile.png,") for line in lines[1:])
    assert str(tmp_path.resolve()) not in catalog_path.read_text(encoding="utf-8")


def test_get_dataloaders_can_use_catalog_path_for_zip_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "tiles.zip"
    members: list[tuple[str, np.ndarray]] = []
    for group_index in range(3):
        for tile_index in range(2):
            members.append(
                (
                    f"source_{group_index}/tile_{tile_index}.png",
                    np.full((24, 20, 3), 20 + group_index * 20 + tile_index, dtype=np.uint8),
                )
            )
    _write_rgb_zip(archive_path, members)
    catalog_path = tmp_path / "catalog.csv"
    create_dataset_catalog(archive_path, catalog_path)
    catalog_text = catalog_path.read_text(encoding="utf-8")
    assert "::source_0/tile_0.png" in catalog_text
    assert str(tmp_path.resolve()) not in catalog_text

    train_loader, val_loader = get_dataloaders(
        archive_path,
        batch_size=2,
        crop_size=16,
        num_workers=0,
        catalog_path=catalog_path,
        val_fraction=0.34,
    )

    assert len(train_loader.dataset) + len(val_loader.dataset) == 6
    assert next(iter(train_loader))["image"].shape == (2, 3, 16, 16)


def test_get_dataloaders_disable_pin_memory_without_cuda(tmp_path: Path, monkeypatch) -> None:
    _write_rgb(tmp_path / "source_a" / "tile.jpg", value=32)
    _write_rgb(tmp_path / "source_b" / "tile.jpg", value=64)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train_loader, val_loader = get_dataloaders(
        tmp_path,
        batch_size=1,
        crop_size=16,
        num_workers=0,
        val_fraction=0.5,
    )

    assert train_loader.pin_memory is False
    assert val_loader.pin_memory is False


def test_group_split_targets_sample_fraction_for_uneven_groups(tmp_path: Path) -> None:
    records = []
    for group, count in {"large": 8, "small_a": 1, "small_b": 1}.items():
        for index in range(count):
            path = tmp_path / group / f"{index}.png"
            records.append(
                ImageRecord(
                    uri=str(path),
                    group=group,
                    source_path=path,
                )
            )

    train_records, val_records = _split_records_by_group(
        records,
        val_fraction=0.2,
        seed=27,
    )

    assert len(train_records) == 8
    assert len(val_records) == 2
    assert {record.group for record in train_records}.isdisjoint(
        {record.group for record in val_records}
    )
