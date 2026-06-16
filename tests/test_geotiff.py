from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from ag_foundation.data.geotiff import (
    compute_tile_starts,
    iter_tile_windows,
    slice_geotiff_collection_to_files,
    slice_geotiff_to_files,
    slice_geotiff_to_memory,
    stitch_geotiff_tiles,
)


def _write_test_raster(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:26916",
        transform=from_origin(100.0, 200.0, 1.0, 1.0),
    ) as dst:
        dst.write(data)


def test_compute_tile_starts_aligns_last_tile_to_image_edge() -> None:
    assert compute_tile_starts(length=7, tile_size=4, stride=4) == [0, 3]
    assert compute_tile_starts(length=7, tile_size=4, stride=3) == [0, 3]
    assert compute_tile_starts(length=4, tile_size=4, stride=4) == [0]


def test_compute_tile_starts_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError):
        compute_tile_starts(length=7, tile_size=0, stride=1)
    with pytest.raises(ValueError):
        compute_tile_starts(length=7, tile_size=4, stride=5)


def test_iter_tile_windows_covers_full_image_without_trailing_sliver() -> None:
    windows = list(iter_tile_windows(width=7, height=5, tile_size=4, stride=4))

    assert windows == [
        (0, 0, 0, 0, 4, 4),
        (0, 1, 3, 0, 7, 4),
        (1, 0, 0, 1, 4, 5),
        (1, 1, 3, 1, 7, 5),
    ]


def test_slice_geotiff_to_memory_aligns_edge_tiles(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    data = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    _write_test_raster(raster_path, data)

    tiles = slice_geotiff_to_memory(raster_path, tile_size=4)

    assert len(tiles) == 4
    assert tiles[0].image_array.shape == (3, 4, 4)
    assert tiles[-1].image_array.shape == (3, 4, 4)
    np.testing.assert_array_equal(tiles[0].image_array[:, :4, :4], data[:, :4, :4])
    np.testing.assert_array_equal(tiles[-1].image_array, data[:, 1:5, 3:7])


def test_slice_geotiff_collection_to_files_handles_directory_input(tmp_path: Path) -> None:
    raster_dir = tmp_path / "geotiffs"
    raster_dir.mkdir()
    nested_dir = raster_dir / "nested"
    nested_dir.mkdir()

    _write_test_raster(raster_dir / "a.tif", np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4))
    _write_test_raster(nested_dir / "b.tif", np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4))

    output_dir = tmp_path / "tiles"
    written = slice_geotiff_collection_to_files(
        raster_dir,
        output_dir,
        tile_size=4,
        output_format="jpg",
    )

    assert len(written) == 2
    assert (output_dir / "a" / "slice_0_0.jpg").exists()
    assert (output_dir / "nested" / "b" / "slice_0_0.jpg").exists()


def test_slice_geotiff_collection_falls_back_when_process_pool_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raster_dir = tmp_path / "geotiffs"
    raster_dir.mkdir()
    _write_test_raster(raster_dir / "a.tif", np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4))
    _write_test_raster(raster_dir / "b.tif", np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4))

    def _raise_permission_error(*args, **kwargs):
        raise PermissionError("semaphores unavailable")

    monkeypatch.setattr("ag_foundation.data.geotiff.ProcessPoolExecutor", _raise_permission_error)

    written = slice_geotiff_collection_to_files(
        raster_dir,
        tmp_path / "tiles",
        tile_size=4,
        output_format="tif",
        workers="auto",
    )

    assert len(written) == 2


def test_stitch_geotiff_tiles_recreates_original_region(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    data = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    _write_test_raster(raster_path, data)

    slices_dir = tmp_path / "slices"
    slice_geotiff_to_files(raster_path, slices_dir, tile_size=4, output_format="tif")
    stitched_path = tmp_path / "stitched.tif"
    stitch_geotiff_tiles(slices_dir, stitched_path)

    with rasterio.open(stitched_path) as src:
        rebuilt = src.read()

    np.testing.assert_array_equal(rebuilt, data)


def test_rotated_geotiff_tiles_preserve_affine_geometry(tmp_path: Path) -> None:
    raster_path = tmp_path / "rotated.tif"
    transform = Affine(1.0, 0.2, 100.0, 0.1, -1.0, 200.0)
    data = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=5,
        width=7,
        count=3,
        dtype=data.dtype,
        crs="EPSG:26916",
        transform=transform,
    ) as dst:
        dst.write(data)

    tiles = slice_geotiff_to_memory(raster_path, tile_size=4)

    assert tiles[-1].transform == window_transform(Window(3, 1, 4, 4), transform)

    slices_dir = tmp_path / "rotated-slices"
    slice_geotiff_to_files(raster_path, slices_dir, tile_size=4, output_format="tif")
    stitched_path = tmp_path / "rotated-stitched.tif"
    stitch_geotiff_tiles(slices_dir, stitched_path)
    with rasterio.open(stitched_path) as src:
        assert src.transform == transform
        np.testing.assert_array_equal(src.read(), data)
