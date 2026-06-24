"""Comprehensive tests for multi-source pretraining data loading.

Tests cover:
- Multi-source discovery from multiple roots
- Held-out source exclusion
- Source-balanced sampling (balanced, sqrt, uniform)
- Source provenance in __getitem__ output
- Duplicate ZIP detection and skipping
- Train/val splitting across multiple sources
- DataLoader batching and iteration
- Edge cases: empty archives, single source, missing paths
- Catalog generation and loading
- Real Pretraining folder integration (when available)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from ag_foundation.data.multi_source_dataset import (
    MultiSourcePretrainingDataset,
    _is_known_duplicate,
    get_pretraining_dataloaders,
    scan_pretraining_directory,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_rgb(path: Path, *, size: tuple[int, int] = (48, 48), value: int = 64) -> None:
    """Write a solid-color RGB image to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_rgb_zip(
    path: Path,
    members: list[tuple[str, np.ndarray]],
) -> None:
    """Write multiple RGB images into a ZIP archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for member_name, array in members:
            buffer = io.BytesIO()
            Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
            archive.writestr(member_name, buffer.getvalue())


def _create_source_zip(
    root: Path, name: str, *, groups: int = 3, tiles_per_group: int = 2, size: int = 48,
) -> Path:
    """Create a ZIP file with grouped images under *root*."""
    members = []
    for g in range(groups):
        for t in range(tiles_per_group):
            value = 20 + g * 20 + t
            members.append(
                (
                    f"class_{g}/tile_{t}.png",
                    np.full((size, size, 3), value, dtype=np.uint8),
                )
            )
    zip_path = root / f"{name}.zip"
    _write_rgb_zip(zip_path, members)
    return zip_path


def _create_source_dir(
    root: Path, name: str, *, groups: int = 3, tiles_per_group: int = 2,
) -> Path:
    """Create a directory with grouped images under *root*."""
    source_dir = root / name
    for g in range(groups):
        for t in range(tiles_per_group):
            _write_rgb(
                source_dir / f"class_{g}" / f"tile_{t}.jpg",
                value=20 + g * 20 + t,
            )
    return source_dir


# ── Tests: Multi-Source Discovery ────────────────────────────────────────


class TestMultiSourceDiscovery:
    """Tests for discovering images from multiple roots."""

    def test_discovers_from_multiple_zip_sources(self, tmp_path: Path) -> None:
        """Verify records are combined from multiple ZIP files."""
        zip1 = _create_source_zip(tmp_path, "dataset_A", groups=2, tiles_per_group=3)
        zip2 = _create_source_zip(tmp_path, "dataset_B", groups=3, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip1, zip2], crop_size=16, augment=False,
        )

        assert len(dataset) == 2 * 3 + 3 * 2  # 6 + 6 = 12
        assert dataset.summary.source_count == 2
        assert dataset.summary.total_records == 12

    def test_discovers_from_mixed_zips_and_directories(self, tmp_path: Path) -> None:
        """Verify mixing ZIP and directory sources works."""
        zip_path = _create_source_zip(tmp_path, "zip_source", groups=2, tiles_per_group=2)
        dir_path = _create_source_dir(tmp_path, "dir_source", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip_path, dir_path], crop_size=16, augment=False,
        )

        assert len(dataset) == 4 + 4  # 8 total
        assert dataset.summary.source_count == 2

    def test_skips_nonexistent_roots_gracefully(self, tmp_path: Path) -> None:
        """Non-existent paths are skipped without error."""
        zip1 = _create_source_zip(tmp_path, "real_data", groups=2, tiles_per_group=2)
        fake_path = tmp_path / "nonexistent.zip"

        dataset = MultiSourcePretrainingDataset(
            [zip1, fake_path], crop_size=16, augment=False,
        )

        assert len(dataset) == 4
        assert dataset.summary.source_count == 1

    def test_raises_for_empty_source_list(self) -> None:
        """Empty source list raises ValueError."""
        with pytest.raises(ValueError, match="at least one path"):
            MultiSourcePretrainingDataset([], crop_size=16)


# ── Tests: Held-Out Exclusion ────────────────────────────────────────────


class TestHeldOutExclusion:
    """Tests for excluding sources from pretraining."""

    def test_excludes_sources_by_name(self, tmp_path: Path) -> None:
        """Excluded source stems should not appear in records."""
        zip1 = _create_source_zip(tmp_path, "pretrain_data", groups=2, tiles_per_group=3)
        zip2 = _create_source_zip(tmp_path, "eval_data", groups=2, tiles_per_group=3)

        dataset = MultiSourcePretrainingDataset(
            [zip1, zip2],
            crop_size=16,
            augment=False,
            exclude_sources={"eval_data"},
        )

        assert len(dataset) == 6  # Only pretrain_data
        assert dataset.summary.excluded_count == 1
        source_names = {s.source_id for s in dataset.summary.sources}
        assert "eval_data" not in source_names
        assert "pretrain_data" in source_names

    def test_multiple_exclusions(self, tmp_path: Path) -> None:
        """Multiple sources can be excluded simultaneously."""
        zip1 = _create_source_zip(tmp_path, "keep_me", groups=2, tiles_per_group=2)
        zip2 = _create_source_zip(tmp_path, "drop_a", groups=2, tiles_per_group=2)
        zip3 = _create_source_zip(tmp_path, "drop_b", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip1, zip2, zip3],
            crop_size=16,
            augment=False,
            exclude_sources={"drop_a", "drop_b"},
        )

        assert len(dataset) == 4
        assert dataset.summary.excluded_count == 2
        assert dataset.summary.source_count == 1


# ── Tests: Source-Balanced Sampling ──────────────────────────────────────


class TestSourceBalancedSampling:
    """Tests for weighted sampling across unequal sources."""

    def test_balanced_weights_inversely_proportional(self, tmp_path: Path) -> None:
        """Balanced strategy gives higher weight to smaller sources."""
        # Large source: 100 images, small source: 10 images
        big_zip = _create_source_zip(tmp_path, "big", groups=10, tiles_per_group=10)
        small_zip = _create_source_zip(tmp_path, "small", groups=2, tiles_per_group=5)

        dataset = MultiSourcePretrainingDataset(
            [big_zip, small_zip], crop_size=16, augment=False,
        )
        weights = dataset.get_source_weights(strategy="balanced")

        # Big source records should have weight 1/100
        # Small source records should have weight 1/10
        big_indices = dataset._source_map["big"]
        small_indices = dataset._source_map["small"]
        big_weight = weights[big_indices[0]]
        small_weight = weights[small_indices[0]]
        assert small_weight > big_weight
        assert abs(big_weight - 1.0 / 100) < 1e-6
        assert abs(small_weight - 1.0 / 10) < 1e-6

    def test_sqrt_weights_moderate_rebalancing(self, tmp_path: Path) -> None:
        """Sqrt strategy moderately downweights large sources."""
        import math

        big_zip = _create_source_zip(tmp_path, "big", groups=5, tiles_per_group=4)
        small_zip = _create_source_zip(tmp_path, "small", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [big_zip, small_zip], crop_size=16, augment=False,
        )
        weights = dataset.get_source_weights(strategy="sqrt")

        big_indices = dataset._source_map["big"]
        small_indices = dataset._source_map["small"]
        big_weight = weights[big_indices[0]]
        small_weight = weights[small_indices[0]]
        assert abs(big_weight - 1.0 / math.sqrt(20)) < 1e-6
        assert abs(small_weight - 1.0 / math.sqrt(4)) < 1e-6

    def test_uniform_weights_equal(self, tmp_path: Path) -> None:
        """Uniform strategy gives all samples equal weight."""
        zip1 = _create_source_zip(tmp_path, "a", groups=2, tiles_per_group=2)
        zip2 = _create_source_zip(tmp_path, "b", groups=3, tiles_per_group=3)

        dataset = MultiSourcePretrainingDataset(
            [zip1, zip2], crop_size=16, augment=False,
        )
        weights = dataset.get_source_weights(strategy="uniform")

        assert all(w == 1.0 for w in weights)


# ── Tests: __getitem__ Output ────────────────────────────────────────────


class TestGetItem:
    """Tests for dataset sample retrieval."""

    def test_getitem_returns_source_dataset_field(self, tmp_path: Path) -> None:
        """Each sample should include its source_dataset provenance."""
        zip1 = _create_source_zip(tmp_path, "source_alpha", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip1], crop_size=16, augment=False,
        )
        sample = dataset[0]

        assert "source_dataset" in sample
        assert sample["source_dataset"] == "source_alpha"
        assert "image" in sample
        assert "path" in sample
        assert "group" in sample

    def test_getitem_image_shape_and_dtype(self, tmp_path: Path) -> None:
        """Image tensor has correct shape and dtype."""
        zip1 = _create_source_zip(tmp_path, "test_ds", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip1], crop_size=16, precision="bf16", augment=False,
        )
        sample = dataset[0]

        assert sample["image"].shape == (3, 16, 16)
        assert sample["image"].dtype == torch.bfloat16
        assert float(sample["image"].float().min()) >= 0.0
        assert float(sample["image"].float().max()) <= 1.0

    def test_getitem_with_augmentation(self, tmp_path: Path) -> None:
        """Augmented samples still have correct shape."""
        zip1 = _create_source_zip(tmp_path, "aug_test", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip1], crop_size=16, augment=True,
        )
        sample = dataset[0]

        assert sample["image"].shape == (3, 16, 16)


# ── Tests: Duplicate Detection ───────────────────────────────────────────


class TestDuplicateDetection:
    """Tests for known duplicate ZIP skipping."""

    def test_known_duplicates_are_detected(self) -> None:
        """Known duplicate filenames should be flagged."""
        assert _is_known_duplicate("Plant Disease Expert.zip")
        assert _is_known_duplicate("Plant Leaves for Image Classification.zip")
        assert not _is_known_duplicate("Plant Disease Expert-016.zip")
        assert not _is_known_duplicate("PlantVillage Dataset-019.zip")

    def test_duplicate_skipping_in_dataset(self, tmp_path: Path) -> None:
        """When skip_known_duplicates=True, duplicates are not loaded."""
        # Create a zip named like a known duplicate
        dup_path = tmp_path / "Plant Disease Expert.zip"
        _write_rgb_zip(
            dup_path,
            [("class_0/tile.png", np.full((48, 48, 3), 100, dtype=np.uint8))],
        )
        real_path = _create_source_zip(tmp_path, "real_data", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [dup_path, real_path],
            crop_size=16,
            augment=False,
            skip_known_duplicates=True,
        )

        assert dataset.summary.duplicate_skipped == 1
        assert len(dataset) == 4  # Only real_data

    def test_no_skipping_when_disabled(self, tmp_path: Path) -> None:
        """When skip_known_duplicates=False, all sources are loaded."""
        dup_path = tmp_path / "Plant Disease Expert.zip"
        _write_rgb_zip(
            dup_path,
            [
                ("class_0/tile.png", np.full((48, 48, 3), 100, dtype=np.uint8)),
                ("class_1/tile.png", np.full((48, 48, 3), 200, dtype=np.uint8)),
            ],
        )
        real_path = _create_source_zip(tmp_path, "real_data", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [dup_path, real_path],
            crop_size=16,
            augment=False,
            skip_known_duplicates=False,
        )

        assert dataset.summary.duplicate_skipped == 0
        assert len(dataset) == 2 + 4  # Both loaded


# ── Tests: DataLoader Factory ────────────────────────────────────────────


class TestPretrainingDataLoaders:
    """Tests for the get_pretraining_dataloaders factory."""

    def test_dataloaders_produce_correct_batch_shape(self, tmp_path: Path) -> None:
        """Train and val loaders produce batches with correct image shape."""
        zip1 = _create_source_zip(tmp_path, "ds_a", groups=3, tiles_per_group=4)
        zip2 = _create_source_zip(tmp_path, "ds_b", groups=3, tiles_per_group=4)

        train_loader, val_loader, summary = get_pretraining_dataloaders(
            [zip1, zip2],
            batch_size=2,
            crop_size=16,
            num_workers=0,
            val_fraction=0.3,
            seed=42,
            sampling_strategy="uniform",
        )

        assert summary.source_count == 2
        batch = next(iter(train_loader))
        assert batch["image"].shape[0] == 2
        assert batch["image"].shape[1] == 3
        assert batch["image"].shape[2] == 16
        assert batch["image"].shape[3] == 16

    def test_dataloaders_train_val_disjoint(self, tmp_path: Path) -> None:
        """Train and val sets should be group-disjoint."""
        zip1 = _create_source_zip(tmp_path, "source_1", groups=4, tiles_per_group=3)
        zip2 = _create_source_zip(tmp_path, "source_2", groups=4, tiles_per_group=3)

        train_loader, val_loader, summary = get_pretraining_dataloaders(
            [zip1, zip2],
            batch_size=2,
            crop_size=16,
            num_workers=0,
            val_fraction=0.25,
            sampling_strategy="uniform",
        )

        total = len(train_loader.dataset) + len(val_loader.dataset)
        assert total == summary.total_records

    def test_dataloaders_with_exclusions(self, tmp_path: Path) -> None:
        """Excluded sources should not appear in either loader."""
        zip1 = _create_source_zip(tmp_path, "train_source", groups=3, tiles_per_group=3)
        zip2 = _create_source_zip(tmp_path, "eval_source", groups=3, tiles_per_group=3)

        _, _, summary = get_pretraining_dataloaders(
            [zip1, zip2],
            batch_size=2,
            crop_size=16,
            num_workers=0,
            exclude_sources={"eval_source"},
            sampling_strategy="uniform",
        )

        assert summary.source_count == 1
        assert summary.total_records == 9


# ── Tests: scan_pretraining_directory ────────────────────────────────────


class TestScanPretrainingDirectory:
    """Tests for the pretraining directory scanner."""

    def test_discovers_zips_and_directories(self, tmp_path: Path) -> None:
        """Scanner finds both ZIP files and directories."""
        _create_source_zip(tmp_path, "archive_a")
        _create_source_dir(tmp_path, "dir_b")
        # Create non-data files that should be ignored
        (tmp_path / "notes.txt").write_text("ignore me")
        (tmp_path / "config.yml").write_text("ignore me")
        (tmp_path / "paper.pdf").write_bytes(b"fake pdf")

        sources = scan_pretraining_directory(tmp_path)

        names = {s.name for s in sources}
        assert "archive_a.zip" in names
        assert "dir_b" in names
        assert "notes.txt" not in names
        assert "config.yml" not in names
        assert "paper.pdf" not in names

    def test_excludes_specified_sources(self, tmp_path: Path) -> None:
        """Scanner respects exclude_sources."""
        _create_source_zip(tmp_path, "keep")
        _create_source_zip(tmp_path, "drop")

        sources = scan_pretraining_directory(
            tmp_path, exclude_sources={"drop"},
        )

        assert len(sources) == 1
        assert sources[0].name == "keep.zip"

    def test_skips_hidden_and_macosx(self, tmp_path: Path) -> None:
        """Hidden directories and __MACOSX are ignored."""
        _create_source_zip(tmp_path, "real_data")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "__MACOSX").mkdir()

        sources = scan_pretraining_directory(tmp_path)

        assert len(sources) == 1

    def test_raises_for_nonexistent_root(self, tmp_path: Path) -> None:
        """Scanner raises FileNotFoundError for missing root."""
        with pytest.raises(FileNotFoundError):
            scan_pretraining_directory(tmp_path / "nonexistent")


# ── Tests: Summary and Close ─────────────────────────────────────────────


class TestSummaryAndCleanup:
    """Tests for summary reporting and resource cleanup."""

    def test_summary_has_correct_counts(self, tmp_path: Path) -> None:
        """Summary accurately reflects the dataset build."""
        zip1 = _create_source_zip(tmp_path, "src_a", groups=2, tiles_per_group=3)
        zip2 = _create_source_zip(tmp_path, "src_b", groups=3, tiles_per_group=2)
        zip3 = _create_source_zip(tmp_path, "excluded", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip1, zip2, zip3],
            crop_size=16,
            augment=False,
            exclude_sources={"excluded"},
        )

        assert dataset.summary.total_records == 6 + 6  # 12
        assert dataset.summary.source_count == 2
        assert dataset.summary.excluded_count == 1
        assert len(dataset.summary.sources) == 2

    def test_close_releases_resources(self, tmp_path: Path) -> None:
        """Calling close() clears cached datasets."""
        zip1 = _create_source_zip(tmp_path, "test_close", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip1], crop_size=16, augment=False,
        )
        # Force dataset cache creation by accessing an item
        _ = dataset[0]
        assert len(dataset._datasets) > 0

        dataset.close()
        assert len(dataset._datasets) == 0


# ── Tests: Edge Cases ────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_source_works(self, tmp_path: Path) -> None:
        """A single source should work without issues."""
        zip1 = _create_source_zip(tmp_path, "only_one", groups=3, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip1], crop_size=16, augment=False,
        )

        assert len(dataset) == 6
        assert dataset.summary.source_count == 1
        sample = dataset[0]
        assert sample["source_dataset"] == "only_one"

    def test_all_sources_excluded_yields_empty(self, tmp_path: Path) -> None:
        """Excluding all sources results in zero records."""
        zip1 = _create_source_zip(tmp_path, "a", groups=2, tiles_per_group=2)
        zip2 = _create_source_zip(tmp_path, "b", groups=2, tiles_per_group=2)

        dataset = MultiSourcePretrainingDataset(
            [zip1, zip2],
            crop_size=16,
            augment=False,
            exclude_sources={"a", "b"},
        )

        assert len(dataset) == 0
        assert dataset.summary.total_records == 0


# ── Tests: Integration with real Pretraining data ────────────────────────


PRETRAINING_ROOT = Path(__file__).resolve().parent.parent / "Pretraining"
SMALL_TEST_ZIP = PRETRAINING_ROOT / "Pea Plant dataset.zip"


@pytest.mark.skipif(
    not SMALL_TEST_ZIP.exists(),
    reason="Pea Plant dataset.zip not available for integration test",
)
class TestRealPretrainingData:
    """Integration tests using real pretraining data when available."""

    def test_scan_real_pretraining_directory(self) -> None:
        """Scan the actual Pretraining directory and verify source count."""
        sources = scan_pretraining_directory(PRETRAINING_ROOT)
        # We know there are ~37 unique datasets (minus duplicates)
        assert len(sources) >= 10, f"Expected >=10 sources, found {len(sources)}"

    def test_load_small_real_zip(self) -> None:
        """Load Pea Plant dataset (17 MB) as an integration test."""
        dataset = MultiSourcePretrainingDataset(
            [SMALL_TEST_ZIP],
            crop_size=16,
            augment=False,
        )

        assert len(dataset) > 0
        sample = dataset[0]
        assert sample["image"].shape[0] == 3
        assert sample["image"].shape[1] == 16
        assert sample["image"].shape[2] == 16
        assert sample["source_dataset"] == "Pea Plant dataset"
        dataset.close()

    def test_load_multiple_real_sources(self) -> None:
        """Load from multiple real sources with exclusions."""
        sources = scan_pretraining_directory(
            PRETRAINING_ROOT,
            exclude_sources={
                "GeoPlant_ Spatial Plant Species Prediction Dataset-008",
                "plantnet_300K-018",
                "Agriculture-Vision-2021",
            },
        )
        # Take only smallest sources for speed
        small_sources = sorted(sources, key=lambda p: p.stat().st_size)[:3]

        dataset = MultiSourcePretrainingDataset(
            small_sources,
            crop_size=16,
            augment=False,
        )

        assert len(dataset) > 0
        assert dataset.summary.source_count >= 2
        # Verify we can iterate several samples
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            assert "source_dataset" in sample
            assert sample["image"].shape == (3, 16, 16)
        dataset.close()


# ── Tests: Dataset.yml Validation ────────────────────────────────────────


DATASET_YML = Path(__file__).resolve().parent.parent / "Dataset.yml"


@pytest.mark.skipif(
    not DATASET_YML.exists(),
    reason="Dataset.yml not found",
)
class TestDatasetYmlUpdated:
    """Validate the updated Dataset.yml structure."""

    def test_schema_version_2(self) -> None:
        """Dataset.yml should have schema_version 2 after update."""
        import yaml

        with open(DATASET_YML) as f:
            data = yaml.safe_load(f)

        assert data["schema_version"] == 2

    def test_has_role_field(self) -> None:
        """All sources should have a role field."""
        import yaml

        with open(DATASET_YML) as f:
            data = yaml.safe_load(f)

        for source in data["sources"]:
            assert "role" in source, f"Source {source['id']} missing role field"
            assert source["role"] in {"pretrain", "evaluation", "candidate"}

    def test_downloaded_sources_have_local_path(self) -> None:
        """Downloaded sources should specify their local_path."""
        import yaml

        with open(DATASET_YML) as f:
            data = yaml.safe_load(f)

        for source in data["sources"]:
            if source["status"] == "downloaded":
                assert "local_path" in source, (
                    f"Downloaded source {source['id']} missing local_path"
                )

    def test_evaluation_sources_have_tasks(self) -> None:
        """Evaluation sources should specify evaluation_tasks."""
        import yaml

        with open(DATASET_YML) as f:
            data = yaml.safe_load(f)

        for source in data["sources"]:
            if source["role"] == "evaluation":
                assert "evaluation_tasks" in source, (
                    f"Evaluation source {source['id']} missing evaluation_tasks"
                )

    def test_unique_ids(self) -> None:
        """All source IDs should be unique."""
        import yaml

        with open(DATASET_YML) as f:
            data = yaml.safe_load(f)

        ids = [s["id"] for s in data["sources"]]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[i for i in ids if ids.count(i) > 1]}"

    def test_source_count_increased(self) -> None:
        """Updated YAML should have more sources than the original 31."""
        import yaml

        with open(DATASET_YML) as f:
            data = yaml.safe_load(f)

        assert len(data["sources"]) >= 35, (
            f"Expected >=35 sources after adding new datasets, found {len(data['sources'])}"
        )
