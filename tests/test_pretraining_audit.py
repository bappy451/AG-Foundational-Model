from __future__ import annotations

import io
import json
import zipfile

import numpy as np
from PIL import Image

from ag_foundation.data.pretraining_audit import parse_manifest, run_audit


def _image_bytes(*, mode: str = "RGB", image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (12, 10), color=(20, 120, 40) if mode == "RGB" else 128).save(buffer, format=image_format)
    return buffer.getvalue()


def test_parse_manifest_extracts_names_urls_and_providers(tmp_path):
    manifest = tmp_path / "Dataset.txt"
    manifest.write_text(
        "\n".join(
            [
                "PlantVillage Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset",
                (
                    "Weed Detection in Soybean Crops: "
                    "https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops "
                    "(https://data.mendeley.com/datasets/3fmjm7ncc6/2)"
                ),
            ]
        ),
        encoding="utf-8",
    )

    entries = parse_manifest(manifest)

    assert [entry.name for entry in entries] == ["PlantVillage Dataset", "Weed Detection in Soybean Crops"]
    assert entries[1].providers == ("Kaggle", "Mendeley Data")
    assert len(entries[1].urls) == 2


def test_run_audit_counts_archives_directories_and_missing_manifest_entries(tmp_path):
    pretraining_root = tmp_path / "Pretraining"
    pretraining_root.mkdir()
    manifest = pretraining_root / "Dataset.txt"
    manifest.write_text(
        "\n".join(
            [
                "Plant Disease Detection: https://www.kaggle.com/datasets/example/plant-disease-detection",
                "GeoPlant: Spatial Plant Species Prediction Dataset: https://www.kaggle.com/datasets/picekl/geoplant",
                "Missing Dataset: https://www.kaggle.com/datasets/example/missing-dataset",
            ]
        ),
        encoding="utf-8",
    )

    archive_path = pretraining_root / "Plant Disease Detection.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Plant Disease Detection/train/healthy/image_001.jpg", _image_bytes(image_format="JPEG"))
        archive.writestr("Plant Disease Detection/train/rust/image_002.png", _image_bytes())
        archive.writestr("Plant Disease Detection/labels/image_001.txt", "0 0.5 0.5 0.2 0.2\n")
        archive.writestr("Plant Disease Detection/metadata.json", "{}")

    geospatial_dir = pretraining_root / "GeoPlant"
    geospatial_dir.mkdir()
    np.save(geospatial_dir / "sentinel_tile.npy", np.zeros((5, 8, 8), dtype=np.float32))

    output_dir = tmp_path / "reports"
    summary = run_audit(
        pretraining_root,
        dataset_list=manifest,
        output_dir=output_dir,
        sample_limit=4,
    )

    assert summary["aggregate"]["local_dataset_count"] == 2
    assert summary["aggregate"]["image_count"] == 3
    assert summary["aggregate"]["annotation_count"] == 2
    assert summary["aggregate"]["matched_manifest_entry_count"] == 2
    assert [entry["name"] for entry in summary["missing_manifest_entries"]] == ["Missing Dataset"]
    assert summary["aggregate"]["extension_counts"][".jpg"] == 1
    assert summary["aggregate"]["extension_counts"][".npy"] == 1
    assert "geospatial / remote sensing" in summary["aggregate"]["theme_counts"]
    assert "sampled multiband imagery" in summary["aggregate"]["modality_counts"]

    report = output_dir / "pretraining_dataset_audit.md"
    payload = json.loads((output_dir / "pretraining_dataset_audit.json").read_text(encoding="utf-8"))
    assert report.exists()
    assert "Missing Dataset" in report.read_text(encoding="utf-8")
    assert payload["outputs"]["markdown"] == str(report)
