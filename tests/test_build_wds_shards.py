import csv
import tarfile
import zipfile
from io import BytesIO

from PIL import Image

from ag_foundation.data.build_wds_shards import build_shards, prepare_source_groups


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (128, 96), color).save(buf, format="PNG")
    return buf.getvalue()


def test_prepare_source_groups_keeps_archive_records_together():
    records = [
        {"path": "a.zip::one.jpg", "source_name": "a"},
        {"path": "b.zip::two.jpg", "source_name": "b"},
        {"path": "a.zip::three.jpg", "source_name": "a"},
    ]

    groups = dict(prepare_source_groups(records))

    assert set(groups) == {"a.zip", "b.zip"}
    assert [item[1] for item in groups["a.zip"]] == ["000000000", "000000002"]
    assert [item[1] for item in groups["b.zip"]] == ["000000001"]


def test_build_shards_from_catalog_zip_and_plain_file(tmp_path):
    pretraining_root = tmp_path / "pretraining"
    pretraining_root.mkdir()

    with zipfile.ZipFile(pretraining_root / "archive1.zip", "w") as zf:
        zf.writestr("valid_image1.png", _image_bytes((255, 0, 0)))

    with zipfile.ZipFile(pretraining_root / "archive2.zip", "w") as zf:
        zf.writestr("nested/path/valid_image2.png", _image_bytes((0, 255, 0)))

    plain_dir = pretraining_root / "plain"
    plain_dir.mkdir()
    plain_path = plain_dir / "valid_image3.png"
    plain_path.write_bytes(_image_bytes((0, 0, 255)))

    catalog_path = pretraining_root / "catalog_v2.csv"
    rows = [
        {"path": "archive1.zip::valid_image1.png", "group": "archive1", "source_name": "archive1"},
        {"path": "archive2.zip::nested/path/valid_image2.png", "group": "archive2", "source_name": "archive2"},
        {"path": "plain/valid_image3.png", "group": "plain", "source_name": "plain"},
    ]
    with catalog_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "group", "source_name"])
        writer.writeheader()
        writer.writerows(rows)

    output_prefix = tmp_path / "shards" / "dataset"
    build_shards(
        catalog_path=catalog_path,
        pretraining_root=pretraining_root,
        output_prefix=str(output_prefix),
        max_count=2,
        max_size=10_000_000,
        workers=2,
        progress_every=1,
    )

    shard_files = sorted((tmp_path / "shards").glob("*.tar"))
    assert len(shard_files) == 2

    jpg_members = []
    for shard_path in shard_files:
        with tarfile.open(shard_path, "r:") as tar:
            jpg_members.extend(name for name in tar.getnames() if name.endswith(".jpg"))

    assert len(jpg_members) == 3
    assert all(name.endswith(".jpg") for name in jpg_members)
