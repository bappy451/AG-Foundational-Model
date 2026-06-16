from __future__ import annotations

from pathlib import Path

import yaml


def test_dataset_registry_is_valid_and_has_unique_sources() -> None:
    registry_path = Path(__file__).resolve().parents[1] / "Dataset.yml"
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    sources = payload["sources"]

    assert payload["schema_version"] == 1
    assert len(sources) == 31
    assert len({source["id"] for source in sources}) == len(sources)
    assert all(source["status"] in payload["status_values"] for source in sources)
    assert all(str(source["url"]).startswith("https://") for source in sources)
