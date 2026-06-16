from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def create_demo_dataset(
    output_dir: str | Path,
    *,
    image_size: int = 64,
    samples_per_group: int = 6,
    multispectral_channels: int = 5,
    seed: int = 27,
) -> dict[str, object]:
    if image_size < 16:
        raise ValueError("image_size must be at least 16.")
    if samples_per_group < 2:
        raise ValueError("samples_per_group must be at least 2.")
    if multispectral_channels < 4:
        raise ValueError("multispectral_channels must be at least 4.")

    root = Path(output_dir).expanduser().resolve()
    rgb_root = root / "rgb"
    multispectral_root = root / "multispectral"
    rng = np.random.default_rng(seed)
    groups = ("healthy_canopy", "stressed_canopy", "mixed_field", "seedlings")

    for group_index, group in enumerate(groups):
        for sample_index in range(samples_per_group):
            base = _make_pattern(
                image_size=image_size,
                group_index=group_index,
                sample_index=sample_index,
                rng=rng,
            )
            rgb_path = rgb_root / group / f"sample_{sample_index:03d}.png"
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((base * 255.0).round().astype(np.uint8), mode="RGB").save(rgb_path)

            extra_bands = [
                np.clip(
                    base[..., 1] * (0.55 + 0.08 * band_index)
                    + base[..., 0] * (0.35 - 0.04 * band_index)
                    + rng.normal(0.0, 0.015, base.shape[:2]),
                    0.0,
                    1.0,
                )
                for band_index in range(multispectral_channels - 3)
            ]
            multispectral = np.concatenate(
                [np.moveaxis(base, -1, 0), np.stack(extra_bands, axis=0)],
                axis=0,
            ).astype(np.float32)
            npy_path = multispectral_root / group / f"sample_{sample_index:03d}.npy"
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, multispectral, allow_pickle=False)

    summary: dict[str, object] = {
        "output_dir": str(root),
        "rgb_root": str(rgb_root),
        "multispectral_root": str(multispectral_root),
        "groups": list(groups),
        "samples_per_group": samples_per_group,
        "image_size": image_size,
        "multispectral_channels": multispectral_channels,
        "total_rgb_images": len(groups) * samples_per_group,
        "total_multispectral_images": len(groups) * samples_per_group,
        "seed": seed,
    }
    (root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _make_pattern(
    *,
    image_size: int,
    group_index: int,
    sample_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    y, x = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    x /= max(1, image_size - 1)
    y /= max(1, image_size - 1)
    phase = 0.4 * sample_index + 0.8 * group_index
    texture = 0.5 + 0.5 * np.sin((x * (4 + group_index) + y * 3 + phase) * np.pi)
    canopy = np.exp(-((x - (0.3 + 0.12 * group_index)) ** 2 + (y - 0.52) ** 2) / 0.08)
    noise = rng.normal(0.0, 0.025, (image_size, image_size))

    red = np.clip(0.15 + 0.20 * texture + 0.25 * canopy + noise, 0.0, 1.0)
    green = np.clip(0.22 + 0.45 * canopy + 0.18 * texture + noise, 0.0, 1.0)
    blue = np.clip(0.10 + 0.18 * (1.0 - canopy) + 0.12 * texture + noise, 0.0, 1.0)
    if group_index == 1:
        red = np.clip(red + 0.18 * canopy, 0.0, 1.0)
        green = np.clip(green - 0.10 * canopy, 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1).astype(np.float32)
