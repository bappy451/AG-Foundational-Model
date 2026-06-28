from __future__ import annotations

import functools
import io
import math
import random
import tarfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import warnings

try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    pass

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy"})
SUPPORTED_ARCHIVE_EXTENSIONS = frozenset({".zip"})
SUPPORTED_TAR_EXTENSIONS = frozenset({".tar", ".tar.gz", ".tgz"})

# Path tokens that identify ground-truth masks / label maps — excluded from training.
_GT_TOKENS = (
    "/masks/", "/mask/", "_mask.", "_masks.", "mask.",
    "/labels/", "/label/", "_label.", "label.",
    "_gt.", "_groundtruth.", "/gt/", "/annotations/",
    "_boundary.", "_plant.", "_weed.",
)


def _is_ground_truth_path(path: str) -> bool:
    """Return True if *path* looks like a mask, label, or ground-truth file."""
    p = path.lower().replace("\\", "/")
    return any(tok in p for tok in _GT_TOKENS)
PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@functools.lru_cache(maxsize=1024)
def _resolve_source_path(source_text: str, base_dir_str: str) -> Path:
    source_path = Path(source_text).expanduser()
    if not source_path.is_absolute():
        source_path = (Path(base_dir_str) / source_path).resolve()
    return source_path


@dataclass(frozen=True)
class ImageRecord:
    uri: str
    group: str
    source_path: Path
    archive_chain: tuple[str, ...] = ()

    @property
    def suffix(self) -> str:
        if self.archive_chain:
            return Path(self.archive_chain[-1]).suffix.lower()
        return self.source_path.suffix.lower()


class AgricultureImageDataset(Dataset[dict[str, Any]]):
    """Load agricultural imagery from folders, GeoTIFFs, NPY arrays, and ZIP archives."""

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 224,
        channels: int | None = None,
        precision: str = "fp32",
        augment: bool = True,
        files: Sequence[ImageRecord | str | Path] | None = None,
        catalog_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Imagery root does not exist: {self.root}")
        if crop_size <= 0:
            raise ValueError("crop_size must be a positive integer.")
        if channels is not None and channels <= 0:
            raise ValueError("channels must be a positive integer or None.")
        if precision not in PRECISION_DTYPES:
            raise ValueError("precision must be fp32, fp16, or bf16.")

        self.crop_size = int(crop_size)
        self.channels = channels
        self.dtype = PRECISION_DTYPES[precision]
        self.precision = precision
        self.augment = bool(augment)
        self.catalog_path = Path(catalog_path).expanduser().resolve() if catalog_path else None
        self._zip_handles: dict[Path, zipfile.ZipFile] = {}
        self.records = self._resolve_records(files)
        try:
            self._expected_channels = channels or self._inspect_channel_count(self.records[0])
        finally:
            # Do not carry an archive descriptor opened during inspection into DataLoader workers.
            self.close()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.records[index]
        if type(item) is tuple:
            record = self._record_from_uri(item[0], item[1]) # type: ignore
        else:
            record = item # type: ignore
            
        image = self._load_image(record)
        actual_channels = int(image.shape[0])
        if self.channels is not None and actual_channels != self.channels:
            if actual_channels == 1 and self.channels == 3:
                # Broadcast 1-channel grayscale or single-band GIS data to 3 channels
                image = image.expand(3, -1, -1)
                actual_channels = 3
            else:
                raise ValueError(
                    f"Expected {self.channels} channels for '{record.uri}', found {actual_channels}."
                )
        if self.channels is None and actual_channels != self._expected_channels:
            raise ValueError(
                "Inconsistent channel count: "
                f"expected {self._expected_channels}, found {actual_channels} in '{record.uri}'."
            )

        _, height, width = image.shape
        if height < self.crop_size or width < self.crop_size:
            # Zero-pad the image on the right/bottom to reach crop_size.
            # This preserves small but valid images (e.g. tiny GIS tiles)
            # instead of discarding them.
            pad_h = max(0, self.crop_size - height)
            pad_w = max(0, self.crop_size - width)
            # F.pad order is (left, right, top, bottom)
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        if self.augment:
            image = self._random_crop(image)
            if _random_bool():
                image = torch.flip(image, dims=(2,))
            if _random_bool():
                image = torch.flip(image, dims=(1,))
            image = _color_jitter(image)
        else:
            image = self._center_crop(image)

        return {
            "image": image.to(dtype=self.dtype),
            "path": record.uri,
            "group": record.group,
        }

    def close(self) -> None:
        for archive in self._zip_handles.values():
            archive.close()
        self._zip_handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_zip_handles"] = {}
        return state

    def __del__(self) -> None:
        handles = getattr(self, "_zip_handles", None)
        if handles:
            self.close()

    def _resolve_records(
        self,
        files: Sequence[ImageRecord | tuple[str, str] | str | Path] | None,
    ) -> list[ImageRecord | tuple[str, str]]:
        if files is not None:
            records: list[ImageRecord | tuple[str, str]] = [self._coerce_record(item) for item in files]
        elif self.catalog_path is not None:
            if not self.catalog_path.exists():
                raise FileNotFoundError(f"Catalog file not found: {self.catalog_path}")
            frame = pd.read_csv(self.catalog_path)
            if "path" not in frame.columns:
                raise ValueError(f"Catalog CSV {self.catalog_path} must contain a 'path' column.")
            frame["group"] = frame["group"].fillna("unknown")
            # Convert paths and groups directly to tuples to save ~30GB of RAM and massive startup delays
            records = list(zip(frame["path"].astype(str).tolist(), frame["group"].astype(str).tolist()))
        else:
            records = self._scan_root()

        # Sort only if it's not a tuple list (catalog is already sorted/randomized by builder)
        if records and isinstance(records[0], ImageRecord):
            records = sorted(records, key=lambda record: record.uri) # type: ignore

        if not records:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(f"No supported imagery found under {self.root}. Supported: {supported}")
        return records

    def _coerce_record(self, item: ImageRecord | tuple[str, str] | str | Path) -> ImageRecord | tuple[str, str]:
        if isinstance(item, (ImageRecord, tuple)):
            return item
        return self._record_from_uri(str(item))

    def _record_from_uri(self, uri: str, group: str | None = None) -> ImageRecord:
        parts = uri.split("::")
        source_text = parts[0]
        if source_text == "" and self.root.is_file():
            source_path = self.root
        else:
            base_dir = self.root if self.root.is_dir() else self.root.parent
            source_path = _resolve_source_path(source_text, str(base_dir))
        archive_chain = tuple(parts[1:])
        if group is None:
            group = _derive_group(source_path, archive_chain, root=self.root)
        return ImageRecord(
            uri=_compose_uri(source_path, archive_chain),
            group=group,
            source_path=source_path,
            archive_chain=archive_chain,
        )

    def _scan_root(self) -> list[ImageRecord]:
        if self.root.is_file():
            return list(self._scan_path(self.root))

        records: list[ImageRecord] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                records.extend(self._scan_path(path))
        return records

    def _scan_path(self, path: Path) -> list[ImageRecord]:
        if _should_skip_member_parts(path.parts):
            return []
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            resolved_path = path.resolve()
            return [
                ImageRecord(
                    uri=str(resolved_path),
                    group=_derive_group(resolved_path, (), root=self.root),
                    source_path=resolved_path,
                )
            ]
        if suffix in SUPPORTED_ARCHIVE_EXTENSIONS:
            return self._scan_zip_file(path.resolve())
        return []

    def _scan_zip_file(self, archive_path: Path) -> list[ImageRecord]:
        with zipfile.ZipFile(archive_path) as archive:
            return self._scan_zip_archive(archive_path, archive, chain_prefix=())

    def _scan_zip_archive(
        self,
        archive_path: Path,
        archive: zipfile.ZipFile,
        *,
        chain_prefix: tuple[str, ...],
    ) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        for member_name in sorted(name for name in archive.namelist() if not name.endswith("/")):
            if _should_skip_member_name(member_name):
                continue
            suffix = Path(member_name).suffix.lower()
            if suffix in SUPPORTED_IMAGE_EXTENSIONS:
                chain = chain_prefix + (member_name,)
                records.append(
                    ImageRecord(
                        uri=_compose_uri(archive_path, chain),
                        group=_derive_group(archive_path, chain, root=self.root),
                        source_path=archive_path,
                        archive_chain=chain,
                    )
                )
            elif suffix in SUPPORTED_ARCHIVE_EXTENSIONS:
                nested_bytes = archive.read(member_name)
                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_archive:
                    records.extend(
                        self._scan_zip_archive(
                            archive_path,
                            nested_archive,
                            chain_prefix=chain_prefix + (member_name,),
                        )
                    )
        return records

    def _inspect_channel_count(self, record: Any) -> int:
        if type(record) is tuple:
            record = self._record_from_uri(record[0], record[1])
        suffix = record.suffix
        if suffix in {".jpg", ".jpeg", ".png"}:
            return 3
        if suffix in {".tif", ".tiff"}:
            payload = self._read_record_bytes(record) if record.archive_chain else None
            return _inspect_tiff_channels(record.uri, record.source_path, payload)
        array = self._load_npy_array(record)
        return int(_array_to_chw(array, requested_channels=None, path=record.uri).shape[0])

    def _load_image(self, record: ImageRecord) -> torch.Tensor:
        suffix = record.suffix
        if suffix in {".jpg", ".jpeg", ".png"}:
            if record.archive_chain:
                with Image.open(io.BytesIO(self._read_record_bytes(record))) as source:
                    array = np.asarray(source.convert("RGB")).copy()
            else:
                with Image.open(record.source_path) as source:
                    array = np.asarray(source.convert("RGB")).copy()
            chw = np.moveaxis(array, -1, 0)
        elif suffix in {".tif", ".tiff"}:
            payload = self._read_record_bytes(record) if record.archive_chain else None
            chw = _load_tiff_array(record.uri, record.source_path, payload)
        elif suffix == ".npy":
            chw = _array_to_chw(
                self._load_npy_array(record),
                requested_channels=self.channels,
                path=record.uri,
            )
        else:
            raise ValueError(f"Unsupported image extension '{suffix}' for '{record.uri}'.")
        return _normalize_image_array(chw, path=record.uri)

    def _load_npy_array(self, record: ImageRecord) -> np.ndarray:
        if record.archive_chain:
            buffer = io.BytesIO(self._read_record_bytes(record))
            buffer.seek(0)
            return np.load(buffer, allow_pickle=False)
        return np.load(record.source_path, allow_pickle=False)

    def _read_record_bytes(self, record: ImageRecord) -> bytes:
        if not record.archive_chain:
            return record.source_path.read_bytes()

        archive_name = record.source_path.name.lower()
        inner_path = record.archive_chain[-1]  # The path inside the archive

        # TAR / TAR.GZ archives (PlantCLEF 2024/2025 and Agriculture-Vision)
        if archive_name.endswith(".tar.gz") or archive_name.endswith(".tar"):
            return _read_tar_member(record.source_path, inner_path)

        # ZIP archives (legacy path — nested ZIP supported)
        current_bytes: bytes | None = None
        current_archive_path = record.source_path
        for depth, member_name in enumerate(record.archive_chain):
            if depth == 0:
                current_bytes = self._get_zip_handle(current_archive_path).read(member_name)
            else:
                assert current_bytes is not None
                with zipfile.ZipFile(io.BytesIO(current_bytes)) as archive:
                    current_bytes = archive.read(member_name)
        assert current_bytes is not None
        return current_bytes

    def _get_zip_handle(self, archive_path: Path) -> zipfile.ZipFile:
        resolved_path = archive_path.resolve()
        archive = self._zip_handles.get(resolved_path)
        if archive is None or archive.fp is None:
            archive = zipfile.ZipFile(resolved_path)
            self._zip_handles[resolved_path] = archive
        return archive

    def _random_crop(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        max_top = height - self.crop_size
        max_left = width - self.crop_size
        top = int(torch.randint(max_top + 1, (1,)).item()) if max_top else 0
        left = int(torch.randint(max_left + 1, (1,)).item()) if max_left else 0
        return image[:, top : top + self.crop_size, left : left + self.crop_size]

    def _center_crop(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        top = (height - self.crop_size) // 2
        left = (width - self.crop_size) // 2
        return image[:, top : top + self.crop_size, left : left + self.crop_size]


def get_dataloaders(
    root: str | Path,
    batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 27,
    crop_size: int = 224,
    channels: int | None = None,
    precision: str = "fp32",
    num_workers: int = 4,
    prefetch_factor: int = 2,
    catalog_path: str | Path | None = None,
    train_augment: bool = True,
    val_augment: bool = False,
) -> tuple[DataLoader, DataLoader]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be a positive integer.")

    catalog = AgricultureImageDataset(
        root,
        crop_size=crop_size,
        channels=channels,
        precision=precision,
        augment=False,
        catalog_path=catalog_path,
    )
    train_records, val_records = _split_records_by_group(
        catalog.records,
        val_fraction=val_fraction,
        seed=seed,
    )
    catalog.close()
    train_dataset = AgricultureImageDataset(
        root,
        files=train_records,
        crop_size=crop_size,
        channels=channels,
        precision=precision,
        augment=train_augment,
    )
    val_dataset = AgricultureImageDataset(
        root,
        files=val_records,
        crop_size=crop_size,
        channels=channels,
        precision=precision,
        augment=val_augment,
    )

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    train_sampler = None
    val_sampler = None
    if distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed,
        )

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        generator=generator,
        worker_init_fn=_seed_worker,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        worker_init_fn=_seed_worker,
        **loader_kwargs,
    )
    return train_loader, val_loader


def create_dataset_catalog(root: str | Path, output_path: str | Path) -> None:
    dataset = AgricultureImageDataset(root, crop_size=1, augment=False)
    frame = pd.DataFrame(
        {
            "path": [_portable_record_uri(record, root=dataset.root) for record in dataset.records],
            "group": [record.group for record in dataset.records],
        }
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _split_records_by_group(
    records: Sequence[Any],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for record in records:
        group = record.group if isinstance(record, ImageRecord) else record[1]
        grouped.setdefault(group, []).append(record)
    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError("At least two source groups are required for a train/validation split.")

    rng = random.Random(seed)
    rng.shuffle(groups)
    target_samples = max(1, round(len(records) * val_fraction))
    tie_rank = {group: index for index, group in enumerate(groups)}
    first_group = min(
        groups,
        key=lambda group: (abs(len(grouped[group]) - target_samples), tie_rank[group]),
    )
    val_groups = {first_group}
    val_samples = len(grouped[first_group])

    while len(val_groups) < len(groups) - 1:
        candidates = [group for group in groups if group not in val_groups]
        best_group = min(
            candidates,
            key=lambda group: (
                abs(val_samples + len(grouped[group]) - target_samples),
                tie_rank[group],
            ),
        )
        new_distance = abs(val_samples + len(grouped[best_group]) - target_samples)
        current_distance = abs(val_samples - target_samples)
        if new_distance >= current_distance:
            break
        val_groups.add(best_group)
        val_samples += len(grouped[best_group])
    train_records = sorted(
        (record for group, items in grouped.items() if group not in val_groups for record in items),
        key=lambda record: record.uri if isinstance(record, ImageRecord) else record[0],
    )
    val_records = sorted(
        (record for group, items in grouped.items() if group in val_groups for record in items),
        key=lambda record: record.uri if isinstance(record, ImageRecord) else record[0],
    )
    return train_records, val_records


def _derive_group(source_path: Path, archive_chain: Sequence[str], *, root: Path) -> str:
    if archive_chain:
        member_parent = PurePosixPath(archive_chain[-1]).parent
        return member_parent.as_posix() if str(member_parent) != "." else source_path.stem

    if root.is_dir():
        try:
            relative_parent = source_path.relative_to(root).parent
            return relative_parent.as_posix() if str(relative_parent) != "." else source_path.stem
        except ValueError:
            pass
    return source_path.parent.name or source_path.stem


def _compose_uri(source_path: Path, archive_chain: Sequence[str]) -> str:
    uri = str(source_path.resolve())
    if archive_chain:
        uri = "::".join((uri, *archive_chain))
    return uri


def _portable_record_uri(record: ImageRecord, *, root: Path) -> str:
    source_path = record.source_path.resolve()
    if root.is_file() and source_path == root.resolve() and record.archive_chain:
        source_text = ""
    elif root.is_dir():
        try:
            source_text = source_path.relative_to(root.resolve()).as_posix()
        except ValueError:
            source_text = str(source_path)
    else:
        source_text = source_path.name if source_path == root.resolve() else str(source_path)

    if record.archive_chain:
        return "::".join((source_text, *record.archive_chain))
    return source_text


def _inspect_tiff_channels(uri: str, path: Path, payload: bytes | None) -> int:
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError as exc:
        raise RuntimeError("Loading multi-band TIFF imagery requires rasterio.") from exc

    if payload is None:
        with rasterio.open(path) as src:
            return int(src.count)
    with MemoryFile(payload) as memory_file:
        with memory_file.open() as src:
            return int(src.count)


def _load_tiff_array(uri: str, path: Path, payload: bytes | None) -> np.ndarray:
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError as exc:
        raise RuntimeError("Loading multi-band TIFF imagery requires rasterio.") from exc

    if payload is None:
        with rasterio.open(path) as src:
            return src.read()
    with MemoryFile(payload) as memory_file:
        with memory_file.open() as src:
            return src.read()


def _array_to_chw(
    array: np.ndarray,
    *,
    requested_channels: int | None,
    path: str,
) -> np.ndarray:
    if array.ndim == 2:
        if requested_channels not in {None, 1}:
            raise ValueError(f"Expected {requested_channels} channels for '{path}', found 1.")
        return array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D NPY array for '{path}', found shape {array.shape}.")

    if requested_channels is not None:
        first_matches = array.shape[0] == requested_channels
        last_matches = array.shape[-1] == requested_channels
        if first_matches and last_matches:
            raise ValueError(
                f"Ambiguous NPY layout for '{path}' with shape {array.shape}; "
                "both first and last axes match channels."
            )
        if first_matches:
            return array
        if last_matches:
            return np.moveaxis(array, -1, 0)
        raise ValueError(f"Expected {requested_channels} channels for '{path}', found shape {array.shape}.")

    first_plausible = array.shape[0] <= 16 and array.shape[1] > array.shape[0] and array.shape[2] > array.shape[0]
    last_plausible = array.shape[-1] <= 16 and array.shape[0] > array.shape[-1] and array.shape[1] > array.shape[-1]
    if first_plausible == last_plausible:
        raise ValueError(f"Ambiguous NPY layout for '{path}' with shape {array.shape}; pass channels explicitly.")
    return array if first_plausible else np.moveaxis(array, -1, 0)


def _normalize_image_array(array: np.ndarray, *, path: str) -> torch.Tensor:
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Image array must be numeric for '{path}', found {array.dtype}.")
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        # Signed integer types may carry GIS NoData values encoded as the
        # minimum integer (e.g. -2147483647 for INT32).  Clamp them to 0
        # before normalization so the pipeline remains stable.
        if array.min() < 0:
            array = np.clip(array, a_min=0, a_max=None)
        normalized = array.astype(np.float32) / float(info.max)
    else:
        normalized = array.astype(np.float32, copy=False)
        if not np.isfinite(normalized).all():
            raise ValueError(f"Floating image contains NaN or infinity: '{path}'.")
        minimum = float(normalized.min())
        maximum = float(normalized.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"Floating image values must already be normalized to [0, 1] for '{path}'; "
                f"found range [{minimum}, {maximum}]."
            )
    return torch.from_numpy(np.ascontiguousarray(normalized))


def _color_jitter(image: torch.Tensor) -> torch.Tensor:
    brightness = _uniform(0.8, 1.2)
    contrast = _uniform(0.8, 1.2)
    image = image * brightness
    mean = image.mean()
    image = (image - mean) * contrast + mean

    if image.shape[0] == 3:
        saturation = _uniform(0.8, 1.2)
        grayscale = image[0:1] * 0.2989 + image[1:2] * 0.5870 + image[2:3] * 0.1140
        image = grayscale + saturation * (image - grayscale)
        image = _adjust_rgb_hue(image, _uniform(-0.05, 0.05))
    return image.clamp_(0.0, 1.0)


def _adjust_rgb_hue(image: torch.Tensor, hue_factor: float) -> torch.Tensor:
    if math.isclose(hue_factor, 0.0, abs_tol=1e-8):
        return image
    hsv = _rgb_to_hsv(image.clamp(0.0, 1.0))
    hsv[0] = torch.remainder(hsv[0] + hue_factor, 1.0)
    return _hsv_to_rgb(hsv)


def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    red, green, blue = image.unbind(dim=0)
    maximum, max_indices = image.max(dim=0)
    minimum = image.min(dim=0).values
    delta = maximum - minimum
    saturation = torch.where(maximum > 0, delta / maximum.clamp_min(1e-8), torch.zeros_like(maximum))
    hue = torch.zeros_like(maximum)
    nonzero = delta > 1e-8
    hue = torch.where(nonzero & (max_indices == 0), torch.remainder((green - blue) / delta.clamp_min(1e-8), 6.0), hue)
    hue = torch.where(nonzero & (max_indices == 1), (blue - red) / delta.clamp_min(1e-8) + 2.0, hue)
    hue = torch.where(nonzero & (max_indices == 2), (red - green) / delta.clamp_min(1e-8) + 4.0, hue)
    return torch.stack((hue / 6.0, saturation, maximum))


def _hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    hue, saturation, value = image.unbind(dim=0)
    sector = torch.floor(hue * 6.0).to(torch.int64)
    fraction = hue * 6.0 - sector.to(hue.dtype)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector = torch.remainder(sector, 6)

    red = torch.where(
        sector == 0,
        value,
        torch.where(
            sector == 1,
            q,
            torch.where(
                sector == 2,
                p,
                torch.where(sector == 3, p, torch.where(sector == 4, t, value)),
            ),
        ),
    )
    green = torch.where(
        sector == 0,
        t,
        torch.where(
            sector == 1,
            value,
            torch.where(
                sector == 2,
                value,
                torch.where(sector == 3, q, torch.where(sector == 4, p, p)),
            ),
        ),
    )
    blue = torch.where(
        sector == 0,
        p,
        torch.where(
            sector == 1,
            p,
            torch.where(
                sector == 2,
                t,
                torch.where(sector == 3, value, torch.where(sector == 4, value, q)),
            ),
        ),
    )
    return torch.stack((red, green, blue))


def _random_bool() -> bool:
    return bool(torch.rand(()) < 0.5)


def _uniform(low: float, high: float) -> float:
    return float(torch.empty(()).uniform_(low, high).item())


def _should_skip_member_name(member_name: str) -> bool:
    return _should_skip_member_parts(PurePosixPath(member_name).parts)


def _read_tar_member(archive_path: Path, inner_path: str) -> bytes:
    """Extract a single member from a TAR or TAR.GZ archive by path.

    Uses streaming mode (``r:`` / ``r:gz``) which does not require seeking
    and works correctly on files stored sequentially without a central index.
    """
    mode = "r:gz" if archive_path.name.lower().endswith(".tar.gz") else "r:"
    with tarfile.open(archive_path, mode) as tf:
        member = tf.getmember(inner_path)
        fobj = tf.extractfile(member)
        if fobj is None:
            raise OSError(f"Cannot extract '{inner_path}' from '{archive_path}'.")
        return fobj.read()


def _should_skip_member_parts(parts: Sequence[str]) -> bool:
    for part in parts:
        if part in {"__MACOSX", ".DS_Store"}:
            return True
        if part.startswith("._"):
            return True
    return False
