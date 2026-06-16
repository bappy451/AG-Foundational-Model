from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch import nn

SUPPORTED_PRECISIONS = ("fp32", "fp16", "bf16")
PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
VIT_CONFIGS: dict[str, dict[str, Any]] = {
    "S": {
        "model_name": "vit_small_patch16_224",
        "embed_dim": 384,
        "patch_size": 16,
    },
    "B": {
        "model_name": "vit_base_patch16_224",
        "embed_dim": 768,
        "patch_size": 16,
    },
    "L": {
        "model_name": "vit_large_patch16_224",
        "embed_dim": 1024,
        "patch_size": 16,
    },
}


def _validate_precision(precision: str) -> str:
    if precision not in SUPPORTED_PRECISIONS:
        supported = ", ".join(SUPPORTED_PRECISIONS[:-1]) + f", or {SUPPORTED_PRECISIONS[-1]}"
        raise ValueError(f"precision must be {supported}.")
    return precision


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _runtime_compute_dtype(device: torch.device, precision: str) -> torch.dtype:
    if device.type == "cpu" and precision == "fp16":
        return torch.float32
    if device.type == "mps" and precision == "bf16":
        return torch.float32
    return PRECISION_DTYPES[precision]


def _output_dtype(device: torch.device, precision: str) -> torch.dtype:
    return _runtime_compute_dtype(device, precision)


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return tensor.normal_(mean=0.0, std=std)


def _resolve_model_name(model_name: str) -> str:
    if model_name in VIT_CONFIGS:
        return str(VIT_CONFIGS[model_name]["model_name"])
    if model_name in {str(spec["model_name"]) for spec in VIT_CONFIGS.values()}:
        return model_name
    supported = ", ".join(sorted(VIT_CONFIGS))
    raise ValueError(f"Unknown ViT model '{model_name}'. Supported variants: {supported}.")


def _load_timm() -> Any:
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - exercised in environments without optional deps
        raise ImportError(
            "Official ViT backbones require timm. Install the ML extras or `pip install timm`."
        ) from exc
    return timm


@dataclass(frozen=True)
class _ImageGeometry:
    image_size: tuple[int, int]
    patch_size: tuple[int, int]

    @property
    def grid_size(self) -> tuple[int, int]:
        return (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])

    @property
    def num_patches(self) -> int:
        grid = self.grid_size
        return grid[0] * grid[1]


class BandAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        *,
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.requested_precision = _validate_precision(precision)
        self.resolved_precision = self.requested_precision
        self.proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        for output_index in range(self.out_channels):
            input_index = min(output_index, self.in_channels - 1)
            self.proj.weight.data[output_index, input_index, 0, 0] = 1.0
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("BandAdapter expects inputs with shape [B, C, H, W].")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(f"BandAdapter expected {self.in_channels} channels, received {inputs.shape[1]}.")

        output_dtype = _output_dtype(inputs.device, self.resolved_precision)
        compute_dtype = _runtime_compute_dtype(inputs.device, self.resolved_precision)
        outputs = F.conv2d(
            inputs.to(dtype=compute_dtype),
            self.proj.weight.to(device=inputs.device, dtype=compute_dtype),
            None if self.proj.bias is None else self.proj.bias.to(device=inputs.device, dtype=compute_dtype),
        )
        return outputs.to(dtype=output_dtype)


class RemoteSensingViT(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int],
        model_name: str,
        *,
        precision: str = "fp32",
        pretrained_backbone: bool = True,
        pretrained_cfg: str | dict[str, Any] | None = None,
        gradient_checkpointing: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.requested_precision = _validate_precision(precision)
        self.resolved_precision = self.requested_precision
        self.model_name = _resolve_model_name(model_name)
        self.use_gradient_checkpointing = bool(gradient_checkpointing)
        self.image_size = _pair(image_size)
        self.geometry = _ImageGeometry(image_size=self.image_size, patch_size=(16, 16))
        self.grid_size = self.geometry.grid_size
        self.num_patches = self.geometry.num_patches

        timm = _load_timm()
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained_backbone,
            pretrained_cfg=pretrained_cfg,
            img_size=self.image_size,
            num_classes=0,
            global_pool="",
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        if self.use_gradient_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        self.embed_dim = int(self.backbone.num_features)
        self.patch_size = _pair(self.backbone.patch_embed.patch_size)
        pretrained_cfg_dict = (
            getattr(self.backbone, "pretrained_cfg", None)
            or getattr(self.backbone, "default_cfg", {})
            or {}
        )
        mean = torch.tensor(pretrained_cfg_dict.get("mean", (0.5, 0.5, 0.5)), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(pretrained_cfg_dict.get("std", (0.5, 0.5, 0.5)), dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_input_mean", mean, persistent=False)
        self.register_buffer("_input_std", std, persistent=False)
        if self.image_size[0] % self.patch_size[0] != 0 or self.image_size[1] % self.patch_size[1] != 0:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by patch_size {self.patch_size} for {self.model_name}."
            )
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

    def _resize_pos_embed(
        self,
        grid_size: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pos_embed = self.backbone.pos_embed
        if pos_embed is None:
            raise RuntimeError("The selected ViT backbone does not expose positional embeddings.")

        pos_embed = pos_embed.to(device=device, dtype=dtype)
        prefix_tokens = pos_embed[:, : self.num_prefix_tokens]
        patch_tokens = pos_embed[:, self.num_prefix_tokens :]
        if patch_tokens.shape[1] == grid_size[0] * grid_size[1] and self.grid_size == grid_size:
            return pos_embed

        original_grid = int(round(math.sqrt(patch_tokens.shape[1])))
        patch_tokens = patch_tokens.reshape(1, original_grid, original_grid, -1).permute(0, 3, 1, 2)
        patch_tokens = F.interpolate(patch_tokens, size=grid_size, mode="bicubic", align_corners=False)
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)
        return torch.cat((prefix_tokens, patch_tokens), dim=1)

    def embed_patches(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if inputs.ndim != 4:
            raise ValueError("RemoteSensingViT expects inputs with shape [B, 3, H, W].")
        if inputs.shape[1] != 3:
            raise ValueError(f"RemoteSensingViT expects 3 channels, received {inputs.shape[1]}.")
        if inputs.shape[2] % self.patch_size[0] != 0 or inputs.shape[3] % self.patch_size[1] != 0:
            raise ValueError(
                f"Runtime input size {(inputs.shape[2], inputs.shape[3])} "
                f"must be divisible by patch_size {self.patch_size}."
            )

        compute_dtype = _runtime_compute_dtype(inputs.device, self.requested_precision)
        normalized_inputs = inputs.to(dtype=compute_dtype) - self._input_mean.to(
            device=inputs.device,
            dtype=compute_dtype,
        )
        normalized_inputs = normalized_inputs / self._input_std.to(device=inputs.device, dtype=compute_dtype)
        patch_tokens = self.backbone.patch_embed(normalized_inputs)
        grid_size = (inputs.shape[2] // self.patch_size[0], inputs.shape[3] // self.patch_size[1])
        return patch_tokens, grid_size

    def add_position_embeddings(
        self,
        patch_tokens: torch.Tensor,
        grid_size: tuple[int, int],
        *,
        include_cls_token: bool = True,
    ) -> torch.Tensor:
        if include_cls_token:
            cls_token = self.backbone.cls_token
            if cls_token is None:
                raise RuntimeError("The selected ViT backbone does not expose a class token.")
            cls_token = cls_token.expand(patch_tokens.shape[0], -1, -1).to(
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            tokens = torch.cat((cls_token, patch_tokens), dim=1)
        else:
            tokens = patch_tokens

        pos_embed = self._resize_pos_embed(grid_size, device=patch_tokens.device, dtype=patch_tokens.dtype)
        if include_cls_token:
            pos_embed = pos_embed[:, : tokens.shape[1]]
        else:
            pos_embed = pos_embed[:, self.num_prefix_tokens :]
        tokens = tokens + pos_embed
        return self.backbone.pos_drop(tokens)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        outputs = tokens
        for block in self.backbone.blocks:
            if self.training and self.use_gradient_checkpointing:
                outputs = checkpoint_utils.checkpoint(block, outputs)
            else:
                outputs = block(outputs)
        outputs = self.backbone.norm(outputs)
        return outputs.to(dtype=_output_dtype(outputs.device, self.requested_precision))

    def forward_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        patch_tokens, grid_size = self.embed_patches(inputs)
        positioned_tokens = self.add_position_embeddings(patch_tokens, grid_size, include_cls_token=True)
        return self.encode_tokens(positioned_tokens)

    def forward_cls_token(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward_sequence(inputs)
        return outputs[:, 0]

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward_sequence(inputs)
        return outputs[:, self.num_prefix_tokens :]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_features(inputs)
