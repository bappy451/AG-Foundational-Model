from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from ._state_loading import load_compatible_state_dict
from .official_vit import (
    DEFAULT_PRETRAINED_SOURCE,
    BandAdapter,
    RemoteSensingViT,
    _pair,
    _trunc_normal_,
    _validate_precision,
)


class RemoteSensingMIMModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int | tuple[int, int],
        model_name: str = "S",
        precision: str = "fp32",
        pretrained_backbone: bool = True,
        pretrained_source: str = DEFAULT_PRETRAINED_SOURCE,
        pretrained_cfg: str | dict[str, object] | None = None,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        gradient_checkpointing: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.requested_precision = _validate_precision(precision)
        self.image_size = _pair(image_size)
        self.mask_ratio = float(mask_ratio)
        self.norm_pix_loss = bool(norm_pix_loss)
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be between 0 and 1.")

        self.adapter = BandAdapter(in_channels=in_channels, out_channels=3, precision=precision)
        self.backbone = RemoteSensingViT(
            image_size=self.image_size,
            model_name=model_name,
            precision=precision,
            pretrained_backbone=pretrained_backbone,
            pretrained_source=pretrained_source,
            pretrained_cfg=pretrained_cfg,
            gradient_checkpointing=gradient_checkpointing,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.resolved_precision = self.backbone.resolved_precision
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        self.patch_dim = 3 * self.patch_size[0] * self.patch_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        _trunc_normal_(self.mask_token)
        self.reconstruction_head = nn.Linear(self.embed_dim, self.patch_dim)
        nn.init.xavier_uniform_(self.reconstruction_head.weight)
        if self.reconstruction_head.bias is not None:
            nn.init.zeros_(self.reconstruction_head.bias)

    @staticmethod
    def compute_masked_loss(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_patch = (reconstruction.float() - target.float()).pow(2).mean(dim=-1)
        if not mask.any():
            return per_patch.mean()
        return per_patch[mask].mean()

    def forward_with_intermediates(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        adapted = self.adapter(inputs)
        patch_tokens, grid_size = self.backbone.embed_patches(adapted)
        num_patches = patch_tokens.shape[1]
        if num_patches == 0:
            raise ValueError("The current crop size produces no patches for MIM.")

        num_masked = int(round(num_patches * self.mask_ratio))
        if self.mask_ratio > 0.0:
            num_masked = max(1, num_masked)
        num_masked = min(num_patches, num_masked)

        noise = torch.rand(patch_tokens.shape[0], num_patches, device=patch_tokens.device)
        mask = torch.zeros(patch_tokens.shape[0], num_patches, dtype=torch.bool, device=patch_tokens.device)
        mask.scatter_(1, noise.argsort(dim=1)[:, :num_masked], True)

        mask_token = self.mask_token.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_token, patch_tokens)
        positioned = self.backbone.add_position_embeddings(masked_tokens, grid_size, include_cls_token=False)
        encoded = self.backbone.encode_tokens(positioned)
        reconstruction = self.reconstruction_head(encoded).to(dtype=adapted.dtype)
        target = F.unfold(
            adapted.detach(),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2).to(dtype=reconstruction.dtype)
        
        if self.norm_pix_loss:
            target_mean = target.mean(dim=-1, keepdim=True)
            target_var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - target_mean) / (target_var + 1e-6).sqrt()

        loss = self.compute_masked_loss(reconstruction, target, mask)
        return {
            "adapted": adapted,
            "tokens": patch_tokens,
            "mask": mask,
            "reconstructed_patches": reconstruction,
            "target_patches": target,
            "loss": loss,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(inputs)["loss"]

    def initialize_from_state_dict(self, state_dict: Mapping[str, object]) -> None:
        migrated = OrderedDict()
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            migrated._metadata = metadata  # type: ignore[attr-defined]

        for source_prefix, target_prefix in (
            ("adapter.", "adapter."),
            ("backbone.", "backbone."),
            ("student_adapter.", "adapter."),
            ("student_backbone.", "backbone."),
        ):
            for key, value in state_dict.items():
                if key.startswith(source_prefix):
                    migrated[f"{target_prefix}{key.removeprefix(source_prefix)}"] = value

        load_compatible_state_dict(self, migrated, context="MIM")
