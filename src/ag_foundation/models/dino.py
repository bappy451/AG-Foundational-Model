from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from ._state_loading import load_compatible_state_dict
from .official_vit import DEFAULT_PRETRAINED_SOURCE, BandAdapter, RemoteSensingViT, _validate_precision


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def sinkhorn_knopp(out: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    Q = torch.exp(out).float()
    B, K = Q.shape
    
    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q.div_(sum_Q)

    for _ in range(iterations):
        # normalize each row: total weight per sample must be 1/B
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q.div_(sum_of_rows)
        Q.div_(B)

        # normalize each column: total weight per class must be 1/K
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        Q.div_(sum_of_cols)
        Q.div_(K)

    Q.mul_(B) # the rows must sum to 1 so that Q is an assignment
    return Q


def koleo_loss(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    with torch.no_grad():
        dist = torch.cdist(features, features)
        dist.fill_diagonal_(float('inf'))
        min_dist = dist.min(dim=1).values
    return -torch.log(min_dist + eps).mean()


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        norm_last_layer: bool = True,
    ) -> None:
        super().__init__()
        if nlayers < 1:
            raise ValueError("nlayers must be at least 1.")
        if in_dim <= 0 or out_dim <= 0 or hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("All DINO head dimensions must be positive.")

        layers: list[nn.Module] = []
        current_dim = in_dim
        if nlayers == 1:
            layers.append(nn.Linear(current_dim, bottleneck_dim))
            current_dim = bottleneck_dim
        else:
            for layer_index in range(nlayers - 1):
                next_dim = hidden_dim if layer_index < nlayers - 2 else bottleneck_dim
                layers.append(nn.Linear(current_dim, next_dim))
                current_dim = next_dim
                if layer_index < nlayers - 2:
                    layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.last_layer = weight_norm(nn.Linear(current_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.mlp(inputs)
        return self.last_layer(outputs)


class RemoteSensingDINOModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int | tuple[int, int],
        model_name: str,
        precision: str = "fp32",
        pretrained_backbone: bool = True,
        pretrained_source: str = DEFAULT_PRETRAINED_SOURCE,
        pretrained_cfg: str | dict[str, Any] | None = None,
        dino_out_dim: int = 65536,
        dino_hidden_dim: int = 2048,
        dino_bottleneck_dim: int = 256,
        head_nlayers: int = 3,
        gradient_checkpointing: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        teacher_temperature: float = 0.04,
    ) -> None:
        super().__init__()
        self.requested_precision = _validate_precision(precision)
        self.teacher_temperature = float(teacher_temperature)
        if self.teacher_temperature <= 0.0:
            raise ValueError("teacher_temperature must be positive.")
        self.student_adapter = BandAdapter(in_channels=in_channels, out_channels=3, precision=precision)
        self.teacher_adapter = copy.deepcopy(self.student_adapter)
        self.student_backbone = RemoteSensingViT(
            image_size=image_size,
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
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.student_head = DINOHead(
            self.student_backbone.embed_dim,
            dino_out_dim,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
            nlayers=head_nlayers,
        )
        self.teacher_head = DINOHead(
            self.student_backbone.embed_dim,
            dino_out_dim,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
            nlayers=head_nlayers,
        )
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        self.student_ibot_head = DINOHead(
            self.student_backbone.embed_dim,
            dino_out_dim,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
            nlayers=head_nlayers,
        )
        self.teacher_ibot_head = DINOHead(
            self.student_backbone.embed_dim,
            dino_out_dim,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
            nlayers=head_nlayers,
        )
        self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.student_backbone.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        _freeze_module(self.teacher_adapter)
        _freeze_module(self.teacher_backbone)
        _freeze_module(self.teacher_head)
        _freeze_module(self.teacher_ibot_head)
        self.teacher_adapter.eval()
        self.teacher_backbone.eval()
        self.teacher_head.eval()
        self.teacher_ibot_head.eval()
        self.register_buffer("center", torch.zeros(1, dino_out_dim), persistent=True)

    @property
    def adapter(self) -> BandAdapter:
        """Return the trainable adapter for compatibility with shared tooling."""
        return self.student_adapter

    @property
    def feature_dim(self) -> int:
        return int(self.student_backbone.embed_dim)

    def adapt(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.adapt_student(inputs)

    def adapt_student(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.student_adapter(inputs)

    @torch.no_grad()
    def adapt_teacher(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.teacher_adapter(inputs)

    def student_features(self, rgb_inputs: torch.Tensor) -> torch.Tensor:
        return self.student_backbone.forward_cls_token(rgb_inputs)

    def teacher_features(self, rgb_inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher_backbone.forward_cls_token(rgb_inputs)

    def _forward_student_view(self, rgb_inputs: torch.Tensor, mask_ratio: float = 0.3) -> dict[str, Any]:
        patch_tokens, grid_size = self.student_backbone.embed_patches(rgb_inputs)
        B, N, C = patch_tokens.shape
        
        rand = torch.rand(B, N, device=patch_tokens.device)
        mask = rand < mask_ratio
        
        masked_patch_tokens = patch_tokens.clone()
        masked_patch_tokens[mask] = self.mask_token.to(patch_tokens.dtype)
        
        positioned_tokens = self.student_backbone.add_position_embeddings(masked_patch_tokens, grid_size, include_cls_token=True)
        encoded_tokens = self.student_backbone.encode_tokens(positioned_tokens)
        
        cls_tokens = encoded_tokens[:, 0]
        patch_tokens_out = encoded_tokens[:, self.student_backbone.num_prefix_tokens:]
        
        cls_logits = self.student_head(cls_tokens)
        patch_logits = self.student_ibot_head(patch_tokens_out)
        
        return {"cls": cls_logits, "patch": patch_logits, "mask": mask, "cls_features": cls_tokens}

    def _forward_teacher_view(self, rgb_inputs: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            patch_tokens, grid_size = self.teacher_backbone.embed_patches(rgb_inputs)
            positioned_tokens = self.teacher_backbone.add_position_embeddings(patch_tokens, grid_size, include_cls_token=True)
            encoded_tokens = self.teacher_backbone.encode_tokens(positioned_tokens)
            
            cls_tokens = encoded_tokens[:, 0]
            patch_tokens_out = encoded_tokens[:, self.teacher_backbone.num_prefix_tokens:]
            
            cls_logits = self.teacher_head(cls_tokens)
            patch_logits = self.teacher_ibot_head(patch_tokens_out)
            
            return {"cls": cls_logits, "patch": patch_logits}

    def forward_student_views(self, views: Sequence[torch.Tensor]) -> list[dict[str, Any]]:
        return [self._forward_student_view(view) for view in views]

    def forward_teacher_views(self, views: Sequence[torch.Tensor]) -> list[dict[str, Any]]:
        return [self._forward_teacher_view(view) for view in views]

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        momentum = float(momentum)
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Teacher momentum must be between 0 and 1.")
        for student_param, teacher_param in zip(self.student_adapter.parameters(), self.teacher_adapter.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for student_param, teacher_param in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for student_param, teacher_param in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for student_param, teacher_param in zip(self.student_ibot_head.parameters(), self.teacher_ibot_head.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def update_center(self, teacher_outputs: Sequence[torch.Tensor], center_momentum: float) -> None:
        if not teacher_outputs:
            return
        batch_center = torch.cat(
            [output["cls"].detach().float() if isinstance(output, dict) else output.detach().float() for output in teacher_outputs],
            dim=0,
        ).mean(dim=0, keepdim=True)
        self.center.mul_(float(center_momentum)).add_(
            batch_center.to(device=self.center.device),
            alpha=1.0 - float(center_momentum),
        )

    @torch.no_grad()
    def _sync_teacher_from_student(self) -> None:
        self.teacher_adapter.load_state_dict(self.student_adapter.state_dict())
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())

    def initialize_from_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        migrated = OrderedDict()
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            migrated._metadata = metadata  # type: ignore[attr-defined]

        prefix_map = (
            ("adapter.", "student_adapter."),
            ("backbone.", "student_backbone."),
            ("student_adapter.", "student_adapter."),
            ("student_backbone.", "student_backbone."),
            ("student_head.", "student_head."),
            ("student_ibot_head.", "student_ibot_head."),
        )
        for source_prefix, target_prefix in prefix_map:
            for key, value in state_dict.items():
                if key.startswith(source_prefix):
                    migrated[f"{target_prefix}{key.removeprefix(source_prefix)}"] = value
        if "center" in state_dict:
            migrated["center"] = state_dict["center"]

        load_compatible_state_dict(self, migrated, context="DINO")
        self._sync_teacher_from_student()

    def dino_loss(
        self,
        student_outputs: Sequence[dict[str, Any]],
        teacher_outputs: Sequence[dict[str, Any]],
        *,
        student_temperature: float,
    ) -> torch.Tensor:
        if not student_outputs:
            raise ValueError("student_outputs cannot be empty.")
        if not teacher_outputs:
            raise ValueError("teacher_outputs cannot be empty.")

        student_temperature = float(student_temperature)
        if student_temperature <= 0.0:
            raise ValueError("student_temperature must be positive.")
        
        total_dino_loss = torch.zeros((), device=student_outputs[0]["cls"].device, dtype=torch.float32)
        total_ibot_loss = torch.zeros((), device=student_outputs[0]["cls"].device, dtype=torch.float32)
        total_koleo_loss = torch.zeros((), device=student_outputs[0]["cls"].device, dtype=torch.float32)
        
        num_dino_terms = 0
        num_ibot_terms = 0
        num_koleo_terms = 0

        for teacher_index, teacher_output in enumerate(teacher_outputs):
            # Sinkhorn-Knopp on CLS
            teacher_cls_probs = sinkhorn_knopp(teacher_output["cls"] / self.teacher_temperature)
            
            # Sinkhorn-Knopp on patches
            B, N, D = teacher_output["patch"].shape
            teacher_patch_logits = teacher_output["patch"].view(-1, D)
            teacher_patch_probs = sinkhorn_knopp(teacher_patch_logits / self.teacher_temperature).view(B, N, D)
            
            for student_index, student_output in enumerate(student_outputs):
                if student_index == teacher_index and student_index < len(teacher_outputs):
                    continue
                    
                # 1. DINO Loss
                student_cls_log_probs = F.log_softmax(student_output["cls"].float() / student_temperature, dim=-1)
                total_dino_loss = total_dino_loss + torch.sum(-teacher_cls_probs * student_cls_log_probs, dim=-1).mean()
                num_dino_terms += 1
                
                # 2. iBOT Loss (only on masked patches)
                mask = student_output["mask"] # (B, N)
                if mask.any():
                    student_patch_log_probs = F.log_softmax(student_output["patch"][mask].float() / student_temperature, dim=-1)
                    target_probs = teacher_patch_probs[mask]
                    total_ibot_loss = total_ibot_loss + torch.sum(-target_probs * student_patch_log_probs, dim=-1).mean()
                    num_ibot_terms += 1

                # 3. KoLeo Loss
                cls_features = student_output["cls_features"]
                normalized_features = F.normalize(cls_features.float(), p=2, dim=-1)
                total_koleo_loss = total_koleo_loss + koleo_loss(normalized_features)
                num_koleo_terms += 1

        if num_dino_terms == 0:
            raise RuntimeError("DINO loss received no valid teacher/student view pairs.")
            
        final_loss = total_dino_loss / float(num_dino_terms)
        if num_ibot_terms > 0:
            final_loss = final_loss + total_ibot_loss / float(num_ibot_terms)
        if num_koleo_terms > 0:
            final_loss = final_loss + 0.1 * (total_koleo_loss / float(num_koleo_terms))
            
        return final_loss

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        migrated = OrderedDict(state_dict)
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            migrated._metadata = metadata  # type: ignore[attr-defined]

        legacy_adapter_keys = [key for key in migrated if key.startswith("adapter.")]
        for key in legacy_adapter_keys:
            suffix = key.removeprefix("adapter.")
            value = migrated.pop(key)
            migrated[f"student_adapter.{suffix}"] = value
            migrated[f"teacher_adapter.{suffix}"] = value.clone() if hasattr(value, "clone") else value

        return super().load_state_dict(migrated, strict=strict, assign=assign)
