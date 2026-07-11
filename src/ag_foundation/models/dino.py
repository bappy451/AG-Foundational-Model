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
    logits = out.float()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    Q = torch.exp(logits)
    B, K = Q.shape
    
    # make the matrix sums to 1
    sum_Q = torch.sum(Q).clamp_min(1e-12)
    Q.div_(sum_Q)

    for _ in range(iterations):
        # normalize each row: total weight per sample must be 1/B
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True).clamp_min(1e-12)
        Q.div_(sum_of_rows)
        Q.div_(B)

        # normalize each column: total weight per class must be 1/K
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True).clamp_min(1e-12)
        Q.div_(sum_of_cols)
        Q.div_(K)

    Q.mul_(B) # the rows must sum to 1 so that Q is an assignment
    return Q


def koleo_loss(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if features.shape[0] < 2:
        return features.sum() * 0.0
    normalized = F.normalize(features.float(), p=2, dim=-1)
    with torch.no_grad():
        dist = torch.cdist(normalized, normalized)
        dist.fill_diagonal_(float("inf"))
        nearest = dist.argmin(dim=1)
    nearest_features = normalized[nearest]
    nearest_dist = torch.linalg.vector_norm(normalized - nearest_features, dim=-1)
    return -torch.log(nearest_dist.clamp_min(eps)).mean()

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
        if nlayers == 1:
            layers.append(nn.Linear(in_dim, bottleneck_dim))
        else:
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.GELU()))
            for _ in range(nlayers - 2):
                layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.normalize(self.mlp(inputs), p=2, dim=-1)
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
        self.register_buffer("patch_center", torch.zeros(1, 1, dino_out_dim), persistent=True)

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

    def _forward_student_view(
        self,
        rgb_inputs: torch.Tensor,
        mask_ratio: float = 0.3,
        *,
        compute_patch_logits: bool = True,
    ) -> dict[str, Any]:
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
        patch_logits = self.student_ibot_head(patch_tokens_out) if compute_patch_logits else None
        
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
        cls_center = torch.cat([output["cls"].detach().float() for output in teacher_outputs], dim=0).mean(dim=0, keepdim=True)
        patch_center = torch.cat(
            [output["patch"].detach().float().reshape(-1, output["patch"].shape[-1]) for output in teacher_outputs],
            dim=0,
        ).mean(dim=0, keepdim=True).unsqueeze(0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            torch.distributed.all_reduce(cls_center)
            torch.distributed.all_reduce(patch_center)
            cls_center.div_(world_size)
            patch_center.div_(world_size)
        momentum = float(center_momentum)
        self.center.mul_(momentum).add_(cls_center.to(self.center.device), alpha=1.0 - momentum)
        self.patch_center.mul_(momentum).add_(patch_center.to(self.patch_center.device), alpha=1.0 - momentum)
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
        teacher_temperature: float | None = None,
        dino_loss_weight: float = 1.0,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not student_outputs or not teacher_outputs:
            raise ValueError("student_outputs and teacher_outputs cannot be empty.")
        student_temperature = float(student_temperature)
        teacher_temperature = self.teacher_temperature if teacher_temperature is None else float(teacher_temperature)
        if student_temperature <= 0.0 or teacher_temperature <= 0.0:
            raise ValueError("Student and teacher temperatures must be positive.")

        device = student_outputs[0]["cls"].device
        dino_total = torch.zeros((), device=device, dtype=torch.float32)
        ibot_total = torch.zeros((), device=device, dtype=torch.float32)
        dino_terms = 0
        ibot_terms = 0

        for teacher_index, teacher_output in enumerate(teacher_outputs):
            teacher_cls_probs = F.softmax(
                (teacher_output["cls"].float() - self.center) / teacher_temperature,
                dim=-1,
            )
            for student_index, student_output in enumerate(student_outputs):
                if student_index == teacher_index:
                    continue
                student_log_probs = F.log_softmax(
                    student_output["cls"].float() / student_temperature,
                    dim=-1,
                )
                dino_total = dino_total + torch.sum(-teacher_cls_probs * student_log_probs, dim=-1).mean()
                dino_terms += 1

            if teacher_index < len(student_outputs):
                matching_student = student_outputs[teacher_index]
                mask = matching_student["mask"]
                if mask.any():
                    teacher_patch_probs = F.softmax(
                        (teacher_output["patch"].float() - self.patch_center) / teacher_temperature,
                        dim=-1,
                    )
                    student_patch_log_probs = F.log_softmax(
                        matching_student["patch"][mask].float() / student_temperature,
                        dim=-1,
                    )
                    ibot_total = ibot_total + torch.sum(
                        -teacher_patch_probs[mask] * student_patch_log_probs,
                        dim=-1,
                    ).mean()
                    ibot_terms += 1

        if dino_terms == 0:
            raise RuntimeError("DINO loss received no valid teacher/student view pairs.")
        dino_component = dino_total / float(dino_terms)
        ibot_component = ibot_total / float(ibot_terms) if ibot_terms else ibot_total
        global_features = [output["cls_features"] for output in student_outputs[: len(teacher_outputs)]]
        koleo_component = torch.stack([koleo_loss(features) for features in global_features]).mean()
        feature_matrix = torch.cat(global_features, dim=0).float()
        feature_variance = feature_matrix.var(dim=0, unbiased=False).mean()
        teacher_probabilities = torch.cat(
            [F.softmax((output["cls"].float() - self.center) / teacher_temperature, dim=-1) for output in teacher_outputs],
            dim=0,
        )
        mean_probability = teacher_probabilities.mean(dim=0)
        prototype_entropy = -torch.sum(mean_probability * torch.log(mean_probability.clamp_min(1e-12)))
        mask_ratio = torch.stack([output["mask"].float().mean() for output in student_outputs]).mean()

        total = float(dino_loss_weight) * dino_component
        total = total + float(ibot_loss_weight) * ibot_component
        total = total + float(koleo_loss_weight) * koleo_component
        if not return_components:
            return total
        return {
            "loss": total,
            "dino_loss": dino_component,
            "ibot_loss": ibot_component,
            "koleo_loss": koleo_component,
            "feature_variance": feature_variance,
            "prototype_entropy": prototype_entropy,
            "mask_ratio": mask_ratio,
        }
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

        if "patch_center" not in migrated:
            migrated["patch_center"] = self.patch_center.clone()

        return super().load_state_dict(migrated, strict=strict, assign=assign)
