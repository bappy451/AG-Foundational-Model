from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _FakePatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs).flatten(2).transpose(1, 2)


class _FakeTimmBackbone(nn.Module):
    def __init__(self, embed_dim: int, num_patches: int) -> None:
        super().__init__()
        self.num_features = embed_dim
        self.num_prefix_tokens = 1
        self.patch_embed = _FakePatchEmbed(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList([nn.Identity()])
        self.norm = nn.Identity()
        self.pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        self.grad_checkpointing = bool(enabled)


@pytest.fixture
def fake_timm(monkeypatch):
    def install(capture: dict[str, object] | None = None):
        def create_model(model_name: str, **kwargs):
            if capture is not None:
                capture["model_name"] = model_name
                capture["kwargs"] = dict(kwargs)
            img_size = kwargs.get("img_size", (32, 32))
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            if "large" in model_name:
                embed_dim = 1024
            elif "base" in model_name:
                embed_dim = 768
            else:
                embed_dim = 384
            num_patches = (img_size[0] // 16) * (img_size[1] // 16)
            return _FakeTimmBackbone(embed_dim=embed_dim, num_patches=num_patches)

        fake_module = types.SimpleNamespace(create_model=create_model)
        monkeypatch.setattr("ag_foundation.models.official_vit._load_timm", lambda: fake_module)
        return capture

    return install
