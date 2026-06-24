from __future__ import annotations

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

from ag_foundation.models.dino import RemoteSensingDINOModel
from ag_foundation.models.mim import RemoteSensingMIMModel
from ag_foundation.models.vit import BandAdapter, RemoteSensingViT


def test_band_adapter_projects_arbitrary_band_count_to_rgb() -> None:
    adapter = BandAdapter(in_channels=5, out_channels=3, precision="fp32")
    inputs = torch.randn(2, 5, 32, 32)

    outputs = adapter(inputs)

    assert outputs.shape == (2, 3, 32, 32)
    assert outputs.dtype == torch.float32


def test_band_adapter_starts_from_available_image_channels() -> None:
    adapter = BandAdapter(in_channels=5, out_channels=3, precision="fp32")
    inputs = torch.randn(2, 5, 32, 32)

    outputs = adapter(inputs)

    assert torch.allclose(outputs, inputs[:, :3], atol=1e-6, rtol=1e-6)


def test_band_adapter_preserves_rgb_at_initialization() -> None:
    adapter = BandAdapter(in_channels=3, out_channels=3, precision="fp32")
    inputs = torch.randn(2, 3, 32, 32)

    outputs = adapter(inputs)

    assert outputs.shape == inputs.shape
    assert torch.allclose(outputs, inputs, atol=1e-6, rtol=1e-6)


def test_vit_forward_features_returns_patch_tokens_and_metadata(fake_timm) -> None:
    capture: dict[str, object] = {}
    fake_timm(capture)

    model = RemoteSensingViT(
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=True,
    )
    inputs = torch.randn(2, 3, 32, 32)

    outputs = model.forward_features(inputs)

    assert outputs.shape == (2, 4, 384)
    assert model.num_patches == 4
    assert model.grid_size == (2, 2)
    assert outputs.dtype == torch.float32
    assert capture["model_name"] == "vit_small_patch16_224"
    assert capture["kwargs"]["pretrained"] is True
    assert capture["kwargs"]["num_classes"] == 0
    assert capture["kwargs"]["global_pool"] == ""
    assert capture["kwargs"]["img_size"] == (32, 32)


def test_vit_can_load_official_dinov2_backbones(fake_timm) -> None:
    capture: dict[str, object] = {}
    fake_timm(capture)

    model = RemoteSensingViT(
        image_size=28,
        model_name="S",
        precision="fp32",
        pretrained_backbone=True,
        pretrained_source="dinov2",
    )
    inputs = torch.randn(2, 3, 28, 28)

    outputs = model.forward_features(inputs)

    assert outputs.shape == (2, 4, 384)
    assert model.patch_size == (14, 14)
    assert capture["model_name"] == "vit_small_patch14_dinov2.lvd142m"
    assert capture["kwargs"]["pretrained"] is True


def test_mim_model_can_load_official_mae_backbones(fake_timm) -> None:
    capture: dict[str, object] = {}
    fake_timm(capture)

    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="B",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=True,
        pretrained_source="mae",
    )
    outputs = model.forward_with_intermediates(torch.randn(2, 4, 32, 32))

    assert outputs["tokens"].shape == (2, 4, 768)
    assert model.backbone.patch_size == (16, 16)
    assert capture["model_name"] == "vit_base_patch16_224.mae"
    assert capture["kwargs"]["pretrained"] is True


def test_dino_model_can_load_official_dinov3_backbones(fake_timm) -> None:
    capture: dict[str, object] = {}
    fake_timm(capture)

    model = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="L",
        precision="fp32",
        pretrained_backbone=True,
        pretrained_source="dinov3",
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )

    assert model.student_backbone.patch_size == (16, 16)
    assert capture["model_name"] == "vit_large_patch16_dinov3.lvd1689m"
    assert capture["kwargs"]["pretrained"] is True


def test_vit_rejects_unsupported_official_family_combinations() -> None:
    with pytest.raises(ValueError, match="not available for pretrained_source='mae'"):
        RemoteSensingViT(
            image_size=32,
            model_name="S",
            precision="fp32",
            pretrained_backbone=True,
            pretrained_source="mae",
        )


def test_mim_model_returns_expected_shapes_and_mask_ratio(fake_timm) -> None:
    fake_timm()
    torch.manual_seed(7)
    model = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    outputs = model.forward_with_intermediates(torch.randn(2, 4, 32, 32))

    assert outputs["adapted"].shape == (2, 3, 32, 32)
    assert outputs["tokens"].shape == (2, 4, 384)
    assert outputs["mask"].shape == (2, 4)
    assert outputs["mask"].dtype == torch.bool
    assert outputs["reconstructed_patches"].shape == (2, 4, 768)
    assert outputs["target_patches"].shape == (2, 4, 768)
    assert outputs["loss"].ndim == 0
    assert torch.isfinite(outputs["loss"])
    assert abs(float(outputs["mask"].float().mean()) - 0.75) < 0.02


def test_mim_model_can_request_imagenet_pretraining(fake_timm) -> None:
    capture: dict[str, object] = {}
    fake_timm(capture)

    _ = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="B",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=True,
    )

    assert capture["kwargs"]["pretrained"] is True
    assert capture["kwargs"]["img_size"] == (32, 32)


def test_dino_model_returns_logits_and_updates_teacher(fake_timm) -> None:
    fake_timm()
    torch.manual_seed(3)
    model = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )

    inputs = torch.randn(2, 4, 32, 32)
    adapted = model.adapt(inputs)
    assert adapted.shape == (2, 3, 32, 32)

    views = [adapted, adapted, adapted]
    student_outputs = model.forward_student_views(views)
    teacher_outputs = model.forward_teacher_views(views[:2])
    loss = model.dino_loss(student_outputs, teacher_outputs, student_temperature=0.1)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.center.shape == (1, 16)

    student_param = next(model.student_backbone.parameters())
    teacher_param_before = next(model.teacher_backbone.parameters()).detach().clone()
    with torch.no_grad():
        student_param.add_(1.0)
    model.update_teacher(momentum=0.0)
    teacher_param_after = next(model.teacher_backbone.parameters()).detach()

    assert not torch.allclose(teacher_param_before, teacher_param_after)
    assert torch.allclose(teacher_param_after, student_param.detach())

    student_adapter_param = next(model.student_adapter.parameters())
    with torch.no_grad():
        student_adapter_param.add_(2.0)
    model.update_teacher(momentum=0.0)
    teacher_adapter_param = next(model.teacher_adapter.parameters())
    assert torch.allclose(teacher_adapter_param, student_adapter_param)
    assert all(not parameter.requires_grad for parameter in model.teacher_adapter.parameters())


def test_dino_model_migrates_legacy_shared_adapter_checkpoint(fake_timm) -> None:
    fake_timm()
    source = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )
    legacy_state = OrderedDict()
    for key, value in source.state_dict().items():
        if key.startswith("student_adapter."):
            legacy_state[key.replace("student_adapter.", "adapter.", 1)] = value
        elif not key.startswith("teacher_adapter."):
            legacy_state[key] = value

    restored = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )
    restored.load_state_dict(legacy_state)

    for student_parameter, teacher_parameter in zip(
        restored.student_adapter.parameters(),
        restored.teacher_adapter.parameters(),
    ):
        assert torch.equal(student_parameter, teacher_parameter)


def test_dino_model_can_initialize_from_mim_checkpoint(fake_timm) -> None:
    fake_timm()
    source = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    with torch.no_grad():
        source.adapter.proj.weight.fill_(0.25)
        source.backbone.backbone.patch_embed.proj.weight.fill_(0.5)

    restored = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )
    restored.initialize_from_state_dict(source.state_dict())

    assert torch.equal(restored.student_adapter.proj.weight, source.adapter.proj.weight)
    assert torch.equal(
        restored.student_backbone.backbone.patch_embed.proj.weight,
        source.backbone.backbone.patch_embed.proj.weight,
    )
    for student_parameter, teacher_parameter in zip(
        restored.student_adapter.parameters(),
        restored.teacher_adapter.parameters(),
    ):
        assert torch.equal(student_parameter, teacher_parameter)


def test_dino_initialize_from_mim_checkpoint_rejects_incompatible_shapes(fake_timm) -> None:
    fake_timm()
    source = RemoteSensingMIMModel(
        in_channels=5,
        image_size=32,
        model_name="B",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    restored = RemoteSensingDINOModel(
        in_channels=3,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )

    with pytest.raises(ValueError, match="same ViT family.*input channel count"):
        restored.initialize_from_state_dict(source.state_dict())


def test_mim_model_can_initialize_from_dino_checkpoint(fake_timm) -> None:
    fake_timm()
    source = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )
    with torch.no_grad():
        source.student_adapter.proj.weight.fill_(0.75)
        source.student_backbone.backbone.patch_embed.proj.weight.fill_(0.125)

    restored = RemoteSensingMIMModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )
    restored.initialize_from_state_dict(source.state_dict())

    assert torch.equal(restored.adapter.proj.weight, source.student_adapter.proj.weight)
    assert torch.equal(
        restored.backbone.backbone.patch_embed.proj.weight,
        source.student_backbone.backbone.patch_embed.proj.weight,
    )


def test_mim_initialize_from_dino_checkpoint_rejects_incompatible_shapes(fake_timm) -> None:
    fake_timm()
    source = RemoteSensingDINOModel(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
    )
    restored = RemoteSensingMIMModel(
        in_channels=5,
        image_size=32,
        model_name="S",
        precision="fp32",
        mask_ratio=0.75,
        pretrained_backbone=False,
    )

    with pytest.raises(ValueError, match="input channel count"):
        restored.initialize_from_state_dict(source.state_dict())


# ─────────────────────────────────────────────────────────────────────
# Tests: RoPE backbone compatibility (4D patch embed + no pos_embed)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_timm_rope(monkeypatch):
    """Fake timm that simulates an EVA02/DINOv3 RoPE backbone.

    Key differences from the standard fake:
    - ``patch_embed`` returns a 4D tensor (B, C, H, W) instead of (B, N, C).
    - ``pos_embed`` is None (RoPE models have no absolute pos embedding).
    """
    import types

    class _RopePatchEmbed(torch.nn.Module):
        def __init__(self, embed_dim: int, patch_size: int = 16) -> None:
            super().__init__()
            self.patch_size = (patch_size, patch_size)
            self.proj = torch.nn.Conv2d(
                3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
            )
            torch.nn.init.xavier_uniform_(self.proj.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # RoPE backbones return (B, C, H, W) instead of (B, N, C)
            return self.proj(x)

    class _RopeBackbone(torch.nn.Module):
        def __init__(self, embed_dim: int, patch_size: int) -> None:
            super().__init__()
            self.num_features = embed_dim
            self.num_prefix_tokens = 1
            self.patch_embed = _RopePatchEmbed(embed_dim, patch_size=patch_size)
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = None  # <─ critical: no absolute pos embed
            self.pos_drop = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.norm = torch.nn.Identity()
            self.pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}

    def install():
        def create_model(model_name: str, **kwargs):
            img_size = kwargs.get("img_size", (32, 32))
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            embed_dim = 384
            patch_size = 16
            return _RopeBackbone(embed_dim=embed_dim, patch_size=patch_size)

        fake_module = types.SimpleNamespace(create_model=create_model)
        monkeypatch.setattr("ag_foundation.models.official_vit._load_timm", lambda: fake_module)

    return install


def test_vit_handles_rope_4d_patch_embed_output(fake_timm_rope) -> None:
    """RemoteSensingViT must handle 4D (B,C,H,W) patch_embed output from RoPE backbones."""
    fake_timm_rope()
    model = RemoteSensingViT(
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
    )
    inputs = torch.randn(2, 3, 32, 32)

    outputs = model.forward_features(inputs)

    # Should produce (B, num_patches, embed_dim) without crashing
    assert outputs.ndim == 3
    assert outputs.shape[0] == 2
    assert outputs.shape[-1] == 384


def test_vit_handles_missing_pos_embed_rope_backbones(fake_timm_rope) -> None:
    """RemoteSensingViT must forward correctly when backbone.pos_embed is None."""
    fake_timm_rope()
    model = RemoteSensingViT(
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
    )
    inputs = torch.randn(1, 3, 32, 32)

    # Must not raise RuntimeError("does not expose positional embeddings")
    outputs = model.forward_cls_token(inputs)

    assert outputs.ndim == 2
    assert outputs.shape[0] == 1
    assert outputs.shape[-1] == 384
