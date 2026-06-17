from __future__ import annotations

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

from ag_foundation.models.dino import RemoteSensingDINOModel, RemoteSensingDINOv3Model
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


def test_dino_v3_gram_anchor_loss_tracks_dense_feature_similarity(fake_timm) -> None:
    fake_timm()
    torch.manual_seed(11)
    model = RemoteSensingDINOv3Model(
        in_channels=4,
        image_size=32,
        model_name="S",
        precision="fp32",
        pretrained_backbone=False,
        dino_out_dim=16,
        dino_hidden_dim=32,
        dino_bottleneck_dim=8,
        head_nlayers=2,
        gram_anchor_weight=0.25,
        gram_anchor_max_tokens=4,
    )

    inputs = torch.randn(2, 4, 32, 32)
    adapted = model.adapt(inputs)
    student_dense = model.student_dense_views([adapted])
    teacher_dense = model.teacher_dense_views([adapted])
    matching_loss = model.gram_anchor_loss(student_dense, teacher_dense)
    shifted_dense = [student_dense[0] + 0.5]
    shifted_loss = model.gram_anchor_loss(shifted_dense, teacher_dense)

    assert matching_loss.ndim == 0
    assert torch.isfinite(matching_loss)
    assert matching_loss <= shifted_loss
    assert model.gram_anchor_weight == pytest.approx(0.25)


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
