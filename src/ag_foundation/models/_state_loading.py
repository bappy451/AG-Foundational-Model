from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def load_compatible_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    context: str,
) -> None:
    target_state = module.state_dict()
    compatible = OrderedDict()
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        compatible._metadata = metadata  # type: ignore[attr-defined]

    mismatches: list[str] = []
    for key, value in state_dict.items():
        target = target_state.get(key)
        if target is None:
            continue
        if isinstance(value, torch.Tensor) and isinstance(target, torch.Tensor) and value.shape != target.shape:
            mismatches.append(f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(target.shape)}")
            continue
        compatible[key] = value

    if mismatches:
        shown = "; ".join(mismatches[:8])
        remaining = len(mismatches) - 8
        suffix = "" if remaining <= 0 else f"; and {remaining} more"
        raise ValueError(
            f"Cannot initialize {context} from this checkpoint because compatible keys have different shapes. "
            f"Use the same ViT family, patch source, crop geometry, and input channel count, or start a fresh stage. "
            f"Mismatches: {shown}{suffix}."
        )

    module.load_state_dict(compatible, strict=False)
