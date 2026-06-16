from __future__ import annotations

import random
from typing import Any

import numpy as np


def capture_rng_state() -> dict[str, Any]:
    import torch

    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            state["mps"] = torch.mps.get_rng_state()
        except RuntimeError:
            pass
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return

    import torch

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and hasattr(torch, "mps") and hasattr(torch.mps, "set_rng_state"):
        try:
            torch.mps.set_rng_state(state["mps"])
        except RuntimeError:
            pass


def capture_loader_generator_state(loader: Any) -> Any:
    generator = getattr(loader, "generator", None)
    if generator is None or not hasattr(generator, "get_state"):
        return None
    return generator.get_state()


def restore_loader_generator_state(loader: Any, state: Any) -> None:
    if state is None:
        return
    generator = getattr(loader, "generator", None)
    if generator is not None and hasattr(generator, "set_state"):
        generator.set_state(state)
