"""Model checkpoint save/load utilities."""
from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    checkpoint_dir: str | Path,
    tag: str = "latest",
) -> Path:
    """Save model + optimizer state to {checkpoint_dir}/{tag}.pt.

    Returns the path written.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / f"{tag}.pt"
    torch.save(
        {
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load checkpoint from path.

    Returns the full checkpoint dict (includes 'step', 'loss').
    Loads weights in-place into model (and optimizer if provided).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
