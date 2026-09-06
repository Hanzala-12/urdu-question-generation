"""Small shared helpers: seeding, device pick, checkpoint I/O, SP loading."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from src.config import SPM_MODEL, ModelConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sp(model_path: str | Path = SPM_MODEL):
    """Load the trained SentencePiece processor."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_cfg: ModelConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    extra: dict | None = None,
) -> None:
    """Write a checkpoint.  ``best.pt`` omits the optimizer to stay small."""
    payload: dict = {
        "model_state": model.state_dict(),
        "model_cfg": vars(model_cfg),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if extra:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_checkpoint(ckpt: dict):
    """Reconstruct the model architecture recorded in a checkpoint."""
    from src.model import Seq2Seq

    cfg = ModelConfig(**ckpt["model_cfg"])
    model = Seq2Seq(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg
