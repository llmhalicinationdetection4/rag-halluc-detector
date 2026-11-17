"""Configuration loading, seeding and device selection."""
from __future__ import annotations

import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """Load the YAML config. Order: explicit path, HALLUDETECT_CONFIG env var, default file."""
    if path is None:
        path = os.environ.get("HALLUDETECT_CONFIG", DEFAULT_CONFIG)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _to_namespace(raw)
    cfg.raw = raw
    return cfg


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preference: str = "auto") -> str:
    """Resolve 'auto' to cuda when available, otherwise cpu."""
    if preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
