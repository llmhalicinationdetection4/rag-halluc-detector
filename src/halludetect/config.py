from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path = "config/default.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    return cfg


def resolve_device(device_setting: str = "auto") -> str:
    if device_setting == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_setting == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"