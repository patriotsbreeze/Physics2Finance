"""
Configuration utilities and experiment reproducibility helpers.
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Any, Dict
from loguru import logger


def set_seed(seed: int = 42):
    """Set random seeds for full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(cfg: Dict[str, Any]) -> torch.device:
    """Resolve device from config, falling back to CPU if CUDA unavailable."""
    requested = cfg.get("hardware", {}).get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def log_model_summary(model: torch.nn.Module, name: str = "Model"):
    trainable = count_parameters(model)
    total = count_all_parameters(model)
    logger.info(
        f"{name}: {trainable:,} trainable / {total:,} total parameters "
        f"({100 * trainable / max(total, 1):.1f}% trainable)"
    )
