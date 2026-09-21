"""
Experiment Configuration Module
================================
Centralises all training hyperparameters and experiment settings.
Load this config at the start of every training or evaluation run to ensure
reproducibility and consistent experiment tracking.
"""

import os
import json
import random
import platform
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch


# ── Candidate model keys ──────────────────────────────────────────────────────
CANDIDATE_MODELS = ['mobilenet_v2', 'efficientnet_b0', 'resnet18']

# ── Default experiment hyperparameters ───────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    # Dataset
    "dataset_dir": "data/dataset",
    "num_classes": 4,
    "img_size": [224, 224],

    # Split ratios (must sum to 1.0)
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # Reproducibility
    "random_seed": 42,

    # Training
    "batch_size": 32,
    "num_epochs": 25,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "early_stopping_patience": 7,
    "loss_type": "weighted_ce",   # 'weighted_ce' or 'focal'

    # Architecture (set to None to run all candidates sequentially)
    "model_name": None,

    # Hardware
    "device": "auto",             # 'auto', 'cpu', or 'cuda'
    "num_workers": 0,

    # Output directories
    "results_dir": "results",
}


def resolve_device(device_str: str) -> str:
    """Resolve 'auto' to 'cuda' (if available) or 'cpu'."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def set_global_seed(seed: int):
    """Fix random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load experiment configuration.
    If config_path is provided and exists, loads from JSON.
    Otherwise returns the DEFAULT_CONFIG copy.
    """
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        config.update(user_config)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save the experiment configuration to a JSON file."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)


def build_experiment_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct a timestamped experiment metadata dictionary.
    Captured at training start and saved alongside results.
    """
    device = resolve_device(config.get("device", "auto"))
    return {
        "experiment_timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "platform": platform.platform(),
        "config": config,
    }
