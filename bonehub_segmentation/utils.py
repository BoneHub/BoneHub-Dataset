"""Utility functions for segmentation pipeline."""

from pathlib import Path
import torch
import numpy as np
import yaml


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str):
    """Save configuration to YAML."""
    if config_path.suffix not in [".yaml", ".yml"]:
        raise ValueError("Unsupported config file format. Use .yaml or .yml")
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(cuda: bool = True) -> torch.device:
    """Get device (CUDA or CPU)."""
    if cuda and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clear_gpu_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
