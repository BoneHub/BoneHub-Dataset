"""Utility functions for segmentation pipeline."""

import json
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


def create_train_val_split(
    image_dir: str,
    label_dir: str,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> tuple:
    """Split dataset into training and validation sets.

    Args:
        image_dir: Path to directory containing images
        label_dir: Path to directory containing labels
        val_split: Fraction of data to use for validation (default: 0.2)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels)
    """
    from sklearn.model_selection import train_test_split

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    # Get list of image files
    image_files = sorted([str(f) for f in image_dir.glob("*") if f.is_file()])

    # Get corresponding label files
    label_files = []
    for img_file in image_files:
        img_name = Path(img_file).stem
        label_file = list(label_dir.glob(f"{img_name}*"))
        if label_file:
            label_files.append(str(label_file[0]))
        else:
            label_files.append(None)

    # Split data
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        image_files,
        label_files,
        test_size=val_split,
        random_state=random_seed,
    )

    return list(train_imgs), list(train_labels), list(val_imgs), list(val_labels)
