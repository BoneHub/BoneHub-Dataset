"""
BoneHub Segmentation - MONAI-based medical image segmentation framework
"""

__version__ = "0.1.0"
__author__ = "BoneHub Team"

# Import submodules
from . import data_loaders
from . import model_wrappers
from . import training
from . import evaluation
from . import inference
from . import utils
from . import visualization

# Convenient imports
from .data_loaders import BaseDataLoader, create_train_val_split
from .model_wrappers import (
    SegmentationModelWrapper,
    TotalSegmentatorWrapper,
    MOOSEWrapper,
    CustomNNUNetWrapper,
    UnifiedSegmentationInterface,
)

__all__ = [
    # Data loaders
    "BaseDataLoader",
    "create_train_val_split",
    # Model wrappers
    "SegmentationModelWrapper",
    "TotalSegmentatorWrapper",
    "MOOSEWrapper",
    "CustomNNUNetWrapper",
    "UnifiedSegmentationInterface",
]
