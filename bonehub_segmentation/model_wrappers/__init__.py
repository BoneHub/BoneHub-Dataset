"""
Package initialization for model wrappers.
Provides convenient imports for all available model wrappers.
"""

from .base_wrapper import SegmentationModelWrapper
from .custom_models import (
    CustomNNUNetWrapper,
    MOOSEWrapper,
    TotalSegmentatorWrapper,
)
from .unified_interface import UnifiedSegmentationInterface

__all__ = [
    "SegmentationModelWrapper",
    "TotalSegmentatorWrapper",
    "MOOSEWrapper",
    "UnifiedSegmentationInterface",
    "CustomNNUNetWrapper",
]
