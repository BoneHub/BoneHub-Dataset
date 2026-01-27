"""
Unified segmentation interface.
Allows easy switching between different models with consistent API.
"""

from typing import Dict, List, Union, Optional
import numpy as np

from .base import SegmentationModelWrapper


class UnifiedSegmentationInterface:
    """
    High-level interface that allows switching between different models.
    Ensures consistent output format across all models.
    """

    def __init__(self):
        """Initialize interface."""
        self.models: Dict[str, SegmentationModelWrapper] = {}

    def register_model(self, name: str, model: SegmentationModelWrapper):
        """
        Register a segmentation model.

        Args:
            name: Name identifier for the model
            model: Model wrapper instance
        """
        self.models[name] = model
        print(f"Model '{name}' registered: {model.model_name}")

    def segment(self, image_path: str, model_name: str) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Run segmentation using specified model.

        Args:
            image_path: Path to input image
            model_name: Name of registered model to use

        Returns:
            Unified segmentation output
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not registered. "
                f"Available: {list(self.models.keys())}"
            )

        return self.models[model_name].segment(image_path)

    def list_models(self) -> List[str]:
        """List available models."""
        return list(self.models.keys())

    def get_model(self, model_name: str) -> SegmentationModelWrapper:
        """Get model wrapper by name."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        return self.models[model_name]

    def __repr__(self) -> str:
        return f"UnifiedSegmentationInterface(models={list(self.models.keys())})"
