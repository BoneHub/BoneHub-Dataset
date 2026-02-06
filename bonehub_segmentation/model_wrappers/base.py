"""
Abstract base class for all segmentation model wrappers.
Ensures unified interface across different model architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, Union
import numpy as np
import torch
import nibabel as nib
import os


class SegmentationModelWrapper(ABC):
    """
    Abstract base class for all segmentation model wrappers.
    Ensures unified interface across different model architectures.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load the model and initialize necessary components."""
        pass

    @abstractmethod
    def preprocess(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess image for inference."""
        pass

    @abstractmethod
    def inference(self, image: torch.Tensor) -> torch.Tensor:
        """Run inference."""
        pass

    @abstractmethod
    def postprocess(self, output: torch.Tensor) -> Dict[str, np.ndarray]:
        """Convert model output to segmentation mask."""
        pass

    def segment(self, image_path: str) -> Dict:
        """Complete segmentation pipeline from image file."""
        img_data = nib.load(image_path)
        image = np.array(img_data.dataobj)
        processed = self.preprocess(image)
        output = self.inference(processed)
        result = self.postprocess(output)
        result["affine"] = img_data.affine
        return result

    def save_segmentation(self, segmentation: np.ndarray, affine: np.ndarray, output_path: str):
        """Save segmentation to NIfTI file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(nib.Nifti1Image(segmentation.astype(np.uint8), affine=affine), output_path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, model_name={self.model_name})"
