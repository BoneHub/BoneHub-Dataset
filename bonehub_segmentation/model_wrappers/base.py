"""
Abstract base class for all segmentation model wrappers.
Ensures unified interface across different model architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional
from enum import Enum
import numpy as np
import torch
import nibabel as nib
import os


class BoneID(Enum):
    """unified bone labels."""

    SACRUM = 1
    HIP_LEFT = 2
    HIP_RIGHT = 3
    FEMUR_LEFT = 4
    FEMUR_RIGHT = 5
    PATELLA_LEFT = 6
    PATELLA_RIGHT = 7
    TIBIA_LEFT = 8
    TIBIA_RIGHT = 9
    FIBULA_LEFT = 10
    FIBULA_RIGHT = 11
    TALUS_LEFT = 12
    TALUS_RIGHT = 13
    CALCANEUS_LEFT = 14
    CALCANEUS_RIGHT = 15
    NAVICULAR_LEFT = 16
    NAVICULAR_RIGHT = 17
    CUBOID_LEFT = 18
    CUBOID_RIGHT = 19
    LATERAL_CUNEIFORM_LEFT = 20
    LATERAL_CUNEIFORM_RIGHT = 21
    INTERMEDIATE_CUNEIFORM_LEFT = 22
    INTERMEDIATE_CUNEIFORM_RIGHT = 23
    MEDIAL_CUNEIFORM_LEFT = 24
    MEDIAL_CUNEIFORM_RIGHT = 25
    METATARSALS_LEFT = 26
    METATARSALS_RIGHT = 27
    PHALANGES_LEFT = 28
    PHALANGES_RIGHT = 29


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
