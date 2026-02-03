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


class BoneHubLabelMap(Enum):
    """unified bone labels."""

    BACKGROUND = 0
    SKULL = 1
    VERTEBRA_C1 = 2
    VERTEBRA_C2 = 3
    VERTEBRA_C3 = 4
    VERTEBRA_C4 = 5
    VERTEBRA_C5 = 6
    VERTEBRA_C6 = 7
    VERTEBRA_C7 = 8
    VERTEBRA_T1 = 9
    VERTEBRA_T2 = 10
    VERTEBRA_T3 = 11
    VERTEBRA_T4 = 12
    VERTEBRA_T5 = 13
    VERTEBRA_T6 = 14
    VERTEBRA_T7 = 15
    VERTEBRA_T8 = 16
    VERTEBRA_T9 = 17
    VERTEBRA_T10 = 18
    VERTEBRA_T11 = 19
    VERTEBRA_T12 = 20
    VERTEBRA_L1 = 21
    VERTEBRA_L2 = 22
    VERTEBRA_L3 = 23
    VERTEBRA_L4 = 24
    VERTEBRA_L5 = 25
    RIB_1_LEFT = 26
    RIB_1_RIGHT = 27
    RIB_2_LEFT = 28
    RIB_2_RIGHT = 29
    RIB_3_LEFT = 30
    RIB_3_RIGHT = 31
    RIB_4_LEFT = 32
    RIB_4_RIGHT = 33
    RIB_5_LEFT = 34
    RIB_5_RIGHT = 35
    RIB_6_LEFT = 36
    RIB_6_RIGHT = 37
    RIB_7_LEFT = 38
    RIB_7_RIGHT = 39
    RIB_8_LEFT = 40
    RIB_8_RIGHT = 41
    RIB_9_LEFT = 42
    RIB_9_RIGHT = 43
    RIB_10_LEFT = 44
    RIB_10_RIGHT = 45
    RIB_11_LEFT = 46
    RIB_11_RIGHT = 47
    RIB_12_LEFT = 48
    RIB_12_RIGHT = 49
    STERNUM = 50
    HUMERUS_LEFT = 51
    HUMERUS_RIGHT = 52
    ULNA_LEFT = 53
    ULNA_RIGHT = 54
    RADIUS_LEFT = 55
    RADIUS_RIGHT = 56
    SCAPHOID_LEFT = 57
    SCAPHOID_RIGHT = 58
    LUNATE_LEFT = 59
    LUNATE_RIGHT = 60
    TRIQUETRUM_LEFT = 61
    TRIQUETRUM_RIGHT = 62
    PISIFORM_LEFT = 63
    PISIFORM_RIGHT = 64
    TRAPEZIUM_LEFT = 65
    TRAPEZIUM_RIGHT = 66
    TRAPEZOID_LEFT = 67
    TRAPEZOID_RIGHT = 68
    CAPITATE_LEFT = 69
    CAPITATE_RIGHT = 70
    HAMATE_LEFT = 71
    HAMATE_RIGHT = 72
    METACARPALS_LEFT = 73
    METACARPALS_RIGHT = 74
    PHALANGES_HAND_LEFT = 75
    PHALANGES_HAND_RIGHT = 76
    SACRUM = 77
    COCCYX = 78
    HIP_LEFT = 79
    HIP_RIGHT = 80
    FEMUR_LEFT = 81
    FEMUR_RIGHT = 82
    PATELLA_LEFT = 83
    PATELLA_RIGHT = 84
    TIBIA_LEFT = 85
    TIBIA_RIGHT = 86
    FIBULA_LEFT = 87
    FIBULA_RIGHT = 88
    TALUS_LEFT = 89
    TALUS_RIGHT = 90
    CALCANEUS_LEFT = 91
    CALCANEUS_RIGHT = 92
    NAVICULAR_LEFT = 93
    NAVICULAR_RIGHT = 94
    CUBOID_LEFT = 95
    CUBOID_RIGHT = 96
    LATERAL_CUNEIFORM_LEFT = 97
    LATERAL_CUNEIFORM_RIGHT = 98
    INTERMEDIATE_CUNEIFORM_LEFT = 99
    INTERMEDIATE_CUNEIFORM_RIGHT = 100
    MEDIAL_CUNEIFORM_LEFT = 101
    MEDIAL_CUNEIFORM_RIGHT = 102
    METATARSALS_LEFT = 103
    METATARSALS_RIGHT = 104
    PHALANGES_FOOT_LEFT = 105
    PHALANGES_FOOT_RIGHT = 106


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
