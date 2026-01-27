"""TotalSegmentator model wrapper."""

import os
import numpy as np
import torch
import nibabel as nib
import tempfile

from ..base import SegmentationModelWrapper


class TotalSegmentatorWrapper(SegmentationModelWrapper):
    """Wrapper for TotalSegmentator model."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(device)
        self.load_model()

    def load_model(self):
        """Load TotalSegmentator."""
        try:
            from totalsegmentator.python_api import totalsegmentator

            self.model = totalsegmentator
        except ImportError:
            raise ImportError("Install TotalSegmentator: pip install TotalSegmentator")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image."""
        return image

    def inference(self, image: np.ndarray) -> np.ndarray:
        """Run inference."""
        return image

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """Postprocess output."""
        return output

    def segment(self, image_path: str) -> dict:
        """Segment using TotalSegmentator."""
        from totalsegmentator.python_api import totalsegmentator

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "seg.nii.gz")
            totalsegmentator(
                input_path=image_path,
                output_path=output_path,
                device=self.device,
                task="total",
            )
            seg_nii = nib.load(output_path)
            segmentation = np.array(seg_nii.dataobj)

        img_nii = nib.load(image_path)
        return {
            "segmentation": segmentation,
            "affine": img_nii.affine,
        }
