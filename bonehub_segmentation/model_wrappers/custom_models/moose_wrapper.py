"""MOOSE model wrapper for musculoskeletal segmentation."""

import numpy as np
import torch

from ..base import SegmentationModelWrapper


class MOOSEWrapper(SegmentationModelWrapper):
    """Wrapper for MOOSE model."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(device)
        self.load_model()

    def load_model(self):
        """Load MOOSE model."""
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            self.model = nnUNetPredictor(device=torch.device(self.device))
        except ImportError:
            raise ImportError("Install nnUNetv2: pip install nnunetv2")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image."""
        image = image.astype(np.float32)
        return torch.from_numpy(image)

    def inference(self, image: torch.Tensor) -> torch.Tensor:
        """Run inference."""
        image_np = image.cpu().numpy()
        if image_np.ndim == 3:
            image_np = image_np[np.newaxis, ...]

        with torch.no_grad():
            output = self.model.predict_single_npy_array(image_np)
        return torch.from_numpy(output).to(self.device)

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess output."""
        return output.cpu().numpy()
