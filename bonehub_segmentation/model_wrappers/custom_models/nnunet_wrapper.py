"""Custom nnUNet model wrapper."""

import torch
import numpy as np
from pathlib import Path

from monai.inferers import SlidingWindowInferer

from ..base import SegmentationModelWrapper


class CustomNNUNetWrapper(SegmentationModelWrapper):
    """Wrapper for custom nnUNet models."""

    def __init__(
        self,
        model_path: str,
        num_classes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize wrapper.

        Args:
            model_path: Path to model checkpoint
            num_classes: Number of output classes
            device: Device for inference
        """
        super().__init__(device)
        self.model_path = model_path
        self.num_classes = num_classes
        self.load_model()

    def load_model(self):
        """Load nnUNet model."""
        try:
            from monai.networks.nets import UNet

            checkpoint = torch.load(self.model_path, map_location=self.device)

            self.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.num_classes,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
                num_res_units=2,
            )

            if "model_state" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image."""
        image = torch.from_numpy(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim == 4:
            image = image.unsqueeze(0)
        return image.float().to(self.device)

    def inference(self, image: torch.Tensor) -> torch.Tensor:
        """Run inference with sliding window."""
        inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            overlap=0.5,
        )
        with torch.no_grad():
            logits = inferer(inputs=image, network=self.model)
        return logits

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess output."""
        probs = torch.softmax(output, dim=1)
        segmentation = torch.argmax(probs, dim=1)
        return segmentation.cpu().numpy()[0]
