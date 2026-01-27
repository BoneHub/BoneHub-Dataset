"""Inference pipeline for segmentation models."""

import os
from pathlib import Path
import numpy as np
import nibabel as nib
from .model_wrappers.base import SegmentationModelWrapper


class SegmentationInference:
    """Simple inference pipeline for segmentation."""

    def __init__(self, model: SegmentationModelWrapper, output_dir: str = "./results"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def segment_image(self, image_path: str) -> dict:
        """Segment a single image."""
        print(f"Processing: {image_path}")
        return self.model.segment(image_path)

    def segment_batch(self, image_paths: list) -> list:
        """Segment multiple images."""
        return [self.segment_image(path) for path in image_paths]

    def save_segmentation(self, result: dict, output_name: str = None):
        """Save segmentation result."""
        if output_name is None:
            output_name = Path(result.get("image_path", "segmentation")).stem

        output_path = self.output_dir / f"{output_name}_segmentation.nii.gz"
        self.model.save_segmentation(result["segmentation"], result["affine"], str(output_path))
        print(f"Saved: {output_path}")

    def process_directory(self, input_dir: str) -> list:
        """Process all images in directory."""
        image_extensions = (".nii.gz", ".nii")
        image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(image_extensions)]

        results = []
        for image_path in image_paths:
            result = self.segment_image(image_path)
            self.save_segmentation(result)
            results.append(result)

        return results
