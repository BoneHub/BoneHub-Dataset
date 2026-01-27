"""Visualization utilities for medical images."""

import numpy as np
import nibabel as nib


def load_nifti(image_path: str) -> np.ndarray:
    """Load NIfTI image."""
    img = nib.load(image_path)
    return img.get_fdata()


def save_nifti(data: np.ndarray, output_path: str, affine: np.ndarray = None):
    """Save image as NIfTI."""
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data.astype(np.int32), affine=affine)
    nib.save(img, output_path)


def print_segmentation_info(image_path: str):
    """Print basic segmentation information."""
    img = nib.load(image_path)
    data = img.get_fdata()
    print(f"Shape: {data.shape}")
    print(f"Unique labels: {np.unique(data)}")
    print(f"Data range: [{data.min()}, {data.max()}]")


def compare_segmentations(seg1_path: str, seg2_path: str):
    """Compare two segmentation masks."""
    seg1 = nib.load(seg1_path).get_fdata()
    seg2 = nib.load(seg2_path).get_fdata()

    intersection = np.sum(seg1 * seg2)
    union = np.sum(np.logical_or(seg1, seg2))
    iou = intersection / union if union > 0 else 0

    print(f"Intersection: {intersection}")
    print(f"Union: {union}")
    print(f"IoU: {iou:.4f}")


if __name__ == "__main__":
    print("Visualization utilities for medical images")
