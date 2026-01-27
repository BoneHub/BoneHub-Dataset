"""
Example 5: Comparing predictions from multiple models
Demonstrates how to compare outputs from different segmentation models.

Shows how to use UnifiedSegmentationInterface with multiple custom model wrappers.
For adding custom models, see bonehub_segmentation/model_wrappers/custom_models/
"""

import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import nibabel as nib
from scipy import ndimage

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bonehub_segmentation.model_wrappers import (
    TotalSegmentatorWrapper,
    MOOSEWrapper,
    CustomNNUNetWrapper,
    UnifiedSegmentationInterface,
)
from bonehub_segmentation.inference import BatchInferencePipeline
from bonehub_segmentation.evaluation import SegmentationEvaluator


def compare_models():
    """
    Compare predictions from multiple segmentation models.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Comparing Multiple Segmentation Models")
    print("=" * 60 + "\n")

    # ============ Configuration ============
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load configs
    from bonehub_segmentation.utils import load_config

    model_configs = load_config("configs/model_config.yaml")

    # ============ Initialize models ============
    print("Initializing models...")
    print("-" * 40)

    interface = UnifiedSegmentationInterface()

    # Try to initialize each model
    models_initialized = []

    # TotalSegmentator
    try:
        ts_wrapper = TotalSegmentatorWrapper(device=device)
        interface.register_model("totalsegmentator", ts_wrapper)
        models_initialized.append("totalsegmentator")
        print("✓ TotalSegmentator initialized")
    except Exception as e:
        print(f"✗ TotalSegmentator failed: {e}")

    # MOOSE
    try:
        moose_wrapper = MOOSEWrapper(device=device)
        interface.register_model("moose", moose_wrapper)
        models_initialized.append("moose")
        print("✓ MOOSE initialized")
    except Exception as e:
        print(f"✗ MOOSE failed: {e}")

    # Custom nnUNet
    model_path = model_configs["custom_nnunet"]["checkpoint"]
    if os.path.exists(model_path):
        try:
            custom_wrapper = CustomNNUNetWrapper(
                model_path=model_path,
                num_classes=model_configs["custom_nnunet"]["num_classes"],
                device=device,
            )
            interface.register_model("custom_nnunet", custom_wrapper)
            models_initialized.append("custom_nnunet")
            print("✓ Custom nnUNet initialized")
        except Exception as e:
            print(f"✗ Custom nnUNet failed: {e}")
    else:
        print(f"✗ Custom nnUNet checkpoint not found: {model_path}")

    if len(models_initialized) == 0:
        print("\nNo models could be initialized!")
        print("Please install required dependencies and download model weights")
        return

    print(f"\nAvailable models: {models_initialized}\n")

    # ============ Test image ============
    sample_image = "data/sample_image.nii.gz"

    if not os.path.exists(sample_image):
        print(f"Sample image not found at: {sample_image}")
        print("\nTo compare models:")
        print("1. Place a test image at: data/sample_image.nii.gz")
        print("2. Run the script again")
        return

    # ============ Run inference with all models ============
    print("Running inference with all models...")
    print("-" * 40)

    batch_pipeline = BatchInferencePipeline(interface)
    comparison_results = batch_pipeline.compare_models(
        image_path=sample_image,
        model_names=models_initialized,
        output_dir="results/model_comparison",
    )

    print("Inference completed!\n")

    # ============ Analyze and compare results ============
    print("=" * 60)
    print("Comparison Analysis")
    print("=" * 60 + "\n")

    # Load original image for reference
    img_nii = nib.load(sample_image)
    img_data = np.array(img_nii.dataobj)

    comparison_metrics = {}

    for model_name, result in comparison_results.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)

        segmentation = result["segmentation"]

        # Basic statistics
        foreground_voxels = np.sum(segmentation > 0)
        total_voxels = segmentation.size
        coverage = (foreground_voxels / total_voxels) * 100

        print(f"Segmentation shape: {segmentation.shape}")
        print(f"Foreground voxels: {foreground_voxels:,}")
        print(f"Coverage: {coverage:.2f}%")

        # Number of connected components
        labeled_array, num_components = ndimage.label(segmentation > 0)
        print(f"Connected components: {num_components}")

        # Unique labels
        unique_labels = np.unique(segmentation)
        unique_labels = unique_labels[unique_labels != 0]
        print(f"Unique structures: {len(unique_labels)}")

        # Store metrics
        comparison_metrics[model_name] = {
            "foreground_voxels": int(foreground_voxels),
            "coverage": float(coverage),
            "num_components": int(num_components),
            "num_structures": len(unique_labels),
        }

    # ============ Compare predictions ============
    print("\n" + "=" * 60)
    print("Pairwise Model Comparison")
    print("=" * 60 + "\n")

    model_list = list(comparison_results.keys())

    if len(model_list) >= 2:
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                model_a = model_list[i]
                model_b = model_list[j]

                seg_a = comparison_results[model_a]["segmentation"] > 0
                seg_b = comparison_results[model_b]["segmentation"] > 0

                # Dice coefficient
                intersection = np.sum(np.logical_and(seg_a, seg_b))
                dice = (2 * intersection) / (np.sum(seg_a) + np.sum(seg_b)) if (np.sum(seg_a) + np.sum(seg_b)) > 0 else 0

                # Jaccard (IoU)
                union = np.sum(np.logical_or(seg_a, seg_b))
                iou = intersection / union if union > 0 else 0

                print(f"{model_a} vs {model_b}:")
                print(f"  Dice coefficient: {dice:.4f}")
                print(f"  Jaccard (IoU): {iou:.4f}")
                print()

    # ============ Generate comparison report ============
    print("=" * 60)
    print("Generating comparison report...")
    print("=" * 60 + "\n")

    report_path = "results/model_comparison/comparison_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Model Comparison Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Test Image: {sample_image}\n")
        f.write(f"Image Shape: {img_data.shape}\n")
        f.write(f"Device: {device}\n\n")

        f.write("Models Compared:\n")
        for model_name in models_initialized:
            f.write(f"  - {model_name}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("Individual Model Metrics:\n")
        f.write("-" * 60 + "\n\n")

        for model_name, metrics in comparison_metrics.items():
            f.write(f"{model_name}:\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value}\n")
            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("Pairwise Similarity:\n")
        f.write("-" * 60 + "\n\n")

        if len(model_list) >= 2:
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    model_a = model_list[i]
                    model_b = model_list[j]

                    seg_a = comparison_results[model_a]["segmentation"] > 0
                    seg_b = comparison_results[model_b]["segmentation"] > 0

                    intersection = np.sum(np.logical_and(seg_a, seg_b))
                    dice = (2 * intersection) / (np.sum(seg_a) + np.sum(seg_b)) if (np.sum(seg_a) + np.sum(seg_b)) > 0 else 0

                    union = np.sum(np.logical_or(seg_a, seg_b))
                    iou = intersection / union if union > 0 else 0

                    f.write(f"{model_a} vs {model_b}:\n")
                    f.write(f"  Dice: {dice:.4f}\n")
                    f.write(f"  IoU: {iou:.4f}\n\n")

    print(f"Report saved to: {report_path}")
    print("\nComparison complete!")


if __name__ == "__main__":
    compare_models()
