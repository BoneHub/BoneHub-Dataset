"""
Example 2: Using TotalSegmentator for pretrained segmentation
Demonstrates how to use TotalSegmentator for anatomical segmentation.

TotalSegmentator is a custom wrapper for pretrained anatomical segmentation.
For other custom model wrappers, see bonehub_segmentation/model_wrappers/custom_models/
"""

import sys
import os
from pathlib import Path
import json
import nibabel as nib

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bonehub_segmentation.model_wrappers import TotalSegmentatorWrapper, UnifiedSegmentationInterface
from bonehub_segmentation.inference import SegmentationInferencePipeline


def use_totalsegmentator():
    """
    Use TotalSegmentator for anatomical segmentation.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Using TotalSegmentator for Anatomical Segmentation")
    print("=" * 60 + "\n")

    # ============ Configuration ============
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
    print(f"Using device: {device}")

    # Load model config
    from bonehub_segmentation.utils import load_config

    model_configs = load_config("configs/model_config.yaml")

    ts_config = model_configs["totalsegmentator"]

    # ============ Initialize TotalSegmentator ============
    print(f"\nInitializing TotalSegmentator...")
    print(f"Description: {ts_config['description']}")

    try:
        ts_wrapper = TotalSegmentatorWrapper(device=device)
        print("TotalSegmentator loaded successfully!")
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo use TotalSegmentator, install it with:")
        print("  pip install TotalSegmentator")
        print("  pip install nnunetv2")
        return

    # ============ Unified Interface ============
    print("\nSetting up unified segmentation interface...")

    interface = UnifiedSegmentationInterface()
    interface.register_model("totalsegmentator", ts_wrapper)

    # ============ Segment an image ============
    print("\nNote: TotalSegmentator requires CT images in Hounsfield units (HU)")
    print("Typical range: -500 to 2000 HU\n")

    # Example usage with image path
    sample_image_path = "data/sample_ct_image.nii.gz"

    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at: {sample_image_path}")
        print("\nTo use this example:")
        print("1. Place your CT image at: data/sample_ct_image.nii.gz")
        print("2. Image should be in NIfTI format (.nii.gz or .nii)")
        print("3. Run the script again")
        print("\nExample:")
        print("  result = ts_wrapper.segment('data/sample_ct_image.nii.gz')")
        return

    # ============ Run segmentation ============
    print(f"Segmenting image: {sample_image_path}")

    result = ts_wrapper.segment(sample_image_path)

    print("\nSegmentation result keys:")
    print(f"  - segmentation shape: {result['segmentation'].shape}")
    print(f"  - model: {result['model_name']}")
    print(f"  - original labels: {len(result['original_labels'])} anatomical structures")

    # ============ Save results ============
    output_dir = "results/totalsegmentator"
    os.makedirs(output_dir, exist_ok=True)

    # Save segmentation
    output_path = os.path.join(output_dir, "totalsegmentator_result.nii.gz")
    ts_wrapper.save_segmentation(
        result["segmentation"],
        result["affine"],
        output_path,
    )

    print(f"\nResults saved to: {output_dir}")

    # ============ Analyze results ============
    print("\nSegmentation Analysis:")
    print("-" * 40)

    segmentation = result["segmentation"]
    unique_labels = set(segmentation.flatten()) - {0}  # Exclude background

    print(f"Number of structures segmented: {len(unique_labels)}")
    print(f"Segmented label indices: {sorted(list(unique_labels))[:20]}...")  # Show first 20

    # Print information about some identified structures
    print("\nIdentified structures (subset):")
    label_mapping = result["original_labels"]
    for label_id in sorted(list(unique_labels))[:10]:
        if label_id in label_mapping:
            structure_name = label_mapping[label_id]
            voxel_count = (segmentation == label_id).sum()
            print(f"  - Label {label_id}: {structure_name} ({voxel_count} voxels)")

    # ============ Compare with other models (Optional) ============
    print("\n" + "=" * 60)
    print("To compare with other pretrained models, you can add:")
    print("  - MOOSE (for musculoskeletal structures)")
    print("  - Custom trained models")
    print("\nSee example_03_compare_models.py for comparison usage")
    print("=" * 60)


if __name__ == "__main__":
    use_totalsegmentator()
