"""
Example 3: Using MOOSE for musculoskeletal segmentation
Demonstrates how to use MOOSE model for bone and muscle segmentation.

MOOSE is a custom wrapper for musculoskeletal-specialized nnUNet model.
For other custom model wrappers, see bonehub_segmentation/model_wrappers/custom_models/
"""

import sys
import os
from pathlib import Path
import json
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bonehub_segmentation.model_wrappers import MOOSEWrapper, UnifiedSegmentationInterface
from bonehub_segmentation.inference import SegmentationInferencePipeline, BatchInferencePipeline


def use_moose():
    """
    Use MOOSE for musculoskeletal segmentation.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Using MOOSE for Musculoskeletal Segmentation")
    print("=" * 60 + "\n")

    # ============ Configuration ============
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
    print(f"Using device: {device}")

    # Load model config
    from bonehub_segmentation.utils import load_config

    model_configs = load_config("configs/model_config.yaml")

    moose_config = model_configs["moose"]

    # ============ Initialize MOOSE ============
    print(f"\nInitializing MOOSE...")
    print(f"Description: {moose_config['description']}")
    print(f"Number of classes: {moose_config['num_classes']}")

    try:
        moose_wrapper = MOOSEWrapper(device=device)
        print("MOOSE loaded successfully!")
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo use MOOSE, install nnUNetv2:")
        print("  pip install nnunetv2")
        print("\nNote: You'll also need to download MOOSE weights")
        return

    # ============ Unified Interface ============
    print("\nSetting up unified segmentation interface...")

    interface = UnifiedSegmentationInterface()
    interface.register_model("moose", moose_wrapper)

    # ============ Segment image ============
    print("\nMOOSE is optimized for musculoskeletal MRI/CT images")
    print("Typical applications:")
    print("  - Femur, Tibia, Fibula segmentation")
    print("  - Muscle segmentation")
    print("  - Bone quality assessment\n")

    sample_image_path = "data/sample_msk_image.nii.gz"

    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at: {sample_image_path}")
        print("\nTo use this example:")
        print("1. Place your MRI/CT image at: data/sample_msk_image.nii.gz")
        print("2. Image should be in NIfTI format (.nii.gz or .nii)")
        print("3. Ensure proper intensity normalization for your modality")
        print("4. Run the script again")
        return

    # ============ Run segmentation ============
    print(f"Segmenting image: {sample_image_path}")

    result = moose_wrapper.segment(sample_image_path)

    print("\nSegmentation completed!")
    print(f"Output shape: {result['segmentation'].shape}")
    print(f"Model: {result['model_name']}")

    # ============ Save results ============
    output_dir = "results/moose"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "moose_result.nii.gz")
    moose_wrapper.save_segmentation(
        result["segmentation"],
        result["affine"],
        output_path,
    )

    print(f"Results saved to: {output_dir}")

    # ============ Analyze segmentation ============
    print("\nSegmentation Analysis:")
    print("-" * 40)

    segmentation = result["segmentation"]
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    print(f"Number of structures: {len(unique_labels)}")
    print(f"Structure labels: {unique_labels}")

    label_names = result["original_labels"]
    print("\nIdentified structures:")
    for label_id in unique_labels:
        if int(label_id) in label_names:
            name = label_names[int(label_id)]
            voxel_count = (segmentation == label_id).sum()
            print(f"  - {name}: {voxel_count} voxels")

    # ============ Batch processing ============
    print("\n" + "=" * 60)
    print("Batch Processing Example")
    print("=" * 60)

    # Create batch pipeline
    batch_pipeline = BatchInferencePipeline(interface)

    # Process multiple images
    image_dir = "data/images"
    import glob

    image_files = glob.glob(f"{image_dir}/*.nii.gz") + glob.glob(f"{image_dir}/*.nii")

    if len(image_files) > 1:
        print(f"\nFound {len(image_files)} images for batch processing")
        print("Processing...")

        results = batch_pipeline.process_with_model(
            image_paths=image_files[:5],  # Process first 5
            model_name="moose",
            output_dir="results",
        )

        print(f"Batch processing completed! Processed {len(results)} images")
    else:
        print("\nTo perform batch processing:")
        print("1. Place multiple images in: data/images/")
        print("2. They will be processed in parallel batches")


if __name__ == "__main__":
    use_moose()
