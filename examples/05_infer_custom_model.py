"""
Example 4: Inference with custom trained nnUNet model
Demonstrates how to load and use a custom trained model for inference.

Uses CustomNNUNetWrapper to wrap your trained nnUNet model.
For other custom model wrappers, see bonehub_segmentation/model_wrappers/custom_models/
"""

import sys
import os
from pathlib import Path
import json
import torch
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bonehub_segmentation.model_wrappers import CustomNNUNetWrapper, UnifiedSegmentationInterface
from bonehub_segmentation.inference import SegmentationInferencePipeline
from bonehub_segmentation.evaluation import SegmentationEvaluator


def infer_custom_model():
    """
    Use custom trained nnUNet model for inference.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Inference with Custom nnUNet Model")
    print("=" * 60 + "\n")

    # ============ Configuration ============
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load configs
    from bonehub_segmentation.utils import load_config

    model_configs = load_config("configs/model_config.yaml")
    inference_config = load_config("configs/inference_config.yaml")

    custom_config = model_configs["custom_nnunet"]

    # ============ Check for model checkpoint ============
    model_path = custom_config["checkpoint"]

    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at: {model_path}")
        print("\nTo train a custom model:")
        print("1. Run: python examples/01_train_custom_model.py")
        print("2. This will create the checkpoint file")
        print("\nAlternatively, specify your own checkpoint path in:")
        print("  configs/model_config.yaml")
        return

    # ============ Initialize custom model ============
    print(f"\nLoading custom model from: {model_path}")

    try:
        model_wrapper = CustomNNUNetWrapper(
            model_path=model_path,
            num_classes=custom_config["num_classes"],
            class_names={
                0: "background",
                1: "foreground",  # Adjust based on your task
            },
            device=device,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # ============ Create inference pipeline ============
    print("\nSetting up inference pipeline...")

    output_dir = inference_config["inference"]["output_dir"]
    pipeline = SegmentationInferencePipeline(
        model=model_wrapper,
        output_dir=output_dir,
        device=device,
    )

    # ============ Single image inference ============
    print("\nSingle Image Inference:")
    print("-" * 40)

    sample_image = "data/sample_image.nii.gz"

    if not os.path.exists(sample_image):
        print(f"Sample image not found at: {sample_image}")
        print("\nTo test inference:")
        print("1. Place a test image at: data/sample_image.nii.gz")
        print("2. Image should match your model's input requirements")
        print("3. Run the script again")
        return

    # Run inference
    result = pipeline.segment_image(sample_image)

    print(f"Segmentation shape: {result['segmentation'].shape}")
    print(f"Model used: {result['model_name']}")

    # Save result
    pipeline.save_segmentation(
        result,
        output_name="inference_result",
        save_probability=inference_config["inference"]["save_probability"],
    )

    print(f"Results saved to: {output_dir}")

    # ============ Batch inference ============
    print("\n" + "=" * 60)
    print("Batch Inference:")
    print("-" * 40)

    import glob

    image_dir = "data/images"
    image_files = glob.glob(f"{image_dir}/*.nii.gz") + glob.glob(f"{image_dir}/*.nii")

    if len(image_files) > 0:
        print(f"Found {len(image_files)} images")
        print("Processing...")

        results = pipeline.segment_batch(image_files)

        # Save all results
        for i, result in enumerate(results):
            output_name = Path(result["image_path"]).stem
            pipeline.save_segmentation(
                result,
                output_name=output_name,
                save_probability=inference_config["inference"]["save_probability"],
            )

        print(f"Batch processing completed! {len(results)} images processed")

        # Generate report
        pipeline.generate_report(results)

    else:
        print("No images found in data/images/")
        print("Add images for batch processing")

    # ============ Evaluation (if ground truth available) ============
    print("\n" + "=" * 60)
    print("Evaluation (if ground truth available):")
    print("-" * 40)

    label_files = glob.glob(f"{image_dir}/*_label.nii.gz") + glob.glob(f"{image_dir}/*_label.nii")

    if len(label_files) > 0:
        print(f"Found {len(label_files)} ground truth labels")

        evaluator = SegmentationEvaluator(
            num_classes=custom_config["num_classes"],
            include_background=inference_config["evaluation"]["include_background"],
        )

        # Evaluate each prediction
        import nibabel as nib

        for pred_file in image_files[: len(label_files)]:
            # Find corresponding label
            base_name = Path(pred_file).stem
            label_file = None

            for lbl_file in label_files:
                if base_name.replace("_image", "") in lbl_file or base_name in lbl_file:
                    label_file = lbl_file
                    break

            if label_file is None:
                continue

            # Load predictions and ground truth
            pred_nii = nib.load(pred_file)
            label_nii = nib.load(label_file)

            pred_data = np.array(pred_nii.dataobj)
            label_data = np.array(label_nii.dataobj)

            # Evaluate
            metrics = evaluator.evaluate(pred_data, label_data)

            print(f"\nImage: {base_name}")
            print(f"  Dice: {metrics['dice']:.4f}")
            print(f"  IoU: {metrics['iou']:.4f}")
            print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")

        # Print summary
        summary = evaluator.get_summary()
        print("\n" + "-" * 40)
        print("Summary Statistics:")
        for metric_name, stats in summary.items():
            print(f"{metric_name}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    else:
        print("No ground truth labels found")
        print("To evaluate, provide label files with '_label' in the filename")


if __name__ == "__main__":
    infer_custom_model()
