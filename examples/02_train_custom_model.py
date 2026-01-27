"""
Example 1: Training a custom nnUNet model from scratch
Demonstrates the full training pipeline with data loading, augmentation, and evaluation.
Uses StandardSegmentationLoader for generic medical images.

For MSD format datasets, see example 00_custom_dataloader.py
For other custom formats, see bonehub_segmentation/data_loaders/custom_dataloaders/
"""

import sys
import os
from pathlib import Path
import json
import torch
from monai.networks.nets import UNet

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bonehub_segmentation.data_loaders import StandardSegmentationLoader, create_train_val_split
from bonehub_segmentation.training import SegmentationTrainer
from bonehub_segmentation.evaluation import SegmentationEvaluator


def train_custom_model():
    """
    Train a custom nnUNet model for segmentation.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Training Custom nnUNet Model")
    print("=" * 60 + "\n")

    # ============ Configuration ============
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load training config
    from bonehub_segmentation.utils import load_config

    config = load_config("configs/training_config.yaml")

    train_config = config["training"]
    data_config = config["data"]
    model_config = config["model"]

    # ============ Create sample data paths ============
    # In practice, you would have real image and label directories
    # For this example, we'll show the structure
    image_dir = "data/images"
    label_dir = "data/labels"

    print(f"\nExpected data structure:")
    print(f"  Images directory: {image_dir}")
    print(f"  Labels directory: {label_dir}")
    print(f"\nNote: Ensure your images and labels are in .nii.gz or .nii format")

    # Create directories if they don't exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # For demonstration, create dummy data paths
    # In real scenario, you would have actual medical images
    import glob

    image_files = sorted(glob.glob(f"{image_dir}/*.nii.gz")) + sorted(glob.glob(f"{image_dir}/*.nii"))
    label_files = sorted(glob.glob(f"{label_dir}/*.nii.gz")) + sorted(glob.glob(f"{label_dir}/*.nii"))

    if len(image_files) == 0:
        print("\nNo images found in data directory!")
        print("Please add your training images to: data/images/")
        print("And corresponding labels to: data/labels/")
        print("\nExample data structure:")
        print("  data/images/sample_001.nii.gz")
        print("  data/labels/sample_001.nii.gz")
        print("  ...")
        return

    # ============ Create train/val split ============
    print(f"\nFound {len(image_files)} images")
    train_images, train_labels, val_images, val_labels = create_train_val_split(
        image_dir=image_dir,
        label_dir=label_dir,
        val_split=data_config["spacing"],
    )

    # ============ Create datasets and dataloaders ============
    print("\nCreating datasets...")

    train_dataset = StandardSegmentationLoader(
        image_paths=train_images,
        label_paths=train_labels,
        mode="train",
        cache_dir=data_config["cache_dir"],
    )

    val_dataset = StandardSegmentationLoader(
        image_paths=val_images,
        label_paths=val_labels,
        mode="val",
    )

    train_loader = train_dataset.get_dataloader(
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=2,
        config=data_config,
    )

    val_loader = val_dataset.get_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=0,
        config=data_config,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ============ Create model ============
    print("\nCreating model...")

    model = UNet(
        spatial_dims=3,
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        channels=model_config["channels"],
        strides=model_config["strides"],
        num_res_units=model_config["num_res_units"],
        dropout=model_config["dropout"],
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ============ Create trainer ============
    print("\nSetting up trainer...")

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir="checkpoints",
        log_dir="logs",
    )

    trainer.setup_training(
        num_classes=model_config["out_channels"],
        learning_rate=train_config["learning_rate"],
        optimizer_type=train_config["optimizer"],
        loss_type=train_config["loss"],
        weight_decay=train_config["weight_decay"],
    )

    trainer.setup_scheduler(
        scheduler_type=train_config["scheduler"],
        num_epochs=train_config["num_epochs"],
        warmup_epochs=train_config["warmup_epochs"],
    )

    # ============ Train ============
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train(
        num_epochs=train_config["num_epochs"],
        warmup_epochs=train_config["warmup_epochs"],
        early_stopping_patience=train_config["early_stopping_patience"],
    )

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best Dice Score: {trainer.best_val_dice:.4f}")
    print(f"Best model saved to: checkpoints/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    train_custom_model()
