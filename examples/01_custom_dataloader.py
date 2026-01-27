"""
Example 0: Using custom dataloaders (MSD format)
Demonstrates how to use custom data loaders with the framework.
This example uses MSDLoader for Medical Segmentation Decathlon datasets.
"""

import sys
import os
from pathlib import Path
import json
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bonehub_segmentation.data_loaders import MSDLoader
from bonehub_segmentation.training import SegmentationTrainer


def train_with_msd_loader():
    """
    Train a custom model using MSD dataset format.
    Demonstrates custom dataloader usage.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 0: Training with Custom MSD DataLoader")
    print("=" * 60 + "\n")

    # ============ Configuration ============
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load training config
    config_path = "configs/training_config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    from bonehub_segmentation.utils import load_config

    config = load_config(config_path)

    train_config = config["training"]
    data_config = config["data"]

    # ============ Initialize MSD DataLoader ============
    print("\nInitializing MSD DataLoader...")
    print("-" * 40)

    # MSD dataset structure:
    # data/MSD/Task01_BrainTumour/
    #   ├── imagesTr/
    #   │   ├── BRATS_001_0000.nii.gz  (T2 modality with _0000 suffix)
    #   │   ├── BRATS_002_0000.nii.gz
    #   │   └── ...
    #   └── labelsTr/
    #       ├── BRATS_001.nii.gz  (label without channel suffix)
    #       ├── BRATS_002.nii.gz
    #       └── ...

    msd_root = "data/MSD"
    task = "Task01_BrainTumour"  # Example task

    if not os.path.exists(os.path.join(msd_root, task)):
        print(f"\nMSD dataset not found at: {os.path.join(msd_root, task)}")
        print("\nTo use this example with MSD data:")
        print(f"1. Download MSD data from: http://medicaldecathlon.com/")
        print(f"2. Place dataset at: {msd_root}/")
        print(f"3. Expected structure: {os.path.join(msd_root, task, 'imagesTr/')}")
        print("\nAlternatively, adapt this example for your custom dataset:")
        print("  - Create custom dataloader inheriting from BaseDataLoader")
        print("  - Place it in: bonehub_segmentation/data_loaders/custom_dataloaders/")
        print("  - Follow MSDLoader as a template")
        return

    # Create train and val dataloaders
    print(f"\nLoading training data from {task}...")
    try:
        train_loader_obj = MSDLoader(
            root_dir=msd_root,
            task=task,
            mode="train",
            val_fraction=0.2,
            seed=42,
            spacing=tuple(data_config.get("spacing", [1.0, 1.0, 1.0])),
            cache_dir=data_config.get("cache_dir"),
            use_persistent_cache=False,
        )

        val_loader_obj = MSDLoader(
            root_dir=msd_root,
            task=task,
            mode="val",
            val_fraction=0.2,
            seed=42,
            spacing=tuple(data_config.get("spacing", [1.0, 1.0, 1.0])),
        )

        print(f"Train samples: {len(train_loader_obj)}")
        print(f"Val samples: {len(val_loader_obj)}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # ============ Create dataloaders ============
    print("\nCreating PyTorch DataLoaders...")
    train_loader = train_loader_obj.get_dataloader(
        batch_size=train_config.get("batch_size", 2),
        shuffle=True,
        num_workers=2,
        config=data_config,
    )

    val_loader = val_loader_obj.get_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=0,
        config=data_config,
    )

    print(f"Train DataLoader batches: {len(train_loader)}")
    print(f"Val DataLoader batches: {len(val_loader)}")

    # ============ Training ============
    print("\n" + "=" * 60)
    print("Training Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Initialize your model")
    print("2. Create SegmentationTrainer instance")
    print("3. Call trainer.train(train_loader, val_loader, num_epochs=100)")
    print("\nExample:")
    print("  from monai.networks.nets import UNet")
    print("  model = UNet(spatial_dims=3, in_channels=1, out_channels=2, ...)")
    print("  trainer = SegmentationTrainer('configs/training_config.yaml')")
    print("  trainer.train(train_loader, val_loader, num_epochs=100)")


if __name__ == "__main__":
    train_with_msd_loader()
