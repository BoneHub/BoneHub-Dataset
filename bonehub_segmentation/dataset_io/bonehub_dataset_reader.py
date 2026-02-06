"""Base classes for data loaders."""

from typing import Optional
from pathlib import Path
from monai.data import DataLoader, Dataset
from monai.data import DataLoader
from monai.transforms import Compose
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    CropForegroundd,
    SelectItemsd,
)

from .base import BaseDatasetReader


class BoneHubDatasetReader(BaseDatasetReader):
    """Base class for loading datasets that have been constructed in BoneHub data structure format."""

    def __init__(
        self,
        dataset_root: Path,
        mode: str = "train",
        target_spacing: Optional[tuple[float]] = None,
        target_size: Optional[tuple[int]] = None,
        target_orientation: Optional[str] = "PLS",
    ):
        """Initialize BoneHub dataset reader.

        Args:
            dataset_root: Root directory containing case folders in BoneHub format
            mode: "train" or "evaluation" or "inference" mode
            target_spacing: Desired voxel spacing (default: None, use original)
            target_size: Desired output spatial size (default: None, use original)
        """
        super().__init__(dataset_root)
        if mode not in {"train", "evaluation", "inference"}:
            raise ValueError("mode must be 'train', 'evaluation', or 'inference'")
        self.mode = mode
        if target_spacing is not None:
            if len(target_spacing) != 3:
                raise ValueError("target_spacing must be a tuple of 3 floats")
        self.target_spacing = target_spacing
        if target_size is not None:
            if len(target_size) != 3:
                raise ValueError("target_size must be a tuple of 3 ints")
        self.target_size = target_size
        self.target_orientation = target_orientation

    def run_data_reader(self):
        """Run data reader to populate self.data."""
        raise NotImplementedError("Subclasses must implement run_data_reader method.")

    def get_transforms_train(self) -> Compose:
        """Get training transforms with data augmentation.

        Returns:
            Composed transforms for training
        """
        keys = ["image", "label"]

        transforms_list = [
            SelectItemsd(keys=keys),
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes=self.target_orientation, labels=None),
        ]

        if self.target_spacing is not None:
            transforms_list.append(Spacingd(keys=keys, pixdim=self.target_spacing, mode=("bilinear", "nearest")))

        transforms_list.extend(
            [
                ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=keys, source_key="image", margin=10),
            ]
        )

        if self.target_size is not None:
            transforms_list.append(Resized(keys=keys, spatial_size=self.target_size, mode=("trilinear", "nearest")))

        transforms_list.extend(
            [
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
                RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
                RandAffined(
                    keys=keys,
                    prob=0.3,
                    rotate_range=(0.26, 0.26, 0.26),  # ~15 degrees
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                ),
                RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
                RandGaussianSmoothd(keys=["image"], prob=0.15, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            ]
        )

        transforms_list.extend(
            [
                EnsureTyped(keys=keys, dtype=torch.float32),
                ToTensord(keys=keys),
            ]
        )

        return Compose(transforms_list)

    def get_transforms_evaluation(self) -> Compose:
        """Get evaluation transforms without data augmentation.

        Returns:
            Composed transforms for evaluation
        """
        keys = ["image", "label"]

        transforms_list = [
            SelectItemsd(keys=keys),
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes=self.target_orientation, labels=None),
        ]

        if self.target_spacing is not None:
            transforms_list.append(Spacingd(keys=keys, pixdim=self.target_spacing, mode=("bilinear", "nearest")))

        transforms_list.extend(
            [
                ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=keys, source_key="image", margin=10),
            ]
        )

        if self.target_size is not None:
            transforms_list.append(Resized(keys=keys, spatial_size=self.target_size, mode="trilinear"))

        transforms_list.extend(
            [
                EnsureTyped(keys=keys, dtype=torch.float32),
                ToTensord(keys=keys),
            ]
        )

        return Compose(transforms_list)

    def get_transforms_inference(self) -> Compose:
        """Get inference transforms for prediction only.

        Returns:
            Composed transforms for inference
        """
        keys = ["image"]

        transforms_list = [
            SelectItemsd(keys=keys),
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes=self.target_orientation, labels=None),
        ]

        if self.target_spacing is not None:
            transforms_list.append(Spacingd(keys=keys, pixdim=self.target_spacing, mode="bilinear"))

        transforms_list.extend(
            [
                ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=keys, source_key="image", margin=10),
            ]
        )

        if self.target_size is not None:
            transforms_list.append(Resized(keys=keys, spatial_size=self.target_size, mode="trilinear"))

        transforms_list.extend(
            [
                EnsureTyped(keys=keys, dtype=torch.float32),
                ToTensord(keys=keys),
            ]
        )

        return Compose(transforms_list)

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        cache_data: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create DataLoader.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            cache_data: Whether to cache loaded data in memory
            num_workers: Number of workers for data loading

        Returns:
            MONAI DataLoader
        """
        if self.mode == "train":
            transforms = self.get_transforms_train()
        elif self.mode == "evaluation":
            transforms = self.get_transforms_evaluation()
        elif self.mode == "inference":
            transforms = self.get_transforms_inference()
        else:
            raise ValueError("mode must be 'train', 'evaluation', or 'inference'")

        if cache_data:
            from monai.data import CacheDataset

            dataset = CacheDataset(
                data=self.data,
                transform=transforms,
                cache_rate=1.0,
                num_workers=num_workers,
            )
        else:
            dataset = Dataset(data=self.data, transform=transforms)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        """Return number of cases."""
        return len(self.data)
