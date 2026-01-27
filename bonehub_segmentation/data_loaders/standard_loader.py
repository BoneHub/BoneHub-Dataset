"""Standard medical image segmentation data loader."""

from typing import Dict, List, Optional
import torch
from pathlib import Path

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    ToTensord,
)

from .base import BaseDataLoader


class StandardSegmentationLoader(BaseDataLoader):
    """Standard loader for medical image segmentation datasets."""

    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        mode: str = "train",
    ):
        """Initialize loader.

        Args:
            image_paths: List of image file paths
            label_paths: Optional list of label file paths
            mode: "train" or "val" mode
        """
        super().__init__(mode)

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.data = self._create_data_dicts()

    def _create_data_dicts(self) -> List[Dict]:
        """Create data dictionaries."""
        data_dicts = []
        for i, img_path in enumerate(self.image_paths):
            data_dict = {"image": img_path}
            if self.label_paths is not None and i < len(self.label_paths):
                data_dict["label"] = self.label_paths[i]
            data_dicts.append(data_dict)
        return data_dicts

    def get_transforms(self) -> Compose:
        """Get transforms."""
        common_transforms = [
            LoadImaged(keys=["image", "label"] if self.label_paths else ["image"]),
            EnsureChannelFirstd(keys=["image", "label"] if self.label_paths else ["image"]),
            Orientationd(keys=["image", "label"] if self.label_paths else ["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=200,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]

        if self.mode == "train":
            return Compose(
                [
                    *common_transforms,
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
                    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
                    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            return Compose(
                [
                    *common_transforms,
                    EnsureTyped(keys=["image", "label"] if self.label_paths else ["image"], dtype=torch.float32),
                    ToTensord(keys=["image", "label"] if self.label_paths else ["image"]),
                ]
            )

    def get_dataloader(self, batch_size: int = 4, num_workers: int = 0) -> DataLoader:
        """Create DataLoader."""
        transforms = self.get_transforms()
        dataset = Dataset(data=self.data, transform=transforms)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(self.mode == "train"),
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
