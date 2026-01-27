"""Minimal Medical Segmentation Decathlon loader."""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ToTensord

from ..base import BaseDataLoader


class MSDLoader(BaseDataLoader):
	"""Lightweight loader for MSD-style datasets."""

	def __init__(self, image_paths: List[str], label_paths: Optional[List[str]] = None, mode: str = "train"):
		super().__init__(mode)
		self.image_paths = image_paths
		self.label_paths = label_paths
		self.data = self._create_data_dicts()

	def _create_data_dicts(self) -> List[Dict]:
		data_dicts: List[Dict] = []
		for i, img_path in enumerate(self.image_paths):
			entry = {"image": str(Path(img_path))}
			if self.label_paths is not None and i < len(self.label_paths):
				entry["label"] = str(Path(self.label_paths[i]))
			data_dicts.append(entry)
		return data_dicts

	def get_transforms(self, mode: str = None) -> Compose:
		mode = mode or self.mode
		keys = ["image", "label"] if self.label_paths else ["image"]
		# Keep transforms simple; callers can wrap with more augmentation upstream.
		return Compose(
			[
				LoadImaged(keys=keys),
				EnsureChannelFirstd(keys=keys),
				EnsureTyped(keys=keys, dtype=torch.float32),
				ToTensord(keys=keys),
			]
		)

	def get_dataloader(self, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
		transforms = self.get_transforms()
		dataset = Dataset(data=self.data, transform=transforms)
		return DataLoader(
			dataset=dataset,
			batch_size=batch_size,
			num_workers=num_workers,
			shuffle=shuffle or (self.mode == "train"),
		)

	def __len__(self) -> int:
		return len(self.data)
