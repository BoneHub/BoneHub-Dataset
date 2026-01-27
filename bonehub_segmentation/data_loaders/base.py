"""Abstract base classes for data loaders."""

from abc import ABC, abstractmethod
from typing import Optional
from monai.data import DataLoader


class BaseDataLoader(ABC):
    """Abstract base class for segmentation data loaders."""

    def __init__(self, mode: str = "train"):
        self.mode = mode
        self.data = []

    @abstractmethod
    def get_transforms(self, mode: str = None):
        """Get transforms for the specified mode."""
        pass

    @abstractmethod
    def get_dataloader(self, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        """Get DataLoader."""
        pass

    def __len__(self) -> int:
        return len(self.data)
