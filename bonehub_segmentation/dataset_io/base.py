from pathlib import Path
from typing import List

from ..constants import SubjectData


class BaseDatasetReader:
    """Base class for dataset readers."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        if not self.dataset_root.exists():
            raise ValueError(f"{dataset_root} does not exist")
        self.data = self.run_data_reader()

    def run_data_reader(self) -> List[SubjectData]:
        """Method to be implemented by subclasses to read dataset information."""
        raise NotImplementedError(
            "Subclasses must implement this method."
        )  # ONLY subclasses must implement this method not the base class

    def get_label_mapping(self) -> dict:
        """
        Method to get label mapping if available.
        returns a dictionary mapping label names specified in the dataset to standardized labels define in BoneHub.
        """
        return {}

    def __len__(self):
        return len(self.data)
