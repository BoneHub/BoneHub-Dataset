from pathlib import Path
import os
import pydicom
from typing import List

from . import SubjectData


class BaseDatasetReader:
    """Base class for dataset readers."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
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


def get_dicom_subject_metadata(dicom_folder: str) -> dict:
    first_file = next(f for f in os.listdir(dicom_folder) if f.endswith(".dcm") or f.endswith(".dicom"))
    ds = pydicom.dcmread(os.path.join(dicom_folder, first_file), stop_before_pixels=True)
    age = getattr(ds, "PatientAge", None)
    if age:
        age = ''.join(filter(str.isdigit, age))
    
    gender = getattr(ds, "PatientSex", None)
    if gender:
        gender = "male" if gender.lower() == "m" else "female" if gender.lower() == "f" else gender
    
    return {
        "age": age,
        "gender": gender,
    }
