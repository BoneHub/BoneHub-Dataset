"""
Module for reading datasets in BoneHub data structure format
BoneHub data structure format is as follows:

BoneHub Dataset/
├── Dataset_001/
│   ├── Dataset_info_001.json
│   ├── Subject_info_001.json
│   ├── Image/
│   │   ├── 001_000001.nii.gz
│   │   ├── 001_000002.nii.gz
│   │   └── ...
│   ├── Segmentation/
│   │   ├── 001_000001.nii.gz
│   │   ├── 001_000002.nii.gz
│   │   └── ...
│   ├── Mesh/
│   │   ├── 001_000001/
│   │   │   ├── 001_000001_femur_left.stl
│   │   │   ├── 001_000001_femur_right.stl
│   │   │   └── ...
│   │   ├── 001_000002/
│   │   │   ├── 001_000002_femur_left.stl
│   │   │   ├── 001_000002_femur_right.stl
│   │   │   └── ...
│   │   └── ...
│   ├── NURBS/
│   │   ├── 001_000001/
│   │   │   ├── 001_000001_femur_left.iges
│   │   │   ├── 001_000001_femur_right.iges
│   │   │   └── ...
│   │   ├── 001_000002/
│   │   │   ├── 001_000002_femur_left.iges
│   │   │   ├── 001_000002_femur_right.iges
│   │   │   └── ...
│   │   └── ...
│   └── Landmark/
│       ├── 001_000001.csv
│       ├── 001_000002.csv
│       └── ...
├── Dataset_002/
│   └── ...
└── ...


"""

from pathlib import Path
import json

from . import DatasetInfo, SubjectInfo


class BoneHubDatasetIO:
    """Base class for loading datasets that have been constructed in BoneHub data structure format."""

    def __init__(self, datasets_root: Path, dataset_id: int):
        self.datasets_root = datasets_root
        self.dataset_id = dataset_id
        self.dataset_path = datasets_root / f"Dataset_{str(dataset_id).zfill(3)}"
        self.dataset_info: DatasetInfo = self._load_dataset_info()
        self.subject_info: list[SubjectInfo] = self._load_subject_info()

    def _load_dataset_info(self) -> DatasetInfo:
        dataset_info_path = self.dataset_path / f"Dataset_info_{str(self.dataset_id).zfill(3)}.json"
        with open(dataset_info_path, "r") as f:
            dataset_info_dict = json.load(f)
        return DatasetInfo(**dataset_info_dict)

    def _load_subject_info(self) -> list[SubjectInfo]:
        subject_info_list = []
        subject_info_path = self.dataset_path / f"Subject_info_{str(self.dataset_id).zfill(3)}.json"
        with open(subject_info_path, "r") as f:
            subject_info_dict = json.load(f)
        for subject in subject_info_dict:
            subject_info = SubjectInfo(**subject)
            subject_info_list.append(subject_info)
        return subject_info_list

    def check_dataset_integrity(self) -> bool:
        """Check if all files referenced in the subject_info exist in the dataset directory."""
        for subject in self.subject_info:
            if subject.image:
                image_path = (
                    self.dataset_path
                    / "Image"
                    / f"{str(subject.dataset_id).zfill(3)}_{str(subject.subject_id).zfill(6)}.nii.gz"
                )
                if not image_path.exists():
                    print(f"Image file {image_path} does not exist.")
                    return False
            if subject.segmentation:
                segmentation_path = (
                    self.dataset_path
                    / "Segmentation"
                    / f"{str(subject.dataset_id).zfill(3)}_{str(subject.subject_id).zfill(6)}.nii.gz"
                )
                if not segmentation_path.exists():
                    print(f"Segmentation file {segmentation_path} does not exist.")
                    return False
            if subject.mesh:
                for label in subject.mesh:
                    mesh_path = (
                        self.dataset_path
                        / "Mesh"
                        / f"{str(subject.dataset_id).zfill(3)}_{str(subject.subject_id).zfill(6)}"
                        / f"{str(subject.dataset_id).zfill(3)}_{str(subject.subject_id).zfill(6)}_{label}.stl"
                    )
                    if not mesh_path.exists():
                        print(f"Mesh file {mesh_path} does not exist.")
                        return False
            if subject.nurbs:
                for label in subject.nurbs:
                    nurbs_path = (
                        self.dataset_path
                        / "NURBS"
                        / f"{str(subject.dataset_id).zfill(3)}_{str(subject.subject_id).zfill(6)}"
                        / f"{str(subject.dataset_id).zfill(3)}_{str(subject.subject_id).zfill(6)}_{label}.iges"
                    )
                    if not nurbs_path.exists():
                        print(f"NURBS file {nurbs_path} does not exist.")
                        return False
        return True

    def __len__(self):
        return len(self.subject_info)
