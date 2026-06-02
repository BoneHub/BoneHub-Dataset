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
│   │   │   ├── 001_000001_FEMUR_LEFT.stl
│   │   │   ├── 001_000001_FEMUR_RIGHT.stl
│   │   │   └── ...
│   │   ├── 001_000002/
│   │   │   ├── 001_000002_FEMUR_LEFT.stl
│   │   │   ├── 001_000002_FEMUR_RIGHT.stl
│   │   │   └── ...
│   │   └── ...
│   └── NURBS/
│       ├── 001_000001/
│       │   ├── 001_000001_FEMUR_LEFT.iges
│       │   ├── 001_000001_FEMUR_RIGHT.iges
│       │   └── ...
│       ├── 001_000002/
│       │   ├── 001_000002_FEMUR_LEFT.iges
│       │   ├── 001_000002_FEMUR_RIGHT.iges
│       │   └── ...
│       └── ...
├── Dataset_002/
│   └── ...
└── ...


"""

from pathlib import Path
import json

from . import DatasetInfo, SubjectInfo

DATASET_ZFILL = 3
SUBJECT_ZFILL = 6


class BoneHubDatasetIO:
    """Base class for loading datasets that have been constructed in BoneHub data structure format."""

    def __init__(self, datasets_root: Path, dataset_id: int):
        self.datasets_root = datasets_root
        self.dataset_id = dataset_id
        self.dataset_path = datasets_root / f"Dataset_{str(dataset_id).zfill(DATASET_ZFILL)}"
        self.dataset_info: DatasetInfo = self._load_dataset_info()
        self.subject_info: list[SubjectInfo] = self._load_subject_info()

    def _load_dataset_info(self) -> DatasetInfo:
        dataset_info_path = self.dataset_path / f"Dataset_info_{str(self.dataset_id).zfill(DATASET_ZFILL)}.json"
        with open(dataset_info_path, "r") as f:
            dataset_info_dict = json.load(f)
        return DatasetInfo(**dataset_info_dict)

    def _load_subject_info(self) -> list[SubjectInfo]:
        subject_info_list = []
        subject_info_path = self.dataset_path / f"Subject_info_{str(self.dataset_id).zfill(DATASET_ZFILL)}.json"
        with open(subject_info_path, "r") as f:
            subject_info_dict = json.load(f)
        for subject in subject_info_dict:
            subject_info = SubjectInfo(**subject)
            subject_info_list.append(subject_info)
        return subject_info_list
    
    def save_subject_info(self) -> None:
        """Save the subject info list to the corresponding JSON file in the dataset directory."""
        subject_info_path = self.dataset_path / f"Subject_info_{str(self.dataset_id).zfill(DATASET_ZFILL)}.json"
        with open(subject_info_path, "w") as f:
            json.dump([subject.sorted_dict() for subject in self.subject_info], f, indent=4)

    def check_dataset_integrity(self) -> bool:
        """Check if all files referenced in the subject_info exist in the dataset directory."""
        for subject in self.subject_info:
            if subject.image:
                image_path = (
                    self.dataset_path
                    / "Image"
                    / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}.nii.gz"
                )
                if not image_path.exists():
                    print(f"Image file {image_path} does not exist.")
                    return False
            if subject.segmentation:
                segmentation_path = (
                    self.dataset_path
                    / "Segmentation"
                    / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}.nii.gz"
                )
                if not segmentation_path.exists():
                    print(f"Segmentation file {segmentation_path} does not exist.")
                    return False
            if subject.mesh:
                for label in subject.mesh:
                    mesh_path = (
                        self.dataset_path
                        / "Mesh"
                        / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}"
                        / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}_{label}.stl"
                    )
                    if not mesh_path.exists():
                        print(f"Mesh file {mesh_path} does not exist.")
                        return False
            if subject.nurbs:
                for label in subject.nurbs:
                    nurbs_path = (
                        self.dataset_path
                        / "NURBS"
                        / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}"
                        / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}_{label}.iges"
                    )
                    if not nurbs_path.exists():
                        print(f"NURBS file {nurbs_path} does not exist.")
                        return False
        return True

    def __len__(self):
        return len(self.subject_info)

    def get_image_path(self, subject: SubjectInfo) -> Path | None:
        # TODO: write tests for this function
        if subject.image:
            return (
                self.dataset_path
                / "Image"
                / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}.nii.gz"
            )
        return None

    def get_segmentation_path(self, subject: SubjectInfo) -> Path | None:
        # TODO: write tests for this function
        if subject.segmentation:
            return (
                self.dataset_path
                / "Segmentation"
                / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}.nii.gz"
            )
        return None

    def get_mesh_path(self, subject: SubjectInfo) -> Path | None:
        # TODO: write tests for this function
        if subject.mesh:
            mesh_paths = {}
            for label in subject.mesh:
                mesh_paths[label] = (
                    self.dataset_path
                    / "Mesh"
                    / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}"
                    / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}_{label}.stl"
                )
            return mesh_paths

    def get_nurbs_path(self, subject: SubjectInfo) -> Path | None:
        # TODO: write tests for this function
        if subject.nurbs:
            nurbs_paths = {}
            for label in subject.nurbs:
                nurbs_paths[label] = (
                    self.dataset_path
                    / "NURBS"
                    / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}"
                    / f"{str(subject.dataset_id).zfill(DATASET_ZFILL)}_{str(subject.subject_id).zfill(SUBJECT_ZFILL)}_{label}.iges"
                )
            return nurbs_paths
        return None
