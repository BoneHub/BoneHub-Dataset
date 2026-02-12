from pathlib import Path
from dataclasses import dataclass
import os
import json
import nibabel as nib
from typing import Callable

from bonehub_data_schema import DatasetInfo, SubjectInfo, BoneLabelMap


@dataclass
class ExtendedSubjectInfo:
    """Extended SubjectInfo with additional attributes for dataset conversion."""

    img_path: Path = None
    segmentation_path: Path = None
    mesh_path: Path = None
    nurbs_path: Path = None
    subject_info: SubjectInfo = None


class BaseDatasetIO:
    """
    Base class for dataset input/output operations.
    Args:
        dataset_root (Path): The root directory of the dataset to be converted.
        dataset_info (DatasetInfo): Information about the dataset to be included in the metadata.
    """

    class CustomDataHandlers(dict):
        valid_keys = {
            "read_dataset",
            "export_image",
            "export_segmentation",
            "export_mesh",
            "export_nurbs",
        }

        def __setitem__(self, key, value):
            if key not in self.valid_keys:
                raise KeyError(f"Invalid function name: {key}. Valid function names are: {self.valid_keys}.")
            if not callable(value):
                raise ValueError(f"Provided object for '{key}' is not callable")
            return super().__setitem__(key, value)

        def __getitem__(self, key):
            if key not in self:
                raise KeyError(
                    f"Function '{key}' has not been registered. Register it using 'register_data_handler' method before accessing."
                )
            return super().__getitem__(key)

    def __init__(self, dataset_root: Path, dataset_info: DatasetInfo):
        self.dataset_root = dataset_root
        self.dataset_info = dataset_info
        self.custom_data_handlers = self.CustomDataHandlers()

    def register_data_handler(self, func_name: str, func: Callable):
        """
        Register a custom function for dataset operations.

        Args:
            func_name: Name of the function to register. Must be one of:
                - 'read_dataset'
                - 'export_image'
                - 'export_segmentation'
                - 'export_mesh'
                - 'export_nurbs'
            func: The callable function to register. Must match the expected signature:
                - read_dataset: (dataset_root: Path) -> List[ExtendedSubjectInfo]
                - export_image: (input_file_path: Path, output_file_path: Path) -> None
                - export_segmentation: (input_file_path: Path, output_file_path: Path) -> None
                - export_mesh: (input_folder_path: Path, output_folder_path: Path) -> None
                - export_nurbs: (input_folder_path: Path, output_folder_path: Path) -> None

        Raises:
            ValueError: If func_name is invalid or func is not callable.

        Example:
            >>> def my_read_dataset(dataset_root: Path) -> List[ExtendedSubjectInfo]:
            ...     # implementation
            ...     return subjects
            >>> io = BaseDatasetIO(root, info)
            >>> io.register_data_handler('read_dataset', my_read_dataset)
        """

        self.custom_data_handlers[func_name] = func

    def export_to_bonehub_format(self, output_root: Path, output_dataset_id: int, overwrite: bool = False):
        """
        Export the dataset to BoneHub's standard format.
        Args:
            output_root (Path): The root directory where the converted dataset will be saved.
            output_dataset_id (int): The dataset ID to assign to the exported dataset.
            overwrite (bool): Whether to overwrite existing files in the output directory. Default is False.
        """
        self.dataset_info["dataset_id"] = output_dataset_id

        dataset_path = output_root / f"Dataset_{self.dataset_info['dataset_id']:03d}"
        dataset_info_path = dataset_path / f"Dataset_info_{self.dataset_info['dataset_id']:03d}.json"
        subject_info_path = dataset_path / f"Subject_info_{self.dataset_info['dataset_id']:03d}.json"

        if not overwrite:
            if dataset_path.exists():
                raise FileExistsError(
                    f"Dataset directory '{dataset_path}' already exists. Please choose a different output_dataset_id or remove the existing directory."
                )

        print(f"Reading dataset from '{self.dataset_root}'...")
        data = self.custom_data_handlers["read_dataset"](self.dataset_root)
        print(f"Finished reading dataset. Found {len(data)} subjects.")
        print(f"Exporting dataset to '{dataset_path}'...")
        os.makedirs(dataset_path, exist_ok=True)
        with open(dataset_info_path, "w") as f:
            json.dump(self.dataset_info.sorted(), f, indent=4)
        print(f"Dataset info saved to {dataset_info_path}")
        subject_info = []

        for subject_id, subject in enumerate(data, start=1):
            sinfo = subject.subject_info
            sinfo["dataset_id"] = self.dataset_info["dataset_id"]
            sinfo["subject_id"] = subject_id

            if subject.img_path:
                os.makedirs(dataset_path / "Image", exist_ok=True)
                export_file_path = dataset_path / "Image" / f"{subject.subject_info['subject_id']:06d}.nii.gz"
                self.custom_data_handlers["export_image"](subject.img_path, export_file_path)
                print(f"Exported image '{subject.img_path}' to '{export_file_path}'")
            if subject.segmentation_path:
                os.makedirs(dataset_path / "Segmentation", exist_ok=True)
                export_file_path = dataset_path / "Segmentation" / f"{subject.subject_info['subject_id']:06d}.nii.gz"
                self.custom_data_handlers["export_segmentation"](subject.segmentation_path, export_file_path)
                available_labels = sorted(list(set(nib.load(export_file_path).get_fdata().flatten())))
                if available_labels:
                    for label_id in available_labels:
                        if label_id == 0:
                            continue  # skip background label
                        sinfo.set_segmentation_value(BoneLabelMap(label_id), 1)
                print(f"Exported segmentation '{subject.segmentation_path}' to '{export_file_path}'")
            if subject.mesh_path:
                os.makedirs(dataset_path / "Mesh", exist_ok=True)
                export_folder_path = dataset_path / "Mesh" / f"{subject.subject_info['subject_id']:06d}"
                self.custom_data_handlers["export_mesh"](subject.mesh_path, export_folder_path)
                print(f"Exported mesh '{subject.mesh_path}' to '{export_folder_path}'")
            if subject.nurbs_path:
                os.makedirs(dataset_path / "NURBS", exist_ok=True)
                export_folder_path = dataset_path / "NURBS" / f"{subject.subject_info['subject_id']:06d}"
                self.custom_data_handlers["export_nurbs"](subject.nurbs_path, export_folder_path)
                print(f"Exported NURBS '{subject.nurbs_path}' to '{export_folder_path}'")

            subject_info.append(sinfo.sorted())

            with open(subject_info_path, "w") as f:
                json.dump(subject_info, f, indent=4)
            print(f"Updated {subject_info_path.name} for subject {subject_id}")

            if subject_id > 2:
                break

        print(f"Finished exporting dataset to '{dataset_path}'. Total subjects exported: {len(subject_info)}.")
