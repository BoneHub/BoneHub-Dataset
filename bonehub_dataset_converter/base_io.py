from pathlib import Path
from dataclasses import dataclass
import os
import json
import nibabel as nib
from typing import Callable

from bonehub_data_schema import DatasetInfo, SubjectInfo, BoneLabelMap


@dataclass
class DataSource:
    """Class to contain information for data conversion for a subject in a dataset."""

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
                - read_dataset: (dataset_root: Path) -> List[DataSource]
                - export_image: (data: DataSource, output_file_path: Path) -> None
                - export_segmentation: (data: DataSource, output_file_path: Path) -> None
                - export_mesh: (data: DataSource, output_folder_path: Path) -> None
                - export_nurbs: (data: DataSource, output_folder_path: Path) -> None

        Raises:
            ValueError: If func_name is invalid or func is not callable.

        Example:
            >>> def my_read_dataset(dataset_root: Path) -> List[DataSource]:
            ...     # implementation
            ...     return datalist
            >>> io = BaseDatasetIO(root, info)
            >>> io.register_data_handler('read_dataset', my_read_dataset)
        """

        self.custom_data_handlers[func_name] = func

    def export_to_bonehub_format(
        self, output_root: Path, output_dataset_id: int, overwrite: bool = False, verbose: bool = True
    ):
        """
        Export the dataset to BoneHub's standard format.
        Args:
            output_root (Path): The root directory where the converted dataset will be saved.
            output_dataset_id (int): The dataset ID to assign to the exported dataset.
            overwrite (bool): Whether to overwrite existing files in the output directory. Default is False.
            verbose (bool): Whether to print progress messages. Default is True.
        """
        self.dataset_info.dataset_id = output_dataset_id

        dataset_path = output_root / f"Dataset_{self.dataset_info.dataset_id:03d}"
        dataset_info_path = dataset_path / f"Dataset_info_{self.dataset_info.dataset_id:03d}.json"
        subject_info_path = dataset_path / f"Subject_info_{self.dataset_info.dataset_id:03d}.json"

        if not overwrite:
            if dataset_path.exists():
                raise FileExistsError(
                    f"Dataset directory '{dataset_path}' already exists. Please choose a different output_dataset_id or remove the existing directory."
                )

        if verbose:
            print(f"Reading dataset from '{self.dataset_root}'...")
        datalist: list[DataSource] = self.custom_data_handlers["read_dataset"](self.dataset_root)
        if verbose:
            print(f"Finished reading dataset. Found {len(datalist)} subjects.")
            print(f"Exporting dataset to '{dataset_path}'...")
        os.makedirs(dataset_path, exist_ok=True)
        with open(dataset_info_path, "w") as f:
            json.dump(self.dataset_info.sorted_dict(), f, indent=4)
        if verbose:
            print(f"Dataset info saved to {dataset_info_path}")
        subject_info = []

        for subject_id, data in enumerate(datalist, start=1):
            sinfo = data.subject_info
            sinfo.dataset_id = self.dataset_info.dataset_id
            sinfo.subject_id = subject_id

            if data.img_path:
                os.makedirs(dataset_path / "Image", exist_ok=True)
                export_file_path = (
                    dataset_path / "Image" / f"{self.dataset_info.dataset_id:03d}_{data.subject_info.subject_id:06d}.nii.gz"
                )
                self.custom_data_handlers["export_image"](data, export_file_path)
                if verbose:
                    print(f"Exported image '{data.img_path}' to '{export_file_path}'")
            if data.segmentation_path:
                os.makedirs(dataset_path / "Segmentation", exist_ok=True)
                export_file_path = (
                    dataset_path
                    / "Segmentation"
                    / f"{self.dataset_info.dataset_id:03d}_{data.subject_info.subject_id:06d}.nii.gz"
                )
                self.custom_data_handlers["export_segmentation"](data, export_file_path)
                available_labels = sorted(list(set(nib.load(export_file_path).get_fdata().flatten())))
                if available_labels:
                    for label_id in available_labels:
                        if label_id == 0:
                            continue  # skip background label
                        sinfo.set_segmentation_value(BoneLabelMap(label_id).name, 1)
                if verbose:
                    print(f"Exported segmentation '{data.segmentation_path}' to '{export_file_path}'")
            if data.mesh_path:
                os.makedirs(dataset_path / "Mesh", exist_ok=True)
                export_folder_path = (
                    dataset_path / "Mesh" / f"{self.dataset_info.dataset_id:03d}_{data.subject_info.subject_id:06d}"
                )
                self.custom_data_handlers["export_mesh"](data, export_folder_path)
                if verbose:
                    print(f"Exported mesh '{data.mesh_path}' to '{export_folder_path}'")
            if data.nurbs_path:
                os.makedirs(dataset_path / "NURBS", exist_ok=True)
                export_folder_path = (
                    dataset_path / "NURBS" / f"{self.dataset_info.dataset_id:03d}_{data.subject_info.subject_id:06d}"
                )
                self.custom_data_handlers["export_nurbs"](data, export_folder_path)
                if verbose:
                    print(f"Exported NURBS '{data.nurbs_path}' to '{export_folder_path}'")

            subject_info.append(sinfo.sorted_dict())

            with open(subject_info_path, "w") as f:
                json.dump(subject_info, f, indent=4)
            if verbose:
                print(f"Updated {subject_info_path.name} for subject {subject_id}")

        if verbose:
            print(f"Finished exporting dataset to '{dataset_path}'. Total subjects exported: {len(subject_info)}.")
