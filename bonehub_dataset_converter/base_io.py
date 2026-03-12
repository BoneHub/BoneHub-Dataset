from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import os
import json
import nibabel as nib
from typing import Callable, List
import SimpleITK as sitk

from bonehub_data_schema import DatasetInfo, SubjectInfo, BoneLabelMap


class DataSource(BaseModel):
    """Class to contain information for data conversion for a subject in a dataset."""

    img_path: Path | None = Field(
        None, description="Path to file or folder containing the image data (e.g. NifTI file or DICOM folder)"
    )
    segmentation_path: Path | None = Field(
        None, description="Path to file containing segmentation data (e.g. NIfTI file or DICOM file)"
    )
    mesh_path: List[Path] | None = Field(None, description="a list of file paths pointing to mesh data (e.g. STL or OBJ files)")
    nurbs_path: List[Path] | None = Field(None, description="a list of file paths pointing to NURBS data.")
    subject_info: SubjectInfo | None = Field(None, description="Information about the subject")

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)


class CustomDataHandlers(BaseModel):
    """
    Class to contain custom data handling functions for dataset conversion.
    """

    read_dataset: Callable[[Path], List[DataSource]] = Field(
        None,
        description="Function to read the dataset and return a list of DataSource objects. Signature: (dataset_root: Path) -> List[DataSource]",
    )
    export_image: Callable[[DataSource, Path], None] = Field(
        None,
        description="Function to export image data. Signature: (data: DataSource, output_file_path: Path) -> None",
    )
    export_segmentation: Callable[[DataSource, Path], None] = Field(
        None,
        description="Function to export segmentation data. Signature: (data: DataSource, output_file_path: Path) -> None",
    )
    export_mesh: Callable[[DataSource, Path], None] = Field(
        None,
        description="Function to export mesh data. Signature: (data: DataSource, output_folder_path: Path) -> None",
    )
    export_nurbs: Callable[[DataSource, Path], None] = Field(
        None,
        description="Function to export NURBS data. Signature: (data: DataSource, output_folder_path: Path) -> None",
    )

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


class BaseDatasetIO:
    """
    Base class for dataset input/output operations.
    Args:
        dataset_root (Path): The root directory of the dataset to be converted.
        dataset_info (DatasetInfo): Information about the dataset to be included in the metadata.
    """

    def __init__(self, dataset_root: Path, dataset_info: DatasetInfo):
        self.dataset_root = dataset_root
        self.dataset_info = dataset_info
        self.custom_data_handlers = CustomDataHandlers()

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
        datalist: list[DataSource] = self.custom_data_handlers.read_dataset(self.dataset_root)
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
                self.custom_data_handlers.export_image(data, export_file_path)
                if verbose:
                    print(f"Exported image '{data.img_path}' to '{export_file_path}'")
            if data.segmentation_path:
                os.makedirs(dataset_path / "Segmentation", exist_ok=True)
                export_file_path = (
                    dataset_path
                    / "Segmentation"
                    / f"{self.dataset_info.dataset_id:03d}_{data.subject_info.subject_id:06d}.nii.gz"
                )
                self.custom_data_handlers.export_segmentation(data, export_file_path)
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
                self.custom_data_handlers.export_mesh(data, export_folder_path)
                available_meshes = [mesh_file.stem for mesh_file in export_folder_path.glob("*.stl")]
                for mesh_name in available_meshes:
                    mesh_name = mesh_name.replace(export_folder_path.name + "_", "")
                    sinfo.set_mesh_value(mesh_name, 1)
                if verbose:
                    print(f"Exported mesh '{data.mesh_path}' to '{export_folder_path}'")
            if data.nurbs_path:
                NotImplementedError("NURBS export is not implemented yet.")

            subject_info.append(sinfo.sorted_dict())

            with open(subject_info_path, "w") as f:
                json.dump(subject_info, f, indent=4)
            if verbose:
                print(f"Updated {subject_info_path.name} for subject {subject_id}")

        if verbose:
            print(f"Finished exporting dataset to '{dataset_path}'. Total subjects exported: {len(subject_info)}.")
