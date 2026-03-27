from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import os
import json
import logging
import nibabel as nib
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from bonehub_data_schema import DatasetInfo, SubjectInfo, BoneLabelMap


class DataSource(BaseModel):
    """Class to contain information for data conversion for a subject in a dataset."""

    img_path: Path | None = Field(
        None, description="Path to file or folder containing the image data (e.g. NifTI file or DICOM folder)"
    )
    segmentation_path: list[Path] | None = Field(
        None, description="a list of file paths to segmentation data (e.g. NIfTI file or DICOM file)"
    )
    mesh_path: list[Path] | None = Field(None, description="a list of file paths pointing to mesh data (e.g. STL or OBJ files)")
    nurbs_path: list[Path] | None = Field(None, description="a list of file paths pointing to NURBS data.")
    subject_info: SubjectInfo | None = Field(None, description="Information about the subject")

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)


class CustomDataHandlers(BaseModel):
    """
    Class to contain custom data handling functions for dataset conversion.
    """

    read_dataset: Callable[[Path], list[DataSource]] = Field(
        None,
        description="Function to read the dataset and return a list of DataSource objects. Signature: (dataset_root: Path) -> list[DataSource]",
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
        self.logger = logging.getLogger(self.__class__.__name__)

    def _process_subject(self, subject_id: int, data: DataSource, dataset_path: Path) -> dict:
        """
        Process a single subject and return its info as a dictionary.
        Args:
            subject_id (int): The subject ID.
            data (DataSource): The subject data.
            dataset_path (Path): The path to the dataset output directory.
        Returns:
            dict: The processed subject info as a dictionary.
        """
        sinfo = data.subject_info
        sinfo.dataset_id = self.dataset_info.dataset_id
        sinfo.subject_id = subject_id

        if data.img_path:
            os.makedirs(dataset_path / "Image", exist_ok=True)
            export_file_path = (
                dataset_path / "Image" / f"{self.dataset_info.dataset_id:03d}_{data.subject_info.subject_id:06d}.nii.gz"
            )
            self.custom_data_handlers.export_image(data, export_file_path)
            self.logger.info(f"Exported image '{data.img_path}' to '{export_file_path}'")
        if data.segmentation_path:
            os.makedirs(dataset_path / "Segmentation", exist_ok=True)
            export_file_path = (
                dataset_path / "Segmentation" / f"{self.dataset_info.dataset_id:03d}_{data.subject_info.subject_id:06d}.nii.gz"
            )
            self.custom_data_handlers.export_segmentation(data, export_file_path)
            available_labels = sorted(list(set(nib.load(export_file_path).get_fdata().flatten())))
            if available_labels:
                for label_id in available_labels:
                    if label_id == 0:
                        continue  # skip background label
                    sinfo.set_segmentation_value(BoneLabelMap(label_id).name, 1)
            self.logger.info(f"Exported {len(available_labels)} segmentation labels to '{export_file_path}'")
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
            self.logger.info(f"Exported {len(available_meshes)} meshes to '{export_folder_path}'")
        if data.nurbs_path:
            NotImplementedError("NURBS export is not implemented yet.")

        return sinfo.sorted_dict()

    def export_to_bonehub_format(
        self,
        output_root: Path,
        output_dataset_id: int,
        overwrite: bool = False,
        num_workers: int = None,
        skip_existing_subjects: bool = False,
    ):
        """
        Export the dataset to BoneHub's standard format.
        Args:
            output_root (Path): The root directory where the converted dataset will be saved.
            output_dataset_id (int): The dataset ID to assign to the exported dataset.
            overwrite (bool): Whether to overwrite existing files in the output directory. Default is False.
            num_workers (int): The number of worker threads to use for parallel processing. Default is None, which uses the number of CPU cores.
            skip_existing_subjects (bool): only process subjects that have not been processed before (based on the existing subject_info JSON file). Default is False.
        """
        self.dataset_info.dataset_id = output_dataset_id

        dataset_path = output_root / f"Dataset_{self.dataset_info.dataset_id:03d}"
        dataset_info_path = dataset_path / f"Dataset_info_{self.dataset_info.dataset_id:03d}.json"
        subject_info_path = dataset_path / f"Subject_info_{self.dataset_info.dataset_id:03d}.json"
        log_file_path = dataset_path / f"Dataset_{output_dataset_id:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Configure logger
        os.makedirs(dataset_path, exist_ok=True)
        log_handler = logging.FileHandler(log_file_path, mode="w")
        log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.INFO)

        if not overwrite:
            if dataset_path.exists():
                raise FileExistsError(
                    f"Dataset directory '{dataset_path}' already exists. Please choose a different output_dataset_id or remove the existing directory."
                )

        self.logger.info(f"Reading dataset from '{self.dataset_root}'...")
        datalist: list[DataSource] = self.custom_data_handlers.read_dataset(self.dataset_root)
        self.logger.info(f"Finished reading dataset. Found {len(datalist)} subjects.")
        self.logger.info(f"Exporting dataset to '{dataset_path}'...")

        subject_info = [None] * len(datalist)
        existing_subject_ids = set()
        if skip_existing_subjects and subject_info_path.exists():
            with open(subject_info_path, "r") as f:
                existing_subject_info = json.load(f)
            existing_subject_info = [
                (subject["subject_id"], SubjectInfo(**subject)) for subject in existing_subject_info if subject is not None
            ]
            existing_subject_info.sort(key=lambda x: x[0])
            for subject_id, info in existing_subject_info:
                subject_info[subject_id - 1] = info.sorted_dict()
                existing_subject_ids.add(subject_id)
            self.logger.info(
                f"Found {len(existing_subject_ids)} existing subjects. Will skip these subjects: {sorted(existing_subject_ids)}"
            )
            del existing_subject_info  # free memory

        os.makedirs(dataset_path, exist_ok=True)
        with open(dataset_info_path, "w") as f:
            json.dump(self.dataset_info.sorted_dict(), f, indent=4)
        self.logger.info(f"Dataset info saved to '{dataset_info_path}'")

        # Process subjects in parallel using ThreadPoolExecutor
        if num_workers is None:
            num_workers = os.cpu_count() or 4
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_subject, subject_id, data, dataset_path): subject_id - 1
                for subject_id, data in enumerate(datalist, start=1)
                if subject_id not in existing_subject_ids
            }
            for future in as_completed(futures):
                index = futures[future]
                subject_info[index] = future.result()
                with open(subject_info_path, "w") as f:
                    json.dump(subject_info, f, indent=4)
                self.logger.info(
                    f"Updated {subject_info_path.name} ({sum(x is not None for x in subject_info)}/{len(subject_info)} subjects)"
                )

        self.logger.info(f"Finished exporting dataset to '{dataset_path}'. Total subjects exported: {len(subject_info)}.")
        self.logger.info(f"Log file saved to {log_file_path}")
