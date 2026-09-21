from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import os
import json
import shutil
import logging
import logging.handlers
import multiprocessing
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import SimpleITK as sitk
from tqdm import tqdm

from bonehub_data_schema import (
    DatasetInfo,
    SubjectInfo,
    SEGMENTATION_SUFFIX,
    read_segmentation_labels,
    is_compatible_schema_version,
    __version__ as SCHEMA_VERSION,
)

from .utils import FROM_SOURCE

# Each worker process loads MONAI, torch and ITK (~0.4 GB), so keep the default modest.
DEFAULT_MAX_WORKERS = 8


class DatasetConversionError(RuntimeError):
    """Raised by export_to_bonehub_format when subjects failed or the conversion stopped; details are in the log."""

    def __init__(self, message: str, log_file_path: Path, failed_subject_ids: tuple[int, ...] = ()):
        # All fields go to args so the error survives pickling back from a worker process.
        super().__init__(message, log_file_path, failed_subject_ids)
        self.message = message
        self.log_file_path = log_file_path
        self.failed_subject_ids = failed_subject_ids

    def __str__(self) -> str:
        return f"{self.message} See '{self.log_file_path}' for details."


def _format_ids(ids: list[int], limit: int = 20) -> str:
    """List the ids, shortened when there are many; the log always has the full list."""
    if len(ids) <= limit:
        return str(ids)
    return f"{str(ids[:limit])[:-1]}, ... ({len(ids) - limit} more)]"


def _init_subject_worker(log_queue, logger_name: str) -> None:
    """Forward the worker's log records and warnings to the parent and keep ITK/torch single-threaded."""
    for name in (logger_name, "py.warnings"):
        logger = logging.getLogger(name)
        logger.handlers = [logging.handlers.QueueHandler(log_queue)]
        logger.setLevel(logging.INFO)
        logger.propagate = False
    logging.captureWarnings(True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


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
        self._skip_existing_images = False
        self._previous_sources: dict[int, str | None] = {}

    def _can_reuse_image(self, subject_id: int, sinfo: SubjectInfo, image_path: Path) -> bool:
        """True if the existing image was made from the same source subject as this subject id."""
        if not self._skip_existing_images or not image_path.exists():
            return False
        previous = self._previous_sources.get(subject_id)
        current = sinfo.source_subject_path
        if previous is None or current is None:
            self.logger.warning(
                f"Subject {subject_id}: cannot confirm that '{image_path.name}' belongs to this subject; converting again."
            )
            return False
        if previous != current:
            self.logger.warning(
                f"Subject {subject_id}: '{image_path.name}' was made from '{previous}', "
                f"but this subject is '{current}'; converting again."
            )
            return False
        return True

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
            if self._can_reuse_image(subject_id, sinfo, export_file_path):
                self.logger.info(f"Kept existing image '{export_file_path}' (same source subject as before)")
            else:
                self.custom_data_handlers.export_image(data, export_file_path)
                self.logger.info(f"Exported image '{data.img_path}' to '{export_file_path}'")
        if data.segmentation_path:
            os.makedirs(dataset_path / "Segmentation", exist_ok=True)
            export_file_path = (
                dataset_path
                / "Segmentation"
                / f"{self.dataset_info.dataset_id:03d}_{data.subject_info.subject_id:06d}{SEGMENTATION_SUFFIX}"
            )
            self.custom_data_handlers.export_segmentation(data, export_file_path)
            available_labels = read_segmentation_labels(export_file_path)
            for label_name, label_status in available_labels.items():
                sinfo.set_segmentation_value(label_name, label_status)
            self.logger.info(f"Exported {len(available_labels)} segmentation labels to '{export_file_path}'")
        if data.mesh_path:
            os.makedirs(dataset_path / "Mesh", exist_ok=True)
            export_folder_path = (
                dataset_path / "Mesh" / f"{self.dataset_info.dataset_id:03d}_{data.subject_info.subject_id:06d}"
            )
            # Labels are read back from the file names, so no file from an earlier run may remain.
            if export_folder_path.exists():
                shutil.rmtree(export_folder_path)
            self.custom_data_handlers.export_mesh(data, export_folder_path)
            available_meshes = [mesh_file.stem for mesh_file in export_folder_path.glob("*.stl")]
            for mesh_name in available_meshes:
                mesh_name = mesh_name.replace(export_folder_path.name + "_", "")
                sinfo.set_mesh_value(mesh_name, FROM_SOURCE)
            self.logger.info(f"Exported {len(available_meshes)} meshes to '{export_folder_path}'")
        if data.nurbs_path:
            raise NotImplementedError("NURBS export is not implemented yet.")

        return sinfo.sorted_dict()

    def export_to_bonehub_format(
        self,
        output_root: Path,
        output_dataset_id: int,
        overwrite: bool = False,
        num_workers: int = None,
        skip_existing_subjects: bool = False,
        skip_existing_images: bool = False,
        verbose: bool = True,
    ):
        """
        Export the dataset to BoneHub's standard format.
        Args:
            output_root (Path): The root directory where the converted dataset will be saved.
            output_dataset_id (int): The dataset ID to assign to the exported dataset.
            overwrite (bool): Whether to write into an existing dataset directory. Default is False.
            num_workers (int): Subjects converted at once, each in its own process. Default is None: the number of
                CPU cores, at most DEFAULT_MAX_WORKERS.
            skip_existing_subjects (bool): only process subjects that have not been processed before (based on the existing subject_info JSON file). Default is False.
            skip_existing_images (bool): keep already exported images whose source subject is unchanged, and
                regenerate everything else. Requires overwrite=True. Default is False.
            verbose (bool): show a progress bar. Default is True.
        Returns:
            Path: The log file of this conversion.
        Raises:
            DatasetConversionError: If any subject failed (the others are still converted and saved) or the
                conversion stopped early. The log file names the failed subjects with their tracebacks.
        """
        self.dataset_info.dataset_id = output_dataset_id
        self.dataset_info.schema_version = SCHEMA_VERSION

        dataset_path = output_root / f"Dataset_{self.dataset_info.dataset_id:03d}"
        log_file_path = dataset_path / f"Dataset_{output_dataset_id:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        if not overwrite and dataset_path.exists():
            raise FileExistsError(
                f"Dataset directory '{dataset_path}' already exists. Please choose a different output_dataset_id, "
                "remove the existing directory, or pass overwrite=True."
            )
        if skip_existing_images and not overwrite:
            raise ValueError("skip_existing_images=True reuses files in an existing dataset, so it needs overwrite=True.")

        os.makedirs(dataset_path, exist_ok=True)
        log_handler = logging.FileHandler(log_file_path, mode="w")
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.INFO)
        try:
            failed_subject_ids = self._export_subjects(
                dataset_path, log_handler, num_workers, skip_existing_subjects, skip_existing_images, verbose
            )
        except Exception as exc:
            self.logger.exception(f"Conversion of {dataset_path.name} stopped by an error.")
            raise DatasetConversionError(f"Conversion of {dataset_path.name} stopped: {exc!r}.", log_file_path) from exc
        finally:
            self.logger.removeHandler(log_handler)
            log_handler.close()

        if failed_subject_ids:
            raise DatasetConversionError(
                f"{len(failed_subject_ids)} subject(s) of {dataset_path.name} failed: {_format_ids(failed_subject_ids)}.",
                log_file_path,
                tuple(failed_subject_ids),
            )
        return log_file_path

    def _export_subjects(
        self,
        dataset_path: Path,
        log_handler: logging.Handler,
        num_workers: int | None,
        skip_existing_subjects: bool,
        skip_existing_images: bool,
        verbose: bool,
    ) -> list[int]:
        """Convert every subject into dataset_path and return the ids of the subjects that failed."""
        dataset_info_path = dataset_path / f"Dataset_info_{self.dataset_info.dataset_id:03d}.json"
        subject_info_path = dataset_path / f"Subject_info_{self.dataset_info.dataset_id:03d}.json"

        self._skip_existing_images = skip_existing_images
        self._previous_sources = {}
        if skip_existing_images:
            if subject_info_path.exists():
                # Plain JSON, not SubjectInfo: the file may predate the current schema.
                with open(subject_info_path, "r") as f:
                    for subject in json.load(f):
                        if subject and subject.get("subject_id") is not None:
                            self._previous_sources[subject["subject_id"]] = subject.get("source_subject_path")
            else:
                self.logger.warning(
                    f"skip_existing_images: '{subject_info_path.name}' not found, so no image can be confirmed; all will be converted."
                )

        self.logger.info(f"Reading dataset from '{self.dataset_root}'...")
        datalist: list[DataSource] = self.custom_data_handlers.read_dataset(self.dataset_root)
        self.logger.info(f"Finished reading dataset. Found {len(datalist)} subjects.")
        self.logger.info(f"Exporting dataset to '{dataset_path}'...")

        subject_info = [None] * len(datalist)
        existing_subject_ids = set()
        if skip_existing_subjects and subject_info_path.exists():
            previous_version = None
            if dataset_info_path.exists():
                with open(dataset_info_path, "r") as f:
                    previous_version = json.load(f).get("schema_version")
            if not is_compatible_schema_version(previous_version):
                raise ValueError(
                    f"'{dataset_path.name}' was written with schema version {previous_version or '(not recorded)'}, "
                    f"but this is {SCHEMA_VERSION}, so its subjects cannot be kept. "
                    "Run again with skip_existing_subjects=False to regenerate them."
                )
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

        # Worker processes are spawned, so scripts calling this need an `if __name__ == "__main__":` guard.
        if num_workers is None:
            num_workers = min(os.cpu_count() or 4, DEFAULT_MAX_WORKERS)
        context = multiprocessing.get_context("spawn")
        log_queue = context.Queue()
        log_listener = logging.handlers.QueueListener(log_queue, log_handler)
        log_listener.start()
        failed_subject_ids = []

        def save_subject_info():
            with open(subject_info_path, "w") as f:
                json.dump(subject_info, f, indent=4)
            self.logger.info(
                f"Updated {subject_info_path.name} ({sum(x is not None for x in subject_info)}/{len(subject_info)} subjects)"
            )

        try:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=context,
                initializer=_init_subject_worker,
                initargs=(log_queue, self.logger.name),
            ) as executor:
                futures = {
                    executor.submit(self._process_subject, subject_id, data, dataset_path): subject_id - 1
                    for subject_id, data in enumerate(datalist, start=1)
                    if subject_id not in existing_subject_ids
                }
                progress_bar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Exporting Dataset_{self.dataset_info.dataset_id:03d}",
                    unit="subject",
                    disable=not verbose,
                )
                converted_count = 0
                for future in progress_bar:
                    index = futures[future]
                    subject_id = index + 1
                    try:
                        subject_info[index] = future.result()
                    except Exception as exc:
                        failed_subject_ids.append(subject_id)
                        progress_bar.set_postfix(failed=len(failed_subject_ids))
                        source = datalist[index].subject_info.source_subject_path
                        if isinstance(exc, BrokenProcessPool):
                            # A worker died without raising, and every unfinished subject ends here; no traceback helps.
                            self.logger.error(
                                f"Subject {subject_id} failed (source: '{source}'): a worker process died unexpectedly, "
                                "often from running out of memory."
                            )
                        else:
                            self.logger.exception(f"Subject {subject_id} failed (source: '{source}').")
                        continue
                    converted_count += 1
                    if converted_count % 10 == 0:  # Write every 10 subjects instead of on every one, to spare the disk.
                        save_subject_info()
                save_subject_info()  # Final write, so the last subjects and any failures are recorded too.
        finally:
            log_listener.stop()

        failed_subject_ids.sort()
        self.logger.info(
            f"Summary of {dataset_path.name}: {len(datalist)} subjects; {len(futures) - len(failed_subject_ids)} converted, "
            f"{len(existing_subject_ids)} kept from an earlier run, {len(failed_subject_ids)} failed."
        )
        if failed_subject_ids:
            self.logger.error(
                f"Failed subjects: {failed_subject_ids}. Their errors are logged above; "
                "run again with skip_existing_subjects=True to convert only the missing subjects."
            )
        return failed_subject_ids
