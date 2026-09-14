"""
SynthRAD2023 dataset.
Dataset link: https://doi.org/10.5281/zenodo.7260705

Note: only the pelvis images are converted:
    - brain subjects are skipped.
    - CBCT images (Task2) are skipped.
    - mask.nii.gz files are skipped (body outline masks, no bone information).
A Task1 training subject provides a CT and an MR image of the same patient. They are exported as two
separate BoneHub subjects, each referring to the other one in its remarks.
"""

from pathlib import Path
import shutil

import pandas as pd
from bonehub_data_schema import SubjectInfo, DatasetInfo

from .. import BaseDatasetIO, DataSource

from . import MAX_SUBJECTS_FOR_TESTING


class SynthRAD2023(BaseDatasetIO):
    """Data reader for SynthRAD2023 dataset.

    Expected structure:
    root_directory/
    ├── train/
    │   ├── Task1/                      (MR-to-CT)
    │   │   ├── brain/                  (skipped)
    │   │   └── pelvis/
    │   │       ├── 1PA001/
    │   │       │   ├── ct.nii.gz
    │   │       │   ├── mr.nii.gz
    │   │       │   └── mask.nii.gz     (skipped)
    │   │       ├── ... (other subjects)
    │   │       └── overview/
    │   │           └── 1_pelvis_train.xlsx
    │   └── Task2/                      (CBCT-to-CT)
    │       ├── brain/                  (skipped)
    │       └── pelvis/
    │           ├── 2PA001/
    │           │   ├── ct.nii.gz
    │           │   ├── cbct.nii.gz     (skipped)
    │           │   └── mask.nii.gz     (skipped)
    │           ├── ... (other subjects)
    │           └── overview/
    │               └── 2_pelvis_train.xlsx
    └── val/
        ├── Task1_val/
        │   └── Task1/
        │       ├── brain/              (skipped)
        │       └── pelvis/
        │           ├── 1PA002/
        │           │   ├── mr.nii.gz
        │           │   └── mask.nii.gz (skipped)
        │           ├── ... (other subjects)
        │           └── overview/
        │               └── 1_pelvis_val.xlsx
        └── Task2_val/                  (skipped, contains only CBCT images)
    """

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="SynthRAD2023",
            description="SynthRAD2023 Grand Challenge dataset (pelvis images only)",
            url="https://doi.org/10.5281/zenodo.7260705; https://doi.org/10.5281/zenodo.8003760",
            modality="CT; MRI",
        )
        super().__init__(dataset_root, dataset_info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_image = export_image


def get_subject_dirs(pelvis_root: Path) -> list[Path]:
    """Return the sorted subject folders of a pelvis task folder, excluding the 'overview' folder."""
    subject_dirs = sorted([d for d in pelvis_root.iterdir() if d.is_dir() and d.name != "overview"])
    if not subject_dirs:
        raise ValueError(f"No subject folders found in {pelvis_root}")
    return subject_dirs[:MAX_SUBJECTS_FOR_TESTING]


def get_source_notes(pelvis_root: Path, overview_file_name: str, sheet_name: str) -> dict[str, str]:
    """Return the per-subject notes of the dataset (e.g. 'hip implant'), keyed by source subject id."""
    metadata = pd.read_excel(pelvis_root / "overview" / overview_file_name, sheet_name=sheet_name)
    if "note" not in metadata.columns:
        return {}
    metadata = metadata.dropna(subset=["note"])
    return {str(row["ID"]): str(row["note"]).strip() for _, row in metadata.iterrows()}


def build_remarks(*parts: str | None) -> str:
    return " ".join(part for part in parts if part)


def read_dataset(dataset_root: Path) -> list[DataSource]:
    datalist = []

    # Task1 train: each subject has a CT and an MR image of the same patient.
    pelvis_root = dataset_root / "train" / "Task1" / "pelvis"
    notes = get_source_notes(pelvis_root, "1_pelvis_train.xlsx", sheet_name="MR")
    for subject_dir in get_subject_dirs(pelvis_root):
        image_paths = {"CT": subject_dir / "ct.nii.gz", "MRI": subject_dir / "mr.nii.gz"}
        for modality, image_path in image_paths.items():
            if not image_path.exists():
                raise ValueError(f"Missing {modality} image file for {subject_dir.name}")

        # Subject ids are assigned by BaseDatasetIO in the order of this list, so the id of the
        # counterpart image is known before it is appended.
        ct_subject_id = len(datalist) + 1
        mr_subject_id = ct_subject_id + 1
        note = notes.get(subject_dir.name)

        for modality, image_path in image_paths.items():
            counterpart = ("MR", mr_subject_id) if modality == "CT" else ("CT", ct_subject_id)
            data = DataSource(
                img_path=image_path,
                subject_info=SubjectInfo(
                    source_subject_path=f"train/Task1/pelvis/{subject_dir.name}/{image_path.name}",
                    imaging_modality=modality,
                    image=True,
                    remarks=build_remarks(
                        f"Task1 (MR-to-CT) pelvis case {subject_dir.name}.",
                        f"The {counterpart[0]} image of the same patient is available as subject {counterpart[1]} "
                        "of this dataset.",
                        f"Note from the dataset: {note}." if note else None,
                    ),
                ),
            )
            datalist.append(data)

    # Task1 val: each subject has only an MR image, the CT image is not publicly released.
    pelvis_root = dataset_root / "val" / "Task1_val" / "Task1" / "pelvis"
    notes = get_source_notes(pelvis_root, "1_pelvis_val.xlsx", sheet_name="MR")
    for subject_dir in get_subject_dirs(pelvis_root):
        image_path = subject_dir / "mr.nii.gz"
        if not image_path.exists():
            raise ValueError(f"Missing MRI image file for {subject_dir.name}")

        note = notes.get(subject_dir.name)
        data = DataSource(
            img_path=image_path,
            subject_info=SubjectInfo(
                source_subject_path=f"val/Task1_val/Task1/pelvis/{subject_dir.name}/{image_path.name}",
                imaging_modality="MRI",
                image=True,
                remarks=build_remarks(
                    f"Task1 (MR-to-CT) validation pelvis case {subject_dir.name}.",
                    "The CT image of the same patient is not part of the public dataset.",
                    f"Note from the dataset: {note}." if note else None,
                ),
            ),
        )
        datalist.append(data)

    # Task2 train: each subject has a CT and a CBCT image. Only the CT image is converted.
    pelvis_root = dataset_root / "train" / "Task2" / "pelvis"
    notes = get_source_notes(pelvis_root, "2_pelvis_train.xlsx", sheet_name="CBCT")
    for subject_dir in get_subject_dirs(pelvis_root):
        image_path = subject_dir / "ct.nii.gz"
        if not image_path.exists():
            raise ValueError(f"Missing CT image file for {subject_dir.name}")

        note = notes.get(subject_dir.name)
        data = DataSource(
            img_path=image_path,
            subject_info=SubjectInfo(
                source_subject_path=f"train/Task2/pelvis/{subject_dir.name}/{image_path.name}",
                imaging_modality="CT",
                image=True,
                remarks=build_remarks(
                    f"Task2 (CBCT-to-CT) pelvis case {subject_dir.name}.",
                    "The paired CBCT image of the same patient was not converted.",
                    f"Note from the dataset: {note}." if note else None,
                ),
            ),
        )
        datalist.append(data)

    if not datalist:
        raise ValueError(f"No valid subjects found in {dataset_root}")

    return datalist


def export_image(data: DataSource, output_file_path: Path):
    shutil.copyfile(data.img_path, output_file_path)
